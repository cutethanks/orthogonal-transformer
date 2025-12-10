"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import csv
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, RMSNorm

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer / optimization hyperparameters
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8
optim = 'adam'  # 'adam' or 'sgd'
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# added keys
ln_learnable = True
rmsnorm = True
peri_ln = False
peri_ln_learnable = True
orthogonal_residual = False
orthogonal_transformer = False
eps_tan = 1e-6
alpha_res = 1.0
beta_branch = 1.0
vel_weight = 1.0
ln_emb = False
tie_emb_unemb = False
sigma_w = 1.0
sigma_qk = 1.0
sigma_emb = 1.0
sigma_unemb = 1.0
# early stopping: stop if no val loss improvement for this many steps (None disables)
early_stop_patience_steps = None
# Save a final checkpoint at the end of training (in addition to any per-interval ones)
save_final_ckpt = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # initialize CSV metrics file for logging eval results
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    # If starting a fresh run, remove any stale metrics file so new logs don't append to old ones
    if init_from == 'scratch' and os.path.exists(metrics_path):
        try:
            os.remove(metrics_path)
        except OSError:
            pass
    # ensure a directory for per-step checkpoints exists
    ckpt_dir = os.path.join(out_dir, 'checkpoints') 
    os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'split', 'loss', 'lr', 'mfu'])
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
last_improve_step = 0  # for early stopping tracking
best_train_loss = 1e9  # track best training loss for early stopping

# in-memory history for logging to disk at the end of training (master_process only)
metrics_history = []         # (step, split, loss, lr, mfu)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# include normalization architecture flags so GPTConfig sees them
model_args.update(dict(rmsnorm=rmsnorm, ln_learnable=ln_learnable,
                       peri_ln=peri_ln, peri_ln_learnable=peri_ln_learnable,
                       orthogonal_residual=orthogonal_residual,
                       orthogonal_transformer=orthogonal_transformer,
                       eps_tan=eps_tan,
                       alpha_res=alpha_res, beta_branch=beta_branch, vel_weight=vel_weight,
                       tie_emb_unemb=tie_emb_unemb,
                       ln_emb=ln_emb,
                       sigma_w=sigma_w,
                       sigma_qk=sigma_qk,
                       sigma_emb=sigma_emb,
                       sigma_unemb=sigma_unemb))
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")

    # 1) Prefer a top-level ckpt.pt if it exists
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        print(f"Found top-level checkpoint: {ckpt_path}")
    else:
        # 2) Otherwise, search for the latest checkpoints/ckpt_*.pt
        ckpt_dir = os.path.join(out_dir, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"No ckpt.pt found and checkpoints directory does not exist: {ckpt_dir}"
            )
        ckpts = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("ckpt_") and f.endswith(".pt")
        ]
        if not ckpts:
            raise FileNotFoundError(
                f"No ckpt.pt found and no checkpoints matching 'ckpt_*.pt' in {ckpt_dir}"
            )
        # e.g. "ckpt_1200.pt" -> 1200
        def _step_from_name(name: str) -> int:
            return int(name.split("_")[1].split(".")[0])
        latest_name = max(ckpts, key=_step_from_name)
        ckpt_path = os.path.join(ckpt_dir, latest_name)
        print(f"Top-level ckpt.pt not found, using latest step checkpoint: {ckpt_path}")

    # Load the chosen checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'rmsnorm', 'ln_learnable',
              'peri_ln', 'peri_ln_learnable', 'orthogonal_residual', 'orthogonal_transformer', 'eps_tan',
              'alpha_res', 'beta_branch', 'vel_weight',
              'tie_emb_unemb', 'ln_emb',
              'sigma_w', 'sigma_qk', 'sigma_emb', 'sigma_unemb']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

print(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type, adam_eps, optim=optim
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def estimate_activation_norms():
    """
    Estimate per-block activation norms (L2 over hidden dim, averaged over batch and time)
    for the outputs of each transformer block on a single validation batch.

    Returns:
        norms: list of length n_layer, where norms[l] is the mean L2 norm of the
               block-l output over (batch, time).
    """
    model_ref = raw_model  # unwrapped model (handles DDP)
    was_training = model_ref.training
    model_ref.eval()

    activations = {}
    block_names = []
    handles = []

    def make_hook(name):
        def hook(module, inp, output):
            # cache block output: shape (B, T, d)
            activations[name] = output.detach()
        return hook

    # Register forward hooks on each transformer block to capture outputs
    for li, block in enumerate(model_ref.transformer.h):
        name = f'block{li}'
        block_names.append(name)
        handles.append(block.register_forward_hook(make_hook(name)))

    # Run a forward pass on a validation batch
    X, _ = get_batch('val')
    with torch.no_grad():
        _ = model_ref(X)

    # Remove hooks
    for h in handles:
        h.remove()

    if not block_names:
        if was_training:
            model_ref.train()
        return []

    # Ensure we have activations for all blocks
    if not all(name in activations for name in block_names):
        if was_training:
            model_ref.train()
        return []

    # Stack activations in order: (B, L, T, d)
    act_list = [activations[name] for name in block_names]
    acts = torch.stack(act_list, dim=1)  # (B, L, T, d)

    # Compute L2 norm along hidden dim, then average over batch and time
    # norms_sq: (B, L, T)
    norms_sq = acts.pow(2).sum(dim=-1)
    norms = norms_sq.sqrt()  # (B, L, T)
    norms_mean = norms.mean(dim=(0, 2))  # (L,)

    if was_training:
        model_ref.train()

    return norms_mean.cpu().tolist()

def estimate_grad_exponents():
    """
    Estimate gradient-based amplification/shrinking coefficients:
    - Global grad exponent between token embeddings and last block output (per-layer).
    - Per-block average log amplification (input -> output of each transformer block).

    Returns:
        lambda_grad: float or None
            Global exponent per layer, computed from per-token log amplification
            between embeddings and last block output.

        block_log_ratios: List[float]
            List of length n_layer. Entry i is the mean over (batch, time) of
            log( ||grad_in_i||^2 / ||grad_out_i||^2 ) for block i.
    """
    model_ref = raw_model  # unwrapped model (handles DDP)
    was_training = model_ref.training
    model_ref.eval()
    model_ref.zero_grad(set_to_none=True)

    handles = []

    # ---- 1) Hooks for embeddings and blocks ----
    embed_grad = {}
    block_grad_in = {}   # block name -> grad wrt block input (B,T,d)
    block_grad_out = {}  # block name -> grad wrt block output (B,T,d)
    block_names = []

    def embed_bwd_hook(module, grad_input, grad_output):
        # grad_output[0]: gradient w.r.t. embedding output (B,T,d)
        if grad_output[0] is not None:
            embed_grad['grad'] = grad_output[0].detach()

    def make_block_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            # grad_input[0]: grad wrt block input state
            # grad_output[0]: grad wrt block output state
            if grad_input[0] is not None:
                block_grad_in[name] = grad_input[0].detach()
            if grad_output[0] is not None:
                block_grad_out[name] = grad_output[0].detach()
        return hook

    # hook on token embeddings
    handles.append(model_ref.transformer.wte.register_full_backward_hook(embed_bwd_hook))
    # hook on each transformer block
    for li, block in enumerate(model_ref.transformer.h):
        name = f'block{li}'
        block_names.append(name)
        handles.append(block.register_full_backward_hook(make_block_bwd_hook(name)))

    # ---- 2) Forward/backward pass ----
    X, Y = get_batch('val')
    with ctx:
        logits, loss = model(X, Y)
    loss.backward()

    # remove hooks
    for h in handles:
        h.remove()

    eps = 1e-12
    L = float(model_ref.config.n_layer)

    # ---- 3) Global lambda_grad using per-token ratios/logs ----
    lambda_mean_log = None   # E[log lambda] / L
    lambda_log_mean = None   # log(E[lambda]) / L
    if 'grad' in embed_grad and block_names:
        g_emb = embed_grad['grad']                       # (B, T, d)
        last_block_name = block_names[-1]
        if last_block_name in block_grad_out:
            g_head = block_grad_out[last_block_name]     # (B, T, d)

            norm_emb_sq = g_emb.pow(2).sum(dim=-1) + eps     # (B, T)
            norm_head_sq = g_head.pow(2).sum(dim=-1) + eps   # (B, T)

            # per-token ratio and log-ratio
            ratio_global = norm_emb_sq / norm_head_sq                     # (B, T)
            mean_ratio_global = ratio_global.mean().item()                # E[lambda]
            lambda_log_mean = math.log(mean_ratio_global) / L

            log_ratio_global = torch.log(norm_emb_sq) - torch.log(norm_head_sq)  # (B, T)
            mean_log_global = log_ratio_global.mean().item()              # E[log lambda]
            lambda_mean_log = mean_log_global / L

    # ---- 4) Per-block amplification stats (input -> output) ----
    block_mean_logs = []   # per block: E[log lambda_l]
    block_log_means = []   # per block: log(E[lambda_l])

    if block_names and all((n in block_grad_in and n in block_grad_out) for n in block_names):
        for name in block_names:
            g_in = block_grad_in[name]                  # (B, T, d)
            g_out = block_grad_out[name]                # (B, T, d)
            norm_in_sq = g_in.pow(2).sum(dim=-1) + eps  # (B, T)
            norm_out_sq = g_out.pow(2).sum(dim=-1) + eps

            ratio_block = norm_in_sq / norm_out_sq                      # (B, T)
            mean_ratio_block = ratio_block.mean().item()                # E[lambda_l]
            block_log_means.append(math.log(mean_ratio_block))

            log_ratio_block = torch.log(norm_in_sq) - torch.log(norm_out_sq)  # (B, T)
            mean_log_block = log_ratio_block.mean().item()              # E[log lambda_l]
            block_mean_logs.append(mean_log_block)
            
    model_ref.zero_grad(set_to_none=True)
    if was_training:
        model_ref.train()

    return lambda_mean_log, lambda_log_mean, block_mean_logs, block_log_means


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
last_iter = iter_num
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        # cache metrics in memory for writing at end of training
        metrics_history.append((iter_num, 'train', float(losses['train']), lr, float(running_mfu)))
        metrics_history.append((iter_num, 'val', float(losses['val']), lr, float(running_mfu)))
        # record latest validation loss (no longer used for early stopping)
        best_val_loss = float(losses['val'])
        # early stopping is based on TRAIN loss improvements
        improved_train = losses['train'] < best_train_loss
        if improved_train:
            best_train_loss = losses['train']
            last_improve_step = iter_num
        if always_save_checkpoint:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            # also save a per-step checkpoint in a separate directory
            torch.save(checkpoint, os.path.join(ckpt_dir, f'ckpt_{iter_num}.pt'))
        # early stopping: no TRAIN loss improvement for configured number of steps
        if early_stop_patience_steps is not None:
            if (iter_num - last_improve_step) >= int(early_stop_patience_steps):
                print(f"No train loss improvement for {early_stop_patience_steps} steps. Early stopping.")
                break
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    last_iter = iter_num
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if master_process:
    # optionally save a final checkpoint of the trained model
    if save_final_ckpt:
        final_checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': last_iter,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        print(f"saving final checkpoint to {out_dir}/ckpt.pt")
        torch.save(final_checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # write cached train/val metrics to disk once training is complete
    metrics_hist_path = os.path.join(out_dir, 'metrics_history.csv')
    with open(metrics_hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'split', 'loss', 'lr', 'mfu'])
        for row in metrics_history:
            writer.writerow(list(row))

if ddp:
    destroy_process_group()
