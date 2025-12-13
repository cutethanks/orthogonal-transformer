import math

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


out_dir = '/content/drive/MyDrive/ml_projects/post_rmsnorm/out-small-model-post-test-6layer'
max_iters = 4000
eval_interval = 400 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
early_stop_patience_steps = None
always_save_checkpoint = True
save_final_ckpt = False
early_stop_patience_steps = 400

wandb_log = False # override via command line if you like

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128 # context of up to 256 previous characters

# model parameters
n_layer = 16
n_head = 4
n_embd = 256
dropout = 0
bias = False
tie_emb_unemb = False
# normalization and architecture
rmsnorm = True
ln_learnable = True
peri_ln = False
peri_ln_learnable = False
orthogonal_residual = False
orthogonal_transformer = False
eps_tan = 1e-4

# initialization parameters
# MLP parameters
sigma_emb = math.sqrt(n_embd)
# Attention parameters
sigma_qk = 1.0
# Shared parameters
sigma_w = 1.0
# Unembedding parameters
sigma_unemb = 1.0
# VelNorm parameters
vel_weight = 1.0

# Residual/branch scaling parameters
alpha_id = 1.0
beta_res = 1.0

# training parameters
optim = 'adam'
learning_rate = 3e-4 # with baby networks can afford to go a bit higher
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay = 0
decay_lr = False
warmup_iters = 0 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model