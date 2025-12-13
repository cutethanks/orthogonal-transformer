### Orthogonal Transformer: Quick Usage

- **Prerequisites**  
  Prepare data in `data/` as in nanoGPT (e.g. `data/shakespeare_char/{train.bin,val.bin,meta.pkl}`). For the Shakespeare char dataset:

```bash
python data/shakespeare_char/prepare.py
```

- **Main training script**  
  Training is driven by `train.py` and Python config files in `config/`.

- **Orthogonal transformer config**  
  Use `config/train_orthogonal_transformer.py`, which enables the new orthogonal update via:
  - `orthogonal_transformer = True` in the config (passed into `GPTConfig`).

- **Launch training with orthogonal updates**  

```bash
python train.py config/train_orthogonal_transformer.py
```

- **Other example config**  
  - `config/train_shakespeare_char.py`: a small baseline character-level GPT config.

- **Hyperparameter sweeps**  
  The script `sweep_params.py` is a small launcher that:
  - Defines a list of hyperparameter values to try (e.g. different `sigma_w` or other scalings).
  - Spawns multiple `python train.py ...` runs (up to `MAX_PARALLEL` at once).
  Use it when you want to automatically compare several configurations instead of starting each run manually.


