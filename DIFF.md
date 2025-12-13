
### Differences from basic nanoGPT

Below are the main architectural changes vs. vanilla nanoGPT and how to control them via config flags (all passed into `GPTConfig`).

- **0. Tied embedding/unembedding (`tie_emb_unemb`)**
  - **What**: Shares token embedding `wte` and output head `lm_head` weights.
  - **Flag**: `tie_emb_unemb: bool` (default `False`).

- **1. RMSNorm (`rmsnorm`, `ln_learnable`)**
  - **What**: Uses `RMSNorm` instead of LayerNorm for transformer norms.
  - **Flags**:
    - `rmsnorm: bool` (default `True`) – `True`: RMSNorm, `False`: LayerNorm.
    - `ln_learnable: bool` (default `True`) – if `False`, freeze all norm weights.

- **2. VelNorm (peri-LN) (`peri_ln`, `vel_weight`, `peri_ln_learnable`)**
  - **What**: Applies `VelNorm` to attention/MLP outputs before adding to the residual (per-branch normalization).
  - **Flags**:
    - `peri_ln: bool` – enable/disable VelNorm branches.
    - `vel_weight: float` – initial VelNorm scale.
    - `peri_ln_learnable: bool` – if `False`, VelNorm weights are frozen.

- **3. Custom init std (`sigma_w`, `sigma_qk`, `sigma_emb`, `sigma_unemb`)**
  - **What**: Controls initialization scales for:
    - `sigma_w`: most Linear layers (attention + MLP).
    - `sigma_qk`: Q/K part of attention projection.
    - `sigma_emb`: token/position embeddings.
    - `sigma_unemb`: `lm_head` when untied.

- **4. Residual/branch scaling (`alpha_id`, `beta_res`)**
  - **What**: Changes residual update from `x + h` to:
    $\alpha_\text{id} \cdot x + \beta_\text{res} \cdot h$
  - **Flags**:
    - `alpha_res: float` (default `1.0`)
    - `beta_branch: float` (default `1.0`)

- **5. Orthogonal residuals (`orthogonal_residual`)**
  - **What**: Projects branch outputs to be orthogonal to the current residual before adding them.
  - **Flag**: `orthogonal_residual: bool` (default `False`).

- **6. Orthogonal transformer (`orthogonal_transformer`, `eps_tan`)**
  - **What**: Replaces `x + h` with a norm-preserving rotation in the `(x, h)` plane.
  - **Flags**:
    - `orthogonal_transformer: bool` (default `False`).
    - `eps_tan: float` (default `1e-6`).
  - **When `orthogonal_transformer=True`**:
    - Forces `orthogonal_residual=True`.
    - Disables block/final layernorms (`ln_1`, `ln_2`, `ln_f`).
    - Forces `ln_emb=True` with frozen parameters.
    - Respects `peri_ln` (VelNorm) if enabled.

