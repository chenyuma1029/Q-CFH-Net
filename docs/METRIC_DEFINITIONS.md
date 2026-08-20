# Physical metric definitions

`scripts/evaluate_physical.py` reports two explicitly labeled physical-domain conventions. Existing output keys are retained so released result files remain reproducible. New `paper_equation_*` keys evaluate the equations as printed in the paper.

Let `h_k` and `hhat_k` denote the true and reconstructed frequency-domain channel vectors on subcarrier `k`.

## Released-result convention

- `physical_rho_*` is the average of `|h_k^H hhat_k| / (||h_k|| ||hhat_k||)` over subcarriers.
- `physical_rate_ratio_snrX_*` is the average of `Rhat_k / R_k` over subcarriers.

These definitions produced the released COST2100 `rho` and `R10` values and are therefore preserved under their original names.

## Literal paper-equation convention

- `paper_equation_rho_*` is the average of `|h_k^H hhat_k|^2 / (||h_k||^2 ||hhat_k||^2)` over subcarriers.
- `paper_equation_rate_ratio_snrX_*` is `(mean_k Rhat_k) / (mean_k R_k)`.

The output JSON includes a `metric_definitions` object so downstream collection scripts can verify the convention instead of inferring it from a column name.

The sparse-domain `rate_ratio_mean_proxy` emitted during training is a diagnostic and is not a replacement for either full frequency-domain definition.
