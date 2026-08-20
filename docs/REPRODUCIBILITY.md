# Reproducibility protocol

## Model and feedback budget

The COST2100 experiments use `Nc = 32`, `Nt = 32`, and compression ratio `eta = 1/128`. Real and imaginary CSI components are encoded separately into eight scalars each, giving `dz = 16` transmitted scalars in total. Uniform scalar quantization therefore produces:

- q4: `16 x 4 = 64` feedback bits;
- q8: `16 x 8 = 128` feedback bits.

The quantizer is symmetric and uniform. Its scale is trainable, the latent input is bounded with `tanh`, and rounding uses a straight-through estimator during training. No quantizer side information is included in the feedback payload.

The symmetric quantizer uses the narrow integer range `[-(2^(q-1)-1), 2^(q-1)-1]`. Thus q4 has 15 reconstruction levels represented by fixed-width 4-bit symbols; one possible 4-bit pattern is unused. `UniformSTEQuantizer.encode_indices` and `decode_indices` expose the integer feedback interface.

## Optimization

Q-CFH-Net is trained for 500 epochs with Adam, an initial learning rate of `1e-3`, cosine annealing, batch size 256, and MSE loss. The `2 x 2` coarse grid gives four implicit seed evaluations. Validation NMSE is evaluated every 10 epochs, and the checkpoint with the lowest validation NMSE is retained.

The test split is not used for model or protocol selection. After selection, the retained checkpoint is evaluated once on the test split. Physical `rho` and `R10` are computed by the separate physical-domain evaluation script using the full frequency-domain COST2100 test channels.

## Repeated runs and statistics

The reported Q-CFH-Net, TransNet-QAT, and DCRNet-QAT rows use seeds 1, 2, and 3. The stabilized CSI-INR-FF rows use seeds 1 through 5. Every seed is retained. Table values are the arithmetic mean and population standard deviation (`ddof = 0`) across seeds.

The Q-CFH-Net per-seed values are provided in `results/qcfh_cost2100_per_seed.csv`. The complete paper comparison is in `results/cost2100_table1.csv`.

## Metric definitions

- **NMSE** is computed per CSI sample in dB and then averaged.
- **Cosine** is the magnitude of the normalized complex inner product of the flattened sparse CSI.
- **rho** is the physical-domain channel correlation after transforming the reconstructed sparse angular-delay CSI back to the frequency domain.
- **R10** is the achievable-rate ratio at 10 dB SNR relative to perfect CSI beamforming.

The `rate_ratio_mean_proxy` field produced by the training script is a sparse-domain diagnostic. Paper `R10` values come from `scripts/evaluate_physical.py` and the full frequency-domain channel data.

The released physical result files used the unsquared normalized complex correlation and the mean of per-subcarrier rate ratios. The paper prints a squared correlation and a ratio of subcarrier-averaged rates. The evaluator now reports both definitions without changing the historical keys. See `docs/METRIC_DEFINITIONS.md` for exact formulas and output names.

## DeepMIMO

The DeepMIMO Nt64 experiment uses DeepMIMO 4.0.1 and the preprocessed NPZ path described in `data/README.md`. Exact q4/q8 model configurations are in `configs/deepmimo/`. The values in `results/deepmimo_nt64.csv` are proxy reconstruction results for the fixed preprocessing protocol; the physical-rate claims in the paper remain limited to the audited COST2100 path.
