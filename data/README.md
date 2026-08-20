# Dataset layout

## COST2100

The repository expects the standard preprocessed COST2100 files used by CsiNet-style CSI feedback studies:

```text
data/COST2100/
  DATA_Htrainin.mat
  DATA_Hvalin.mat
  DATA_Htestin.mat
  DATA_Htrainout.mat
  DATA_Hvalout.mat
  DATA_Htestout.mat
  DATA_HtestFin_all.mat
  DATA_HtestFout_all.mat
```

The sparse angular-delay files must contain a MATLAB variable named `HT` with shape `[N, 2048]` or `[N, 2, 32, 32]`. Values follow the usual `[0, 1]` CsiNet representation; the loader subtracts `0.5` before training.

The optional full frequency-domain files used by `scripts/evaluate_physical.py` must contain a complex variable named `HF_all` with shape `[N, 4000]` or `[N, 32, 125]`.

The train, validation, and test files must remain disjoint. Checkpoint selection uses only validation NMSE.

## NPZ / DeepMIMO

For an NPZ dataset, store arrays under `h_train`, `h_val`, and `h_test`, each with shape `[N, 2, Nc, Nt]`. An optional complex `hf_test` array with shape `[N, Nt, Nf]` enables physical-domain evaluation. `scripts/generate_deepmimo_source_npz.py` and `scripts/prepare_deepmimo.py` provide the preprocessing entry points.
