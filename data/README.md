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

`COST2100_MANIFEST.json` records the expected byte size and SHA256 of every file used by the released configurations. Verify a local copy before a long run:

```bash
python scripts/verify_data_manifest.py --data-dir data/COST2100
```

Use `--size-only` for a quick structural check when hashing the multi-gigabyte files is inconvenient.

## NPZ / DeepMIMO

For an NPZ dataset, store arrays under `h_train`, `h_val`, and `h_test`, each with shape `[N, 2, Nc, Nt]`. An optional complex `hf_test` array with shape `[N, Nt, Nf]` enables physical-domain evaluation.

The Nt64 protocol in the paper uses DeepMIMO 4.0.1, scenario `asu_campus_3p5`, 25 paths, 512 OFDM subcarriers, the first 125 selected subcarriers, 100 MHz bandwidth, and a deterministic seed-1 split of 50,000/10,000/10,000 users. Generate and preprocess it with:

```bash
pip install -e ".[deepmimo]"

python scripts/generate_deepmimo_source_npz.py \
  --scenario asu_campus_3p5 \
  --nt 64 \
  --n-subcarriers 125 \
  --ofdm-subcarriers 512 \
  --bandwidth-hz 100000000 \
  --train-samples 50000 \
  --val-samples 10000 \
  --test-samples 10000 \
  --seed 1 \
  --out data/DeepMIMO/deepmimo_asu_campus_3p5_nt64_seed1_full_source.npz

python scripts/prepare_deepmimo.py \
  --source-npz data/DeepMIMO/deepmimo_asu_campus_3p5_nt64_seed1_full_source.npz \
  --input-domain frequency_complex \
  --complex-layout n_nc_nt \
  --scenario asu_campus_3p5 \
  --nt 64 \
  --nc 32 \
  --hf-eval-subcarriers 125 \
  --normalization per_sample_complex_rms \
  --out data/DeepMIMO/deepmimo_asu_campus_3p5_ad_nc32_nt64_seed1_full_psrms.npz
```

The transformation applies an inverse FFT over frequency, truncates to the first 32 delay rows, applies the antenna FFT with `fftshift`, and performs per-sample complex-RMS normalization. The packaged metadata records the split and every conversion setting.
