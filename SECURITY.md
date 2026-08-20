# Security

## Supported version

Security fixes are applied to the latest release on the default branch.

## Artifact trust

Dataset NPZ files are loaded with NumPy pickle support disabled. PyTorch checkpoints are loaded in `weights_only` mode and must contain a model-state mapping. Use checkpoints and datasets from sources you trust, and verify large data files against the manifest before training or evaluation.

Evaluation also validates model, quantizer, dataset dimensions, scenario, and normalization settings against the configuration stored in the checkpoint. Local file paths may differ; model-defining settings may not.

## Reporting

Please report a suspected vulnerability privately to the repository owner through GitHub before opening a public issue containing exploit details.
