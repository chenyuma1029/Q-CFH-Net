from __future__ import annotations

from .cfhnet import CFHNet
from .csi_inr_ff import CSIINRFFNet
from .qcfhnet import QCFHNet


def build_model(config: dict):
    model_cfg = config.get("model", config)
    name = model_cfg.get("name", "qcfhnet").lower()
    kwargs = {
        "nc": int(model_cfg.get("nc", config.get("data", {}).get("nc", 32))),
        "nt": int(model_cfg.get("nt", config.get("data", {}).get("nt", 32))),
        "latent_dim": int(model_cfg.get("latent_dim", 16)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "num_layers": int(model_cfg.get("num_layers", model_cfg.get("num_wire_layers", 3))),
        "coarse_grid": int(model_cfg.get("coarse_grid", 2)),
        "seed_channels": int(model_cfg.get("seed_channels", model_cfg.get("upsample_feat", 32))),
        "backbone": model_cfg.get("backbone", "wire"),
        "use_hyperfilm": bool(model_cfg.get("use_hyperfilm", True)),
        "upsampler": model_cfg.get("upsampler", "interp_conv"),
        "split_real_imag": bool(model_cfg.get("split_real_imag", False)),
    }
    if name in {"cfhnet", "cfh", "original_cfhnet"}:
        return CFHNet(**kwargs)
    if name in {"qcfhnet", "q-cfh-net", "qcfh"}:
        return QCFHNet(quantization=config.get("quantization", {}), **kwargs)
    if name in {"csi_inr_ff", "csi-inr-ff", "csiinrff"}:
        return CSIINRFFNet(quantization=config.get("quantization", {}), **kwargs)
    raise ValueError(f"Unsupported model name: {name}")
