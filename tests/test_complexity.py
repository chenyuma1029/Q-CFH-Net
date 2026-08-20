import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qcfhnet.utils.complexity import count_named_parameter_groups, estimate_forward_flops


def test_estimate_forward_flops_counts_linear():
    layer = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    estimate = estimate_forward_flops(lambda: layer(x), layer)
    assert estimate.flops == 2 * 3 * 4 * 2


def test_count_named_parameter_groups():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 3)
            self.quantizer = nn.Linear(3, 3)
            self.decoder = nn.Linear(3, 4)

    groups = count_named_parameter_groups(Tiny())
    assert groups["ue_encoder"] > 0
    assert groups["quantizer"] > 0
    assert groups["bs_decoder"] > 0
    assert groups["total"] == groups["ue_encoder"] + groups["quantizer"] + groups["bs_decoder"] + groups["other"]
