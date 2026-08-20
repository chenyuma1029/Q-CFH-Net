import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qcfhnet.datasets import Cost2100Dataset, load_cost2100_hf_all
from qcfhnet.metrics import (
    physical_rate_components,
    physical_rate_components_paper_equation,
    physical_rate_ratio,
    physical_rate_ratio_paper_equation,
    physical_rho,
    physical_rho_paper_equation,
    sparse_to_frequency,
)
from evaluate_physical import evaluate_physical, validate_hf_pairing


def test_sparse_to_frequency_layouts_match_after_transpose():
    sparse_nc_nt = torch.randn(2, 2, 4, 3)
    sparse_nt_nc = sparse_nc_nt.transpose(2, 3).contiguous()
    freq_nc_nt = sparse_to_frequency(sparse_nc_nt, internal_layout="b_2_nc_nt", nc_fft=8, n_eval=5)
    freq_nt_nc = sparse_to_frequency(sparse_nt_nc, internal_layout="b_2_nt_nc", nc_fft=8, n_eval=5)
    assert freq_nc_nt.shape == (2, 3, 5)
    assert torch.allclose(freq_nc_nt, freq_nt_nc)


def test_sparse_to_frequency_inverts_shifted_antenna_fft():
    spatial_delay = torch.randn(2, 3, 4, dtype=torch.complex64)
    angular_delay = torch.fft.fftshift(torch.fft.fft(spatial_delay, dim=1) / (3.0**0.5), dim=1)
    sparse = torch.stack([angular_delay.transpose(1, 2).real, angular_delay.transpose(1, 2).imag], dim=1)

    expected = torch.fft.fft(torch.nn.functional.pad(spatial_delay, (0, 4)), dim=-1)[..., :5]
    actual = sparse_to_frequency(
        sparse,
        internal_layout="b_2_nc_nt",
        nc_fft=8,
        n_eval=5,
        antenna_domain="angular_fftshift",
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_physical_rho_and_rate_are_one_for_identical_channels():
    sparse = torch.randn(2, 2, 4, 3)
    freq = sparse_to_frequency(sparse, internal_layout="b_2_nc_nt", nc_fft=8, n_eval=5)
    assert torch.allclose(physical_rho(freq, freq), torch.ones(2), atol=1e-6)
    assert torch.allclose(physical_rate_ratio(freq, freq, snr_db=10.0), torch.ones(2), atol=1e-6)
    assert torch.allclose(physical_rho_paper_equation(freq, freq), torch.ones(2), atol=1e-6)
    assert torch.allclose(physical_rate_ratio_paper_equation(freq, freq, snr_db=10.0), torch.ones(2), atol=1e-6)
    pred_rate, oracle_rate, ratio = physical_rate_components(freq, freq, snr_db=10.0)
    assert torch.allclose(pred_rate, oracle_rate, atol=1e-6)
    assert torch.allclose(ratio, torch.ones(2), atol=1e-6)


def test_physical_rho_matches_dcrnet_reference_formula():
    sparse = torch.randn(2, 2, 4, 3)
    freq_pred = sparse_to_frequency(sparse, internal_layout="b_2_nc_nt", nc_fft=8, n_eval=5)
    freq_true = torch.randn_like(freq_pred)

    pred_ref = torch.view_as_real(freq_pred)
    true_ref = torch.view_as_real(freq_true)
    norm_pred = torch.sqrt((pred_ref[..., 0] ** 2 + pred_ref[..., 1] ** 2).sum(dim=1))
    norm_gt = torch.sqrt((true_ref[..., 0] ** 2 + true_ref[..., 1] ** 2).sum(dim=1))
    real_cross = (pred_ref[..., 0] * true_ref[..., 0] + pred_ref[..., 1] * true_ref[..., 1]).sum(dim=1)
    imag_cross = (pred_ref[..., 0] * true_ref[..., 1] - pred_ref[..., 1] * true_ref[..., 0]).sum(dim=1)
    rho_ref = (torch.sqrt(real_cross**2 + imag_cross**2) / (norm_pred * norm_gt)).mean(dim=1)

    assert torch.allclose(physical_rho(freq_true, freq_pred), rho_ref, atol=1e-6)


def test_paper_equation_rho_uses_squared_per_subcarrier_correlation():
    freq_true = torch.tensor([[[1.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]])
    freq_pred = torch.tensor([[[0.5 + 0.0j, 0.0 + 0.0j], [0.5 + 0.0j, 1.0 + 0.0j]]])

    published = physical_rho(freq_true, freq_pred)
    paper = physical_rho_paper_equation(freq_true, freq_pred)

    assert torch.allclose(published, torch.tensor([(2.0**-0.5) / 2.0]), atol=1e-6)
    assert torch.allclose(paper, torch.tensor([0.25]), atol=1e-6)


def test_paper_equation_rate_ratio_is_ratio_of_average_rates():
    freq_true = torch.tensor([[[1.0 + 0.0j, 3.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]])
    freq_pred = torch.tensor([[[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]]])

    pred, oracle, published_ratio = physical_rate_components(freq_true, freq_pred, snr_db=0.0)
    paper_pred, paper_oracle, paper_ratio = physical_rate_components_paper_equation(
        freq_true,
        freq_pred,
        snr_db=0.0,
    )

    assert torch.allclose(pred, paper_pred)
    assert torch.allclose(oracle, paper_oracle)
    assert not torch.allclose(published_ratio, paper_ratio)
    assert torch.allclose(paper_ratio, paper_pred / paper_oracle)


def test_load_cost2100_hf_all_complex_flat(tmp_path):
    raw = (np.arange(24).reshape(2, 12) + 1j * np.arange(24).reshape(2, 12)).astype(np.complex64)
    path = tmp_path / "hf.mat"
    sio.savemat(path, {"HF_all": raw})
    h, meta = load_cost2100_hf_all(path, nt=3, n_eval=4)
    assert h.shape == (2, 3, 4)
    assert h.dtype == np.complex64
    assert meta["shape"] == [2, 3, 4]


def test_hf_pairing_rejects_wrong_cost2100_scenario_file():
    config = {"data": {"name": "cost2100", "scenario": "indoor"}}
    with pytest.raises(ValueError, match="Unverified COST2100"):
        validate_hf_pairing(
            config,
            "data/COST2100/DATA_HtestFout_all.mat",
            {},
            allow_unverified=False,
        )


def test_hf_pairing_accepts_canonical_cost2100_file():
    config = {"data": {"name": "cost2100", "scenario": "indoor"}}
    pairing = validate_hf_pairing(
        config,
        "data/COST2100/DATA_HtestFin_all.mat",
        {},
        allow_unverified=False,
    )
    assert pairing["verified"] is True


def test_physical_evaluator_emits_both_metric_conventions():
    class IdentityModel(torch.nn.Module):
        def forward(self, h):
            return {"h_hat": h}

    h = np.random.default_rng(3).normal(size=(2, 2, 2, 2)).astype(np.float32)
    h_tensor = torch.from_numpy(h)
    hf = sparse_to_frequency(
        h_tensor,
        internal_layout="b_2_nc_nt",
        nc_fft=4,
        n_eval=3,
    ).numpy()
    metrics = evaluate_physical(
        model=IdentityModel(),
        loader=DataLoader(Cost2100Dataset(h), batch_size=2),
        hf_all=hf,
        device=torch.device("cpu"),
        layout="b_2_nc_nt",
        antenna_domain="spatial",
        nc_fft=4,
        n_eval=3,
        snr_db_list=[10.0],
        pred_already_centered=True,
        sparse_layout="b_2_nc_nt",
    )

    assert np.isclose(metrics["physical_rho_mean"], 1.0)
    assert np.isclose(metrics["paper_equation_rho_mean"], 1.0)
    assert np.isclose(metrics["physical_rate_ratio_snr10_mean"], 1.0)
    assert np.isclose(metrics["paper_equation_rate_ratio_snr10_mean"], 1.0)
    assert "metric_definitions" in metrics
