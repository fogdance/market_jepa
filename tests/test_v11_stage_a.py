from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from market_jepa_v1_1.dataset import V11ContractDataset, fit_v11_shared_scaler
from market_jepa_v1_1.temporal import (build_temporal_split, training_view, validate_split,
                                      freeze_split, training_contracts, isolation_gate)
from market_jepa_v1_1.sampler import HierarchicalCommodityContractSampler, sampling_population_sha256
from market_jepa_v1_1.evaluation.stage_a import (stage_a_status, predictive_gate,
    sanity_predictions, a1_reports, stage_a_manifest, run_stage_a)
from market_jepa_v1_1.evaluation.latent_health import latent_health
from test_market_jepa_v1_1 import _make_store, _dataset_config


def dataset(n=5):
    config = _dataset_config()
    config["data"]["train_commodities"] = ["FG"]
    return V11ContractDataset(_make_store(contracts_per_commodity=n), config)


def test_stage_a_split_chronology_atomicity_and_small_commodity():
    ds = dataset()
    rem = deepcopy(ds.episode_arrays[0])
    rem.episode = SimpleNamespace(**vars(rem.episode))
    rem.episode.episode_id = 999
    ds.episode_arrays.append(rem)
    split = build_temporal_split(ds, "data")
    validate_split(split)
    item = split["commodities"]["FG"]
    assert item["probe_fit_contracts"] == ["FG01", "FG02", "FG03"]
    assert item["probe_dev_contracts"] == ["FG04"]
    assert item["probe_test_contracts"] == ["FG05"]
    small = dataset(2)
    ss = build_temporal_split(small, "data")
    assert ss["excluded_formal_commodities"] == {"FG": "insufficient_contract_groups"}
    assert len(training_view(small, ss)) == len(small)


def test_stage_a_sampler_and_scaler_exclude_test_multiple_cycles():
    ds = dataset()
    split = build_temporal_split(ds, "data")
    train = training_view(ds, split)
    sampler = HierarchicalCommodityContractSampler(train, 1000, 42)
    for cycle in range(3):
        sampler.configure_cycle(cycle, 1000)
        for index in sampler:
            episode = np.searchsorted(train.offsets, index, side="right") - 1
            assert train.episode_arrays[episode].episode.contract_uid != "FG05"
    fit_v11_shared_scaler(train, 8)
    assert len(train.scaler_selected_anchors) == 8
    assert all(r["contract_uid"] != "FG05" for r in train.scaler_selected_anchors)
    state = {"data_manifest_sha256": "data", "stage_a_temporal_split_sha256": split["sha256"],
             "sampling_population_sha256": sampling_population_sha256(train, "data"),
             "stage_a_training_contracts": [list(x) for x in training_contracts(train)],
             "stage_a_scaler_selected_anchors": train.scaler_selected_anchors}
    assert isolation_gate(state, split, ds) == "PASS"
    state["stage_a_scaler_selected_anchors"][0]["contract_uid"] = "FG05"
    with pytest.raises(ValueError, match="leakage"):
        isolation_gate(state, split, ds)


def test_stage_a_split_immutable_binding(tmp_path):
    split = build_temporal_split(dataset(), "data")
    path = tmp_path / "stage_a_temporal_split.json"
    freeze_split(path, split)
    freeze_split(path, split)
    path.write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        freeze_split(path, split)
    changed = deepcopy(split)
    changed["commodities"]["FG"]["probe_test_contracts"] = []
    with pytest.raises(ValueError, match="checksum"):
        validate_split(changed)


def test_stage_a_future_contract_sentinel_does_not_change_training_inputs():
    ds = dataset()
    split = build_temporal_split(ds, "data")
    before = training_view(ds, split)[0]
    store = deepcopy(ds.store)
    for frame in store.frames["FG"].values():
        selected = frame.contract_uid == "FG05"
        for name in ("open", "high", "low", "close", "volume", "open_interest"):
            frame.loc[selected, name] *= 10000
    changed = V11ContractDataset(store, ds.config)
    after = training_view(changed, split)[0]
    for key in before:
        if isinstance(before[key], torch.Tensor):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        elif isinstance(before[key], dict) and key != "metadata":
            for h in before[key]:
                torch.testing.assert_close(before[key][h], after[key][h], rtol=0, atol=0)


def report(low=.1, high=.3):
    value = {"point": (low + high) / 2, "ci95_low": low, "ci95_high": high}
    return {"all": value, "horizons": {str(h): dict(value) for h in (16, 64, 256)}}


def test_stage_a_status_and_h16_diagnostic_only():
    positive = report()
    positive["horizons"]["16"] = report(-.3, -.1)["all"]
    assert predictive_gate(positive) == "PASS"
    assert predictive_gate(report(-.1, .2)) == "INCONCLUSIVE"
    assert predictive_gate(report(-.2, -.1)) == "FAIL"
    assert stage_a_status("PASS", "PASS", "PASS", "PASS") == "STAGE_A_GO"
    assert stage_a_status("PASS", "PASS", "FAIL", "PASS") == "STAGE_A_NO_GO"
    assert stage_a_status("PASS", "PASS", "INCONCLUSIVE", "PASS") == "STAGE_A_INCONCLUSIVE"
    assert stage_a_status("PASS", "BLOCKED_TRIVIAL_COLLAPSE", "PASS", "PASS") == "STAGE_A_BLOCKED"


def test_stage_a_latent_health_collapse_nonfinite_and_spectrum():
    assert latent_health(np.ones((30, 8)))["status"] == "BLOCKED_TRIVIAL_COLLAPSE"
    assert latent_health(np.full((30, 8), np.nan))["status"] == "BLOCKED_TRIVIAL_COLLAPSE"
    x = np.random.default_rng(42).normal(size=(100, 8))
    health = latent_health(x)
    assert health["status"] == "PASS" and health["numerical_rank"] == 8
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    expected = (u @ u.T)[~np.eye(len(u), dtype=bool)].mean()
    assert health["mean_raw_pairwise_cosine"] == pytest.approx(expected)


def test_stage_a_mean_shuffle_fit_only_and_a1_test_only():
    rng = np.random.default_rng(42)
    records = [{"commodity": "FG", "contract_uid": str(i), "trading_date": "2020-01-01",
                "split": ["probe_fit", "probe_dev", "probe_test"][i // 10]} for i in range(30)]
    targets = rng.normal(size=(30, 3, 8))
    mean, shuffled, donors = sanity_predictions(targets, records)
    changed = targets.copy(); changed[20:] *= 500
    other = sanity_predictions(changed, records)
    np.testing.assert_array_equal(mean, other[0])
    np.testing.assert_array_equal(shuffled, other[1])
    assert donors == other[2] and all(i < 10 for i in donors)
    exports = {"target_latents": targets, "jepa": np.zeros((30, 3)), "persistence": np.ones((30, 3))}
    first = a1_reports(exports, records)[0]
    exports["jepa"][:20] = 1e8
    assert a1_reports(exports, records)[0] == first
    assert first["status"] == "PASS"


def test_stage_a_manifest_frozen_group_membership_no_replacement():
    ds = dataset()
    split = build_temporal_split(ds, "data")
    manifest = stage_a_manifest(ds, split)
    assert manifest == stage_a_manifest(ds, split)
    for r in manifest["anchors"]:
        assert r["contract_uid"] in split["commodities"]["FG"][r["split"] + "_contracts"]
    assert len({(r["contract_uid"], r["anchor_timestamp"]) for r in manifest["anchors"]}) == len(manifest["anchors"])


def test_stage_a_legacy_blocks_before_data_read(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import stage_a
    checkpoint = tmp_path / "old.pt"; checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(stage_a, "load_v11_checkpoint", lambda _: {})
    def forbidden(*a, **k):
        raise AssertionError("data must not be opened")
    monkeypatch.setattr(stage_a, "dataset_for", forbidden)
    result = run_stage_a(checkpoint, tmp_path / "audit")
    assert result["status"] == "STAGE_A_BLOCKED"
    assert "LEGACY_CHECKPOINT_NO_TEMPORAL_HOLDOUT" in result["reason"]
    assert not list(tmp_path.rglob("rb_test_consumption.json"))


def test_stage_a_scheme_a_budget_unchanged():
    from market_jepa_v1_1.config import DEFAULT_V11_CONFIG
    t = DEFAULT_V11_CONFIG["training"]
    assert (t["max_optimizer_steps"], t["warmup_optimizer_steps"]) == (250000, 12500)
    assert t["batch_size"] * t["gradient_accumulation"] == 128
    assert t["max_optimizer_steps"] * 128 == 32000000


def test_stage_a_a2_aggregates_test_only(monkeypatch):
    from market_jepa_v1_1.evaluation import runner
    monkeypatch.setattr(runner, "fit_probe", lambda *a: {"y_std": [1.] * 12})
    monkeypatch.setattr(runner, "predict_probe", lambda model, x: x)
    records = [{"commodity": "FG", "contract_uid": str(i), "trading_date": "2020-01-01",
                "split": ["probe_fit", "probe_dev", "probe_test"][i // 5]} for i in range(15)]
    exports = {k: np.ones((15, 12)) for k in ("belief", "context", "minute_raw24", "summary", "outcomes")}
    exports["belief"][:] = 1.1
    exports["summary"][:] = 2.
    first = runner._probe_results(exports, records)[1]
    exports["outcomes"][:10] = 1e6
    assert runner._probe_results(exports, records)[1] == first
    assert predictive_gate(first) == "PASS"


def test_stage_a_resume_rejects_changed_split_before_loading_weights(tmp_path):
    from market_jepa_v1_1.training import V11Trainer
    from test_market_jepa_v1_1 import _trainer_config, _TrainerDataset
    from market_jepa_v1_1.model import MarketJEPAV11
    config = _trainer_config(tmp_path)
    ds = _TrainerDataset()
    split = build_temporal_split(dataset(), "fixture")
    ds.stage_a_temporal_split = split
    trainer = V11Trainer(MarketJEPAV11(config["model"], debug=True), config, ds,
                         torch.device("cpu"), data_manifest_sha256="fixture")
    state = trainer._state()
    ds.stage_a_temporal_split = {**split, "sha256": "changed"}
    with pytest.raises(ValueError, match="changed Stage-A temporal split"):
        trainer.resume(state)


def test_stage_a_integration_no_deferred_audits_or_rb(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import stage_a, runner
    from market_jepa_v1_1.model import MarketJEPAV11
    from test_market_jepa_v1_1 import _debug_model_config
    from market_jepa_v1_1.evaluation.protocol import write_json, file_hash
    torch.set_num_threads(2)
    ds = dataset(3)
    ds.config["data"]["root"] = str(tmp_path)
    (tmp_path / "build_manifest.json").write_text("{}")
    data_hash = file_hash(tmp_path / "build_manifest.json")
    split = build_temporal_split(ds, data_hash)
    train = training_view(ds, split)
    scaler = fit_v11_shared_scaler(train, 3)
    ds.set_scaler(scaler)
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"synthetic endpoint")
    freeze_split(tmp_path / "stage_a_temporal_split.json", split)
    state = {"v11_config": ds.config, "stage_a_temporal_split": split,
             "stage_a_temporal_split_sha256": split["sha256"], "model_size": "S",
             "stage_a_training_contracts": [list(x) for x in training_contracts(train)],
             "stage_a_scaler_selected_anchors": train.scaler_selected_anchors,
             "data_manifest_sha256": data_hash, "v11_implementation_sha256": "fixture",
             "sampling_population_sha256": sampling_population_sha256(train, data_hash),
             "shared_imc_scaler": scaler.to_dict(), "training_protocol_version": "fixed_sample_budget_v1",
             "max_optimizer_steps": 250000, "effective_batch_size": 128,
             "warmup_optimizer_steps": 12500, "target_samples_seen": 32000000}
    config = _debug_model_config()
    config.update(minute_capacity=16, daily_capacity=8, current_weekly_capacity=4, history_weekly_capacity=8)
    model = MarketJEPAV11(config, debug=True)
    monkeypatch.setattr(stage_a, "load_v11_checkpoint", lambda _: state)
    monkeypatch.setattr(stage_a, "model_from_checkpoint", lambda _: model)
    for gate in ("verify_data_files", "semantic_implementation_gate", "completed_run_gate", "final_checkpoint_gate"):
        monkeypatch.setattr(stage_a, gate, lambda *a: None)
    opened = []
    def load(config, scaler, commodities, **kwargs):
        assert "RB" not in commodities and not kwargs.get("rb", False)
        opened.extend(commodities)
        return ds
    monkeypatch.setattr(stage_a, "dataset_for", load)
    monkeypatch.setattr(runner, "dataset_for", load)
    def deferred(*a, **k):
        raise AssertionError("deferred audit executed")
    monkeypatch.setattr(runner, "structure_audit", deferred)
    monkeypatch.setattr(runner, "remove_sources", deferred)
    output = tmp_path / "audit"
    result = run_stage_a(checkpoint, output, batch_size=8)
    assert result["status"] in {"STAGE_A_GO", "STAGE_A_NO_GO", "STAGE_A_INCONCLUSIVE"}, result
    for name in ("a0_latent_health.json", "a0_probe_test_latents.npz", "a1_oos_prediction.json",
                 "a2_oos_probe.json", "stage_a_summary.md", "evaluation_manifest.json"):
        assert (output / name).is_file()
    assert set(opened) == {"FG"}
    assert not list(tmp_path.rglob("rb_test_consumption.json"))
