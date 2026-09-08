from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from market_jepa.data.dataset import future_outcomes as v0_outcomes
from market_jepa_v1_1.config import DEFAULT_V11_CONFIG
from market_jepa_v1_1.dataset import V11ContractDataset
from market_jepa_v1_1.evaluation.controls import LateFusionControl, SourceControl
from market_jepa_v1_1.evaluation.metrics import bootstrap, cosine_loss, predictive_status
from market_jepa_v1_1.evaluation.probe import future_outcomes, fit_probe, predict_probe, multiscale_summary
from market_jepa_v1_1.evaluation.protocol import build_manifest, validate_manifest, balanced_select, split_episodes, digest
from market_jepa_v1_1.evaluation.structure import donor_manifest, replace_channels, remove_sources
from market_jepa_v1_1.evaluation.heldout import run_heldout
from market_jepa_v1_1.training import jepa_loss_fp32, assert_all_trainable_gradients
from test_market_jepa_v1_1 import _bars, _debug_model_config, _model_batch, _make_store, _dataset_config


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_outcomes_exact_v0_parity():
    bars = _bars(600)
    frame = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume", "open_interest"])
    for h in (16, 64, 256):
        np.testing.assert_array_equal(future_outcomes(bars, 200, h), v0_outcomes(SimpleNamespace(minute=frame), 200, h))
    with pytest.raises(ValueError): future_outcomes(bars, 599, 16)


def test_manifest_deterministic_episode_split_no_replacement():
    ds = V11ContractDataset(_make_store(), _dataset_config())
    first = build_manifest(ds, "data")
    assert first == build_manifest(ds, "data")
    validate_manifest(first)
    assert len({(r["contract_uid"], r["anchor_timestamp"]) for r in first["anchors"]}) == len(first["anchors"])
    splits = {r["split"] for r in first["anchors"]}
    assert splits == {"probe_fit", "probe_dev", "probe_test"}
    corrupt = deepcopy(first); corrupt["anchors"][0]["split"] = "rb_test"
    with pytest.raises(ValueError): validate_manifest(corrupt)
    assert balanced_select({"a": [1, 2], "b": [3]}, 100) == [1, 2, 3]


def test_split_minimum_episodes():
    episodes = [SimpleNamespace(main_start=i, contract_uid=str(i), episode_id=0, key=str(i)) for i in range(2)]
    assert split_episodes(episodes)[1] == "insufficient_episodes"
    assert set(split_episodes(episodes, rb=True)[0].values()) == {"rb_dev", "rb_test"}


def test_bootstrap_equal_commodity_ratio_of_means():
    records = [{"commodity": c, "contract_uid": c, "trading_date": "2020-01-01"} for c in ["A"] * 10 + ["B"]]
    a = np.asarray([[1.] * 3] * 10 + [[3.] * 3])
    b = np.asarray([[2.] * 3] * 10 + [[6.] * 3])
    report = bootstrap(a, b, records, replicates=20)
    assert report["all"]["point"] == .5
    assert report == bootstrap(a, b, records, replicates=20)
    assert predictive_status(report) == "PASS"
    with pytest.raises(ValueError): bootstrap(a, np.zeros_like(b), records, replicates=2)


def test_ridge_preprocessing_frozen_on_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3)); y = x @ rng.normal(size=(3, 12))
    fit = np.arange(40) < 20; dev = (np.arange(40) >= 20) & (np.arange(40) < 30)
    first = fit_probe(x, y, fit, dev)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[30:] += 10000; changed_y[30:] -= 9999
    assert first == fit_probe(changed_x, changed_y, fit, dev)
    np.testing.assert_allclose(first["x_mean"], x[fit].mean(0))
    assert predict_probe(first, x).shape == (40, 12)


def test_summary_masks_ignore_padding():
    sample = {k: v[0] for k, v in _model_batch().items() if isinstance(v, torch.Tensor)}
    sample["daily_mask"][:] = True
    initial = multiscale_summary(sample)
    sample["daily_market"][:] = 9999
    np.testing.assert_array_equal(initial, multiscale_summary(sample))


def test_donors_preserve_partition_series_and_clean_target():
    records = []
    for i in range(6):
        records.append({"commodity": "FG", "series_key": "FG-01", "contract_uid": str(i),
                        "split": "probe_test", "full_minute": True, "trading_date": f"2020-01-{i+1:02d}",
                        "window_start": f"2020-01-{i+1:02d}T09:00", "anchor_timestamp": f"2020-01-{i+1:02d}T15:00"})
    mapping = donor_manifest(records)
    assert mapping == donor_manifest(records)
    assert all(p["recipient"] != p["donor"] for p in mapping["pairs"])
    recipient, donor = _model_batch(), _model_batch()
    donor["minute_market"] += 10
    changed = replace_channels(recipient, donor, "Joint")
    assert changed["target_minute_market"] is recipient["target_minute_market"]
    torch.testing.assert_close(changed["minute_market"][..., :5], recipient["minute_market"][..., :5])
    torch.testing.assert_close(changed["minute_market"][..., 5:], donor["minute_market"][..., 5:])


def test_latefusion_exact_parameter_match():
    model = LateFusionControl(DEFAULT_V11_CONFIG["model"])
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 8_318_208
    assert abs(count / 8_262_661 - 1) < .05


@pytest.mark.parametrize("variant", ["LateFusion", "MinuteOnly", "NoHistoricalWeekly"])
def test_controls_full_backward_ema_roundtrip(variant):
    config = _debug_model_config()
    model = LateFusionControl(config, debug=True) if variant == "LateFusion" else SourceControl(config, variant, debug=True)
    batch = _model_batch()
    loss, _ = jepa_loss_fp32(model(**batch)); loss.backward()
    assert_all_trainable_gradients(model)
    model.update_target()
    restored = LateFusionControl(config, debug=True) if variant == "LateFusion" else SourceControl(config, variant, debug=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval(); restored.eval()
    torch.testing.assert_close(model(**batch)["z_market"], restored(**batch)["z_market"])


def test_latefusion_independent_minute_before_compression():
    model = LateFusionControl(_debug_model_config(), debug=True).eval()
    batch = _model_batch()
    before = model(**batch, return_intermediates=True)
    batch["history_weekly_market"] += torch.randn_like(batch["history_weekly_market"])
    after = model(**batch, return_intermediates=True)
    torch.testing.assert_close(before["intermediates"]["minute_local_tokens"], after["intermediates"]["minute_local_tokens"])
    assert not torch.equal(before["z_market"], after["z_market"])
    for h in (16, 64, 256): torch.testing.assert_close(before["targets"][h], after["targets"][h])


def test_rb_test_requires_review_before_data_read(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import heldout
    frozen = {"evaluation_code_sha256": "code"}; frozen["freeze_sha256"] = digest(frozen)
    path = tmp_path / "freeze.json"
    import json
    path.write_text(json.dumps(frozen))
    monkeypatch.setattr(heldout, "evaluation_code_hash", lambda: "code")
    with pytest.raises(PermissionError):
        run_heldout(path, tmp_path / "out", partition="rb_test")
    assert not (tmp_path / "rb_test_consumption.json").exists()


@pytest.mark.parametrize("variant", ["LateFusion", "MinuteOnly", "NoHistoricalWeekly"])
def test_control_trainer_checkpoint_resume(tmp_path, variant):
    from test_market_jepa_v1_1 import _TrainerDataset, _trainer_config
    from market_jepa_v1_1.evaluation.controls import ControlTrainer, control_model
    from market_jepa_v1_1.checkpoint import load_v11_checkpoint, model_from_checkpoint
    config = _trainer_config(tmp_path)
    ds = _TrainerDataset()
    model = control_model(config["model"], variant, debug=True)
    protocol = {"variant": variant, "reference_checkpoint_sha256": "reference", "reference_sampler_num_samples": 2, "from_scratch": True}
    trainer = ControlTrainer(model, config, ds, torch.device("cpu"), control_protocol=protocol, samples_per_epoch=2, data_manifest_sha256="fixture")
    trainer.fit()
    state = load_v11_checkpoint(tmp_path / "last.pt")
    restored = ControlTrainer(control_model(config["model"], variant, debug=True), config, ds,
                               torch.device("cpu"), control_protocol=protocol, samples_per_epoch=2, data_manifest_sha256="fixture")
    restored.resume(state)
    assert restored.global_step == 1 and restored.start_epoch == 1
    assert state["evaluation_variant"] == variant
    mismatched = deepcopy(state)
    mismatched["control_provenance"]["reference_checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        restored.resume(mismatched)
    with pytest.raises(ValueError, match="control"):
        model_from_checkpoint(state)


def test_extraction_probe_integration(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import runner
    from market_jepa_v1_1.model import MarketJEPAV11
    config = _dataset_config()
    config["data"]["anchor_stride"] = 100
    ds = V11ContractDataset(_make_store(), config)
    manifest = build_manifest(ds, "fixture")
    model_config = _debug_model_config()
    model_config.update(minute_capacity=16, daily_capacity=8, current_weekly_capacity=4, history_weekly_capacity=8)
    model = MarketJEPAV11(model_config, debug=True).eval().requires_grad_(False)
    monkeypatch.setattr(runner, "dataset_for", lambda *a, **k: ds)
    exports, records = runner.extract(model, config, None, manifest["anchors"], torch.device("cpu"), diagnostics=True)
    assert exports["outcomes"].shape == (len(records), 12)
    assert exports["persistence"].shape == (len(records), 3)
    probes, report, rows, errors, test = runner._probe_results(exports, records)
    assert set(probes) == {"belief", "context", "minute_raw24", "summary", "unconditional_mean"}
    assert len(rows) == 60
    assert np.isfinite(errors["belief"]).all()
    assert report["replicates"] == 2000
    assert all(p.grad is None for p in model.parameters())


def test_evaluation_wandb_failure_is_nonfatal(tmp_path):
    from market_jepa_v1_1.evaluation.tracking import log_evaluation
    class Broken:
        def init(self, **kwargs): raise RuntimeError("network unavailable")
    log_evaluation(tmp_path, {"protocol": {}, "A1": "PASS"}, mode="online", backend=Broken())


def test_comparison_requires_rb_and_never_invents_h16_gate():
    from market_jepa_v1_1.evaluation.comparison import comparison_status
    positive = {k: {"point": .1, "ci95_low": .01, "ci95_high": .2} for k in ("all", "16", "64", "256")}
    assert comparison_status("Full", positive, positive, None, None) == "NOT_EVALUATED_RB_REQUIRED"
    assert comparison_status("Full", positive, positive, positive, positive) == "CAPACITY_BENEFIT_VS_S"
    assert comparison_status("LateFusion", positive, positive, positive, positive) == "V1_1_ARCHITECTURE_PASS"
    negative_h16 = deepcopy(positive); negative_h16["16"]["point"] = -.01
    assert comparison_status("LateFusion", negative_h16, positive, positive, positive) == "V1_1_ARCHITECTURE_PASS"
    negative_h16["16"]["ci95_high"] = -.001
    assert comparison_status("LateFusion", negative_h16, positive, positive, positive) == "FAIL"
    assert comparison_status("MinuteOnly", positive, positive, positive, positive) == "V1_1_MULTISCALE_PASS"


def test_completed_fixture_evaluation_writes_required_outputs(tmp_path, monkeypatch):
    import json
    from market_jepa_v1_1.evaluation import runner
    from market_jepa_v1_1.evaluation.protocol import file_hash, write_json
    from market_jepa_v1_1.checkpoint import v11_implementation_manifest
    from market_jepa_v1_1.dataset import fit_v11_shared_scaler
    from market_jepa_v1_1.model import MarketJEPAV11
    config = _dataset_config()
    config["data"].update(anchor_stride=200, train_commodities=["FG"], root=str(tmp_path))
    config["training"]["max_epochs"] = 1
    (tmp_path / "build_manifest.json").write_text("{}")
    data_hash = file_hash(tmp_path / "build_manifest.json")
    ds = V11ContractDataset(_make_store(), config)
    scaler = fit_v11_shared_scaler(ds, anchors_per_commodity=2)
    ds.set_scaler(scaler)
    manifest = build_manifest(ds, data_hash)
    manifest_path = tmp_path / "manifest.json"; write_json(manifest_path, manifest)
    model_config = _debug_model_config()
    model_config.update(minute_capacity=16, daily_capacity=8, current_weekly_capacity=4, history_weekly_capacity=8)
    model = MarketJEPAV11(model_config, debug=True)
    checkpoint = tmp_path / "run" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"fixture checkpoint")
    write_json(checkpoint.parent.parent / "final_summary.json", {"status": "V1_1_FORMAL_TRAINING_PASS", "checkpoint_sha256": file_hash(checkpoint)})
    write_json(checkpoint.parent.parent / "data_audit.json", {"status": "PASS", "held_out_bar_files_opened": False})
    state = {"v11_config": config, "epoch": 0, "checkpoint_selection": "fixed_budget_final",
             "sampler": {"num_samples": len(ds)},
             "shared_imc_scaler": scaler.to_dict(), "data_manifest_sha256": data_hash,
             "history": [{"epoch": 0, "train_loss": 1., "prediction_loss_h16": 1., "prediction_loss_h64": 1., "prediction_loss_h256": 1.}],
             "gradient_connectivity": {"missing": [], "nonfinite": []},
             "v11_implementation_manifest": v11_implementation_manifest(), "v11_implementation_sha256": "fixture"}
    monkeypatch.setattr(runner, "load_v11_checkpoint", lambda _: state)
    monkeypatch.setattr(runner, "model_from_checkpoint", lambda _: model)
    monkeypatch.setattr(runner, "verify_data_files", lambda *a: {"status": "FIXTURE"})
    monkeypatch.setattr(runner, "dataset_for", lambda *a, **k: ds)
    output = tmp_path / "evaluation"
    summary = runner.evaluate(checkpoint, manifest_path, output)
    assert summary["rb_test_consumed"] is False
    for name in ("evaluation_manifest.json", "a1_jepa_vs_persistence.csv", "a1_bootstrap.json",
                 "a2_probe_hyperparams.json", "a2_probe_results.csv", "a2_baselines.csv", "a2_bootstrap.json",
                 "a3_structure_results.csv", "a3_donor_manifest.json", "a3_bootstrap.json",
                 "rb_results.csv", "rb_bootstrap.json", "summary.json", "summary.md"):
        assert (output / name).is_file()


def test_data_checksum_gate_never_opens_unrequested_rb(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import protocol
    files = ["contract_episodes.csv", "FG/FG_1m.csv", "FG/FG_1d.csv", "FG/FG_1w.csv"]
    protocol.write_json(tmp_path / "build_manifest.json", {"files": [{"path": name, "sha256": "ok"} for name in files]})
    opened = []
    def checksum(path):
        opened.append(str(path)); return "ok"
    monkeypatch.setattr(protocol, "file_hash", checksum)
    assert protocol.verify_data_files(tmp_path, ["FG"])["status"] == "PASS"
    assert not any("RB" in name for name in opened)


def test_persistence_exact_horizon_slices_and_padding():
    from market_jepa_v1_1.evaluation.runner import horizon_persistence
    market = torch.arange(512 * 9, dtype=torch.float32).reshape(1, 512, 9)
    validity = torch.ones_like(market, dtype=torch.bool)
    mask = torch.arange(512)[None] < 500
    calls = []
    def encode(m, v, pad):
        calls.append((m.clone(), v.clone(), pad.clone()))
        return m.sum(1)
    values = horizon_persistence(SimpleNamespace(target_minute=encode), {
        "minute_market": market, "minute_imc_validity": validity, "minute_mask": mask})
    for h, (m, v, pad) in zip((16, 64, 256), calls):
        assert m.shape[1] == h
        torch.testing.assert_close(m, market[:, -h:])
        torch.testing.assert_close(v, validity[:, -h:])
        torch.testing.assert_close(pad, mask[:, -h:])
        torch.testing.assert_close(values[h], market[:, -h:].sum(1))


def test_latefusion_uses_learned_cls_exactly():
    model = LateFusionControl(_debug_model_config(), debug=True).eval()
    batch = _model_batch()
    context = model.minute_local.context_projection(batch["minute_context"].masked_fill(batch["minute_mask"].unsqueeze(-1), 0))
    encoded, _ = model.minute_local.market_core.encode_tokens(batch["minute_market"], batch["minute_imc_validity"], batch["minute_mask"], additive_context=context)
    result = model(**batch, return_intermediates=True)
    torch.testing.assert_close(result["intermediates"]["minute_vector"], encoded[:, 0])


def test_matched_control_provenance_and_semantic_hashes():
    from market_jepa_v1_1.evaluation.protocol import matched_control_gate, semantic_implementation_gate, control_semantic_hashes
    from market_jepa_v1_1.checkpoint import v11_implementation_manifest
    protocol = {"variant": "LateFusion", "from_scratch": True, "reference_checkpoint_sha256": "A", "reference_sampler_num_samples": 100}
    provenance = {k: protocol[k] for k in ("reference_checkpoint_sha256", "reference_sampler_num_samples")}
    provenance["control_protocol_sha256"] = digest(protocol)
    state = {"evaluation_variant": "LateFusion", "control_protocol": protocol, "control_provenance": provenance,
             "control_semantic_hashes": control_semantic_hashes(), "v11_implementation_manifest": v11_implementation_manifest(), "sampler": {"num_samples": 100}}
    semantic_implementation_gate(state)
    control = {"control_provenance": provenance, "sampler_num_samples": 100}
    matched_control_gate(control, {"checkpoint_sha256": "A", "sampler_num_samples": 100})
    with pytest.raises(ValueError): matched_control_gate(control, {"checkpoint_sha256": "B", "sampler_num_samples": 100})
    with pytest.raises(ValueError): matched_control_gate(control, {"checkpoint_sha256": "A", "sampler_num_samples": 99})
    for name in state["control_semantic_hashes"]:
        changed = deepcopy(state); changed["control_semantic_hashes"][name] = "changed"
        with pytest.raises(ValueError, match="semantic"): semantic_implementation_gate(changed)


def test_campaign_exact_cohort_canonical_paths_and_consumption(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import campaign
    from market_jepa_v1_1.evaluation.protocol import write_json
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", tmp_path)
    plan = tmp_path / "campaign_plan.json"; frozen = tmp_path / "rb_freeze.json"
    write_json(plan, {"expected_models": [{"model_size": s, "variant": "Full"} for s in ("S", "M", "L", "XL")]})
    expected, _ = campaign.campaign_plan(plan, frozen)
    campaign.exact_cohort(expected, expected)
    with pytest.raises(ValueError): campaign.exact_cohort(expected[:1], expected)
    with pytest.raises(ValueError): campaign.exact_cohort(expected + expected[:1], expected)
    with pytest.raises(ValueError): campaign.campaign_plan(plan, tmp_path / "elsewhere" / "rb_freeze.json")
    write_json(tmp_path / "rb_test_consumption.json", {"rb_test_consumed": True})
    with pytest.raises(FileExistsError): campaign.campaign_plan(plan, frozen)


def test_history_negative_transfer_and_minuteonly_uncertain_rb():
    from market_jepa_v1_1.evaluation.comparison import comparison_status
    positive = {k: {"point": .1, "ci95_low": .01, "ci95_high": .2} for k in ("all", "16", "64", "256")}
    negative = deepcopy(positive); negative["all"] = {"point": -.1, "ci95_low": -.2, "ci95_high": -.01}
    for i in range(4):
        metrics = [positive] * 4; metrics[i] = negative
        assert comparison_status("NoHistoricalWeekly", *metrics) == "NOT_PASS"
    uncertain = deepcopy(negative); uncertain["all"]["ci95_high"] = .02
    assert comparison_status("MinuteOnly", positive, positive, positive, uncertain) == "V1_1_MULTISCALE_PASS"


def test_scaling_requires_complete_adjacent_improvements():
    from market_jepa_v1_1.evaluation.comparison import scaling_sequence_status
    positive = {"all": {"point": .1}}
    pair = {k: positive for k in ("delta_gain", "delta_skill", "delta_rb_gain", "delta_rb_skill")}
    pairs = [deepcopy(pair) for _ in range(3)]
    assert scaling_sequence_status(["S", "M", "L", "XL"], pairs) == "V1_1_SCALING_SUPPORTED"
    pairs[1]["delta_rb_skill"] = {"all": {"point": -.1}}
    assert scaling_sequence_status(["S", "M", "L", "XL"], pairs) == "IN_DOMAIN_CAPACITY_SCALING_ONLY"
    assert scaling_sequence_status(["S", "M"], pairs[:1]) == "INCOMPLETE_SCALING_COHORT"


def test_structure_batching_matches_single_recipient(monkeypatch):
    from market_jepa_v1_1.evaluation import runner
    from market_jepa_v1_1.model import MarketJEPAV11
    batch = _model_batch()
    samples = []
    for i in range(4):
        sample = {}
        for key, value in batch.items():
            sample[key] = {h: t[0].clone() for h, t in value.items()} if isinstance(value, dict) else value[0].clone()
        sample["minute_market"] += i * .1
        sample["metadata"] = {}
        samples.append(sample)
    records = [{"commodity": "FG", "i": i} for i in range(4)]
    mapping = {"pairs": [{"recipient": 0, "donor": 2}, {"recipient": 1, "donor": 3}]}
    monkeypatch.setattr(runner, "donor_manifest", lambda _: mapping)
    monkeypatch.setattr(runner, "dataset_for", lambda *a, **k: samples)
    monkeypatch.setattr(runner, "resolve_records", lambda ds, rec: [(r["i"], None, None) for r in rec])
    captured = []
    def record_errors(a, b, recipients, **kwargs):
        captured.append((np.asarray(a), np.asarray(b), recipients))
        return {}
    monkeypatch.setattr(runner, "bootstrap", record_errors)
    model = MarketJEPAV11(_debug_model_config(), debug=True).eval()
    runner.structure_audit(model, {}, None, records, torch.device("cpu"), batch_size=1)
    first = captured.copy(); captured.clear()
    runner.structure_audit(model, {}, None, records, torch.device("cpu"), batch_size=2)
    for before, after in zip(first, captured):
        np.testing.assert_allclose(before[0], after[0], atol=1e-6)
        np.testing.assert_allclose(before[1], after[1], atol=1e-6)
        assert before[2] == after[2]


@pytest.mark.parametrize("size,variant,name", [("M", "Full", "v1.1-M-eval"), ("S", "LateFusion", "v1.1-S-latefusion-eval")])
def test_evaluation_wandb_names(tmp_path, size, variant, name):
    from market_jepa_v1_1.evaluation.tracking import log_evaluation
    class Run:
        id = "fixture"
        summary = {}
        def define_metric(self, *a, **k): pass
        def finish(self, **k): pass
    class Backend:
        def init(self, **kwargs):
            self.kwargs = kwargs
            return Run()
    backend = Backend()
    log_evaluation(tmp_path, {"protocol": {"model_size": size, "variant": variant}}, mode="offline", backend=backend)
    assert backend.kwargs["name"] == name


def test_rb_freeze_and_one_shot_integration(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import heldout, campaign
    from market_jepa_v1_1.evaluation.protocol import write_json, file_hash
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", tmp_path)
    monkeypatch.setattr(heldout, "evaluation_code_hash", lambda: "code")
    for name in ("final_checkpoint_gate", "completed_run_gate", "semantic_implementation_gate", "verify_data_files"):
        monkeypatch.setattr(heldout, name, lambda *a, **k: None)
    monkeypatch.setattr(heldout, "validate_manifest", lambda _: None)
    # Explicitly reviewed minimal test campaign, not the seven-model production template.
    plan = tmp_path / "campaign_plan.json"
    write_json(plan, {"expected_models": [{"model_size": "S", "variant": "Full"}]})
    checkpoint = tmp_path / "last.pt"; checkpoint.write_bytes(b"fixture")
    write_json(tmp_path / "build_manifest.json", {})
    data_hash = file_hash(tmp_path / "build_manifest.json")
    state = {"model_size": "S", "sampler": {"num_samples": 10}, "v11_config": {"data": {"root": str(tmp_path)}}}
    monkeypatch.setattr(heldout, "load_v11_checkpoint", lambda _: state)
    directory = tmp_path / "train_eval"
    protocol = {"model_size": "S", "variant": "Full", "sampler_num_samples": 10,
                "checkpoint_sha256": file_hash(checkpoint), "evaluation_code_sha256": "code",
                "manifest_sha256": "train", "epochs": 50, "scaler_sha256": "scaler", "data_manifest_sha256": data_hash}
    write_json(directory / "summary.json", {"protocol": protocol})
    write_json(directory / "evaluation_protocol.json", protocol)
    write_json(directory / "a2_probe_hyperparams.json", {})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"rb_inventory": True, "sha256": "rb", "data_manifest_sha256": data_hash,
                          "anchors": [{"split": "rb_test"}]})
    frozen_path = tmp_path / "rb_freeze.json"
    frozen = heldout.freeze([checkpoint], [directory], manifest, frozen_path, plan_path=plan)
    approval = tmp_path / "approval.json"
    write_json(approval, {"freeze_sha256": frozen["freeze_sha256"], "human_review_pass": True, "tests_pass": True})
    original = (directory / "a2_probe_hyperparams.json").read_text()
    write_json(directory / "a2_probe_hyperparams.json", {"changed": True})
    with pytest.raises(ValueError, match="probe changed"):
        heldout.run_heldout(frozen_path, tmp_path / "out", partition="rb_test", approval=approval)
    assert not (tmp_path / "rb_test_consumption.json").exists()
    (directory / "a2_probe_hyperparams.json").write_text(original)
    def fail_model(*a): raise RuntimeError("fixture inference failure")
    monkeypatch.setattr(heldout, "model_from_checkpoint", fail_model)
    with pytest.raises(RuntimeError, match="fixture inference"):
        heldout.run_heldout(frozen_path, tmp_path / "out", partition="rb_test", approval=approval)
    assert (tmp_path / "rb_test_consumption.json").exists()
    with pytest.raises(FileExistsError, match="consumed"):
        heldout.run_heldout(frozen_path, tmp_path / "another_output", partition="rb_test", approval=approval)


def test_scaling_report_adjacent_pairs_not_only_vs_s(tmp_path, monkeypatch):
    from market_jepa_v1_1.evaluation import comparison
    from market_jepa_v1_1.evaluation.protocol import write_json
    def paired(a, b, c, d, records):
        delta = float(np.mean(c / d) - np.mean(a / b))
        return {k: {"point": delta, "ci95_low": delta - .01, "ci95_high": delta + .01} for k in ("all", "16", "64", "256")}
    monkeypatch.setattr(comparison, "paired_improvement", paired)
    runs, rb_runs = [], []
    records = [{"commodity": "FG", "contract_uid": "FG1", "trading_date": "2020-01-01"}]
    for i, (size, gain) in enumerate(zip(("S", "M", "L", "XL"), (.1, .5, .3, .2))):
        directory, rb = tmp_path / size, tmp_path / (size + "_rb")
        protocol = {"manifest_sha256": "same", "epochs": 50, "scaler_sha256": "same", "evaluation_code_sha256": "same",
                    "data_manifest_sha256": "same", "training_protocol": {"batch_size": 64, "gradient_accumulation": 2},
                    "model_size": size, "variant": "Full", "trainable_params": i + 1, "checkpoint_sha256": size}
        write_json(directory / "summary.json", {"protocol": protocol, "Gain_ALL": {"point": gain}, "Skill_ALL": {"point": gain}, "RB": "NOT_RUN"})
        write_json(directory / "paired_records.json", records)
        write_json(rb / "paired_records.json", records)
        values = {"jepa": np.full((1, 3), 1 - gain), "persistence": np.ones((1, 3)),
                  "belief_error": np.full((1, 12), 1 - gain), "summary_error": np.ones((1, 12))}
        np.savez(directory / "paired_errors.npz", **values, test_mask=np.ones(1, dtype=bool))
        np.savez(rb / "paired_errors.npz", **values)
        write_json(rb / "rb_bootstrap.json", {"partition": "rb_test", "checkpoint_sha256": size,
            "RB_Gain": {"all": {"point": gain}}, "RB_Skill": {"all": {"point": gain}}, "status": "fixture"})
        runs.append(directory); rb_runs.append(rb)
    result = comparison.compare_runs(runs, tmp_path / "report", rb_runs)
    assert [(p["from"], p["to"]) for p in result["adjacent_pairs"]] == [("S", "M"), ("M", "L"), ("L", "XL")]
    assert result["scaling_status"] == "SCALING_MIXED"
    assert all(p["formal_status"] == "CAPACITY_BENEFIT_VS_S" for p in result["paired_comparisons"])
