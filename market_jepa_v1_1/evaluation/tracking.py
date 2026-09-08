from market_jepa_v1_1.wandb_logging import V11WandbLogger


def log_evaluation(output, summary, *, mode="disabled", backend=None):
    """Copy completed local results; never influences calculation or gates."""
    logger = V11WandbLogger({"enabled": mode != "disabled", "mode": mode,
        "project": "market-jepa", "group": "v1.1-evaluation", "entity": None,
        "log_every_optimizer_steps": 50}, output, backend=backend)
    protocol = summary.get("protocol", {})
    size, variant = protocol.get("model_size", "unknown"), protocol.get("variant", "Full")
    suffix = "" if variant == "Full" else "-" + variant.lower()
    logger.init(protocol, run_name=f"v1.1-{size}{suffix}-eval", tags=["v1.1", "evaluation", size, variant])
    logger.finish(summary)
