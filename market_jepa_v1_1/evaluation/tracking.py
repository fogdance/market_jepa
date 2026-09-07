from market_jepa_v1_1.wandb_logging import V11WandbLogger


def log_evaluation(output, summary, *, mode="disabled", backend=None):
    """Copy completed local results; never influences calculation or gates."""
    logger = V11WandbLogger({"enabled": mode != "disabled", "mode": mode,
        "project": "market-jepa", "group": "v1.1-evaluation", "entity": None,
        "log_every_optimizer_steps": 50}, output, backend=backend)
    logger.init(summary.get("protocol", {}), run_name="v1.1-evaluation", tags=["v1.1", "evaluation"])
    logger.finish(summary)
