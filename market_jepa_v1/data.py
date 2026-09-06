from __future__ import annotations

import torch


def adapt_market_batch(batch: dict, config: dict) -> dict:
    """Adapt the frozen V0 collator explicitly; metadata and future context stay out."""
    result = {key: batch[key] for key in ("minute_market", "minute_context", "targets")}
    if "persistence" in batch:
        result["persistence"] = batch["persistence"]
    for scale in ("daily", "weekly"):
        values = batch[scale]
        market_dim, context_dim = config[f"{scale}_market_dim"], config[f"{scale}_context_dim"]
        if values.shape[-1] != market_dim + context_dim:
            raise ValueError(f"{scale} upstream feature dimensions differ from the V1 config")
        lengths = batch[f"{scale}_lengths"].to(values.device)
        if lengths.shape != (values.shape[0],) or lengths.is_floating_point() or lengths.dtype == torch.bool:
            raise ValueError(f"{scale} lengths must be integer [B]")
        if torch.any((lengths < 0) | (lengths > values.shape[1])):
            raise ValueError(f"{scale} lengths exceed sequence bounds")
        mask = torch.arange(values.shape[1], device=values.device)[None] >= lengths[:, None]
        if f"{scale}_source_max" in batch and "anchor_index" in batch:
            source = batch[f"{scale}_source_max"].to(values.device)
            anchors = batch["anchor_index"].to(values.device)
            if torch.any((source > anchors[:, None]) & ~mask):
                raise ValueError(f"{scale} observation contains a future source index")
        result[f"{scale}_market"] = values[..., :market_dim]
        result[f"{scale}_context"] = values[..., market_dim:]
        result[f"{scale}_padding_mask"] = mask
    for key in ("minute_padding_mask", "target_padding_masks", "persistence_padding_masks"):
        if key in batch:
            result[key] = batch[key]
    return result
