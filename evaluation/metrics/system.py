"""GPU memory and inference throughput measurement.

Provides utilities for tracking peak VRAM usage, counting model
parameters, and benchmarking inference latency with proper CUDA
synchronisation and warmup.

Usage:
    from evaluation.metrics.system import reset_vram_peak, get_peak_vram_gb
    from evaluation.metrics.system import count_trainable_params
    from evaluation.metrics.system import measure_inference_latency
"""

import time

import numpy
import torch

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def reset_vram_peak() -> None:
    """Reset peak memory statistics on all available CUDA GPUs.

    No-op if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return

    for device_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(device_idx)
        except Exception:
            pass


def get_peak_vram_gb() -> float:
    """Return the sum of peak VRAM allocated across all GPUs, in GB.

    Returns:
        Peak memory in gigabytes, rounded to 2 decimal places.
        Returns ``0.0`` if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0

    total_bytes = 0
    for device_idx in range(torch.cuda.device_count()):
        try:
            total_bytes += torch.cuda.max_memory_allocated(device_idx)
        except Exception:
            pass
    return round(total_bytes / 1e9, 2)


def count_trainable_params(model) -> dict[str, float]:
    """Count total and trainable parameters in a PyTorch model.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        A dict with keys:

        - ``"total_M"``: total parameters in millions.
        - ``"trainable_M"``: trainable parameters in millions.
        - ``"trainable_pct"``: percentage of parameters that are
          trainable.

        All values rounded to appropriate precision.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return {
        "total_M": round(total_params / 1e6, 2),
        "trainable_M": round(trainable_params / 1e6, 2),
        "trainable_pct": round(
            100.0 * trainable_params / max(total_params, 1), 4
        ),
    }


def measure_inference_latency(
    model,
    tokenizer,
    sample_prompts: list[str],
    model_config: dict,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_timed: int = 20,
    device: str = "cuda",
) -> dict[str, float]:
    """Measure mean latency and throughput for model inference.

    Runs *n_warmup* untimed forward passes followed by *n_timed*
    measured passes with proper CUDA synchronisation.

    Args:
        model: A HuggingFace causal LM on *device*.
        tokenizer: The corresponding tokenizer.
        sample_prompts: Prompts to use for benchmarking. At least
            *batch_size* prompts are required.
        model_config: Parsed ``configs/model.yaml`` dict.
        batch_size: Number of prompts per batch.
        n_warmup: Warmup iterations (not timed).
        n_timed: Timed iterations for statistics.
        device: Torch device string.

    Returns:
        A dict with:

        - ``"latency_ms_mean"``: Mean latency per batch in ms.
        - ``"latency_ms_p95"``: 95th-percentile latency in ms.
        - ``"tokens_per_sec"``: Estimated throughput (tokens/second).
    """
    # Build batched inputs with left-padding for generation.
    tokenizer.padding_side = "left"
    max_length = min(256, model_config["max_context_length"])

    if not sample_prompts:
        raise ValueError("sample_prompts must not be empty")
    prompts = list(sample_prompts[:batch_size])
    while len(prompts) < batch_size:
        prompts.append(sample_prompts[len(prompts) % len(sample_prompts)])

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Warmup passes ----------------------------------------------------
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # --- Timed passes -----------------------------------------------------
    durations_ms: list[float] = []
    cuda_available = torch.cuda.is_available()
    for _ in range(n_timed):
        if cuda_available:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if cuda_available:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    latency_ms_mean = float(numpy.mean(durations_ms))
    latency_ms_p95 = float(numpy.percentile(durations_ms, 95))
    tokens_per_sec = (batch_size * 32) / (latency_ms_mean / 1000.0)

    logger.info(
        "Latency (bs=%d): mean=%.2f ms, p95=%.2f ms, throughput=%.2f tok/s",
        batch_size,
        latency_ms_mean,
        latency_ms_p95,
        tokens_per_sec,
    )

    return {
        "latency_ms_mean": round(latency_ms_mean, 2),
        "latency_ms_p95": round(latency_ms_p95, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
