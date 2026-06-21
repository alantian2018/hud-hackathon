from __future__ import annotations

from typing import Any

import jax


def jax_device_summary() -> dict[str, Any]:
    devices = jax.devices()
    default_backend = jax.default_backend()
    device_payloads = [
        {
            "id": int(getattr(device, "id", -1)),
            "platform": str(getattr(device, "platform", "unknown")),
            "kind": str(getattr(device, "device_kind", "unknown")),
        }
        for device in devices
    ]
    return {
        "default_backend": str(default_backend),
        "gpu_available": any(device["platform"] == "gpu" for device in device_payloads),
        "devices": device_payloads,
    }


def require_gpu_available() -> dict[str, Any]:
    summary = jax_device_summary()
    if not summary["gpu_available"]:
        raise RuntimeError(
            "JAX did not find a GPU device. Install the CUDA extra with "
            '`python3 -m pip install -e ".[gpu]"`, verify the NVIDIA driver, '
            "then rerun `jax-fleet check-gpu --require-gpu`."
        )
    return summary
