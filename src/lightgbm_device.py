from __future__ import annotations

import os


def add_lightgbm_device_args(parser) -> None:
    parser.add_argument(
        "--lgbm-device",
        choices=["gpu", "cpu"],
        default=os.getenv("IOT_LGBM_DEVICE", "gpu").lower(),
        help="Device for LightGBM training. Defaults to GPU; use cpu to force CPU.",
    )
    parser.add_argument(
        "--gpu-platform-id",
        type=int,
        default=_optional_int_env("IOT_LGBM_GPU_PLATFORM_ID"),
        help="Optional LightGBM OpenCL platform id.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=_optional_int_env("IOT_LGBM_GPU_DEVICE_ID"),
        help="Optional LightGBM OpenCL device id.",
    )


def lightgbm_device_params(
    device: str,
    *,
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
) -> dict:
    params = {"device_type": device}
    if device == "gpu":
        if gpu_platform_id is not None:
            params["gpu_platform_id"] = gpu_platform_id
        if gpu_device_id is not None:
            params["gpu_device_id"] = gpu_device_id
    return params


def print_lightgbm_device(device: str) -> None:
    if device == "gpu":
        print("LightGBM device: GPU")
        print("If this fails, install a GPU-enabled LightGBM build and OpenCL/CUDA drivers, or pass --lgbm-device cpu.")
    else:
        print("LightGBM device: CPU")


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return int(value)
