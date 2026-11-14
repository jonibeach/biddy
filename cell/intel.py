"""Make every Intel accelerator visible to OpenVINO: CPU, GPU and NPU.

The NPU's user-space driver on this laptop comes from Canonical's
`intel-npu-driver` snap, which keeps its Level Zero driver inside the snap
instead of a system library path, so OpenVINO lists only CPU and GPU. Pointing
the Level Zero loader at it (ZE_ENABLE_ALT_DRIVERS) before OpenVINO starts
makes the NPU appear. No root needed. Imported by cell/__init__.py.
"""
import os
from pathlib import Path

SNAP_NPU = Path("/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so")


def enable_npu() -> bool:
    if "ZE_ENABLE_ALT_DRIVERS" in os.environ:
        return True
    if SNAP_NPU.exists():
        os.environ["ZE_ENABLE_ALT_DRIVERS"] = str(SNAP_NPU)
        return True
    return False


def devices() -> list[str]:
    import openvino as ov

    return ov.Core().available_devices


if __name__ == "__main__":
    enable_npu()
    print(" ".join(devices()))
