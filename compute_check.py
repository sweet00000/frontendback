"""
Quick compute capability probe for a Python environment.

What it reports:
- CPU: logical/physical cores, max frequency (if available)
- RAM: total/available (via psutil if installed, else fallback)
- GPU:
    * PyTorch CUDA devices (name, capability, memory)
    * PyTorch MPS (Apple) presence
    * Optional: nvidia-smi summary if available
    * Optional: cupy if installed

Usage:
    python compute_check.py
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def try_import(module_name):
    try:
        return __import__(module_name)
    except Exception:
        return None


def cpu_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import psutil

        info["logical_cpus"] = psutil.cpu_count(logical=True)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_max_mhz"] = freq.max
            info["cpu_current_mhz"] = freq.current
    except Exception as e:
        info["psutil_note"] = f"psutil unavailable or error: {e}"
        info["logical_cpus"] = os.cpu_count()
    return info


def ram_info():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
        }
    except Exception as e:
        return {"ram_note": f"psutil unavailable or error: {e}"}


def torch_gpu_info():
    torch = try_import("torch")
    if torch is None:
        return {"torch": "not installed"}

    info = {"torch_version": torch.__version__}
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "id": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / 1e9, 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            info["cuda_devices"] = devices
        info["mps_available"] = torch.backends.mps.is_available()
    except Exception as e:
        info["torch_error"] = str(e)
    return info


def cupy_info():
    cp = try_import("cupy")
    if cp is None:
        return {"cupy": "not installed"}
    try:
        dev = cp.cuda.Device()
        return {
            "cupy_version": cp.__version__,
            "device": str(dev),
            "total_memory_gb": round(dev.mem_info[1] / 1e9, 2),
            "free_memory_gb": round(dev.mem_info[0] / 1e9, 2),
        }
    except Exception as e:
        return {"cupy_error": str(e)}


def nvidia_smi_info():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"nvidia_smi": "not found"}
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        lines = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                lines.append(
                    {
                        "name": parts[0],
                        "total_mem": parts[1],
                        "free_mem": parts[2],
                        "driver": parts[3],
                    }
                )
        return {"nvidia_smi": lines}
    except Exception as e:
        return {"nvidia_smi_error": str(e)}


def main():
    report = {
        "cpu": cpu_info(),
        "ram": ram_info(),
        "torch": torch_gpu_info(),
        "cupy": cupy_info(),
        "nvidia_smi": nvidia_smi_info(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
