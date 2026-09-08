import platform
import logging

logger = logging.getLogger(__name__)

# Supported platforms
PLATFORM_MACOS_ARM = "macos_arm"
PLATFORM_LINUX_X86 = "linux_x86"
PLATFORM_WINDOWS_X86 = "windows_x86"
PLATFORM_UNKNOWN = "unknown"


def detect_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin" and machine in ("arm64", "aarch64"):
        return PLATFORM_MACOS_ARM
    elif system == "linux" and machine in ("x86_64", "amd64"):
        return PLATFORM_LINUX_X86
    elif system == "windows" and machine in ("amd64", "x86_64"):
        return PLATFORM_WINDOWS_X86
    else:
        return PLATFORM_UNKNOWN


def get_quantization_engine(detected: str = None) -> str:
    """Return the correct PyTorch quantization backend for this platform."""
    p = detected or detect_platform()
    if p == PLATFORM_MACOS_ARM:
        return "qnnpack"
    elif p == PLATFORM_LINUX_X86:
        return "fbgemm"
    elif p == PLATFORM_WINDOWS_X86:
        return "qnnpack"
    else:
        logger.warning("Unknown platform, defaulting quantization engine to qnnpack")
        return "qnnpack"


def get_codecarbon_kwargs(detected: str = None) -> dict:
    """Return platform-appropriate codecarbon EmissionsTracker kwargs."""
    p = detected or detect_platform()

    base = {
        "measure_power_secs": 5,
        "save_to_file": False,
        "allow_multiple_runs": True,
        "log_level": "error",
    }

    if p == PLATFORM_MACOS_ARM:
        # PowerMetrics is the native Apple Silicon energy API
        base["tracking_mode"] = "process"
    elif p == PLATFORM_LINUX_X86:
        # RAPL is available on most Intel/AMD CPUs — no sudo needed if
        # /sys/class/powercap is readable. codecarbon auto-detects this.
        pass
    elif p == PLATFORM_WINDOWS_X86:
        # codecarbon falls back to CPU TDP estimation on Windows
        pass
    else:
        logger.warning("Unknown platform — codecarbon will use its own auto-detection")

    return base


def log_platform_info():
    p = detect_platform()
    labels = {
        PLATFORM_MACOS_ARM: "macOS Apple Silicon (arm64)",
        PLATFORM_LINUX_X86: "Linux x86_64",
        PLATFORM_WINDOWS_X86: "Windows x86_64",
        PLATFORM_UNKNOWN: f"Unknown ({platform.system()} / {platform.machine()})",
    }
    logger.info(f"Detected platform: {labels[p]}")
    logger.info(f"Quantization engine: {get_quantization_engine(p)}")
    return p
