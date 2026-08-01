
from configs.settings import settings
from src.utils.logger import app_logger

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class SystemMetricsMonitor:
    """
    Monitors CPU utilization, Memory usage (RAM MB / %), and system resource health.
    """

    @staticmethod
    def get_system_health() -> dict:
        health_info = {
            "status": "healthy",
            "app_version": settings.app_version,
            "environment": settings.environment,
            "cpu_utilization_pct": 0.0,
            "memory_used_mb": 0.0,
            "memory_utilization_pct": 0.0,
        }

        if HAS_PSUTIL:
            try:
                health_info["cpu_utilization_pct"] = round(
                    psutil.cpu_percent(interval=None), 1
                )
                mem = psutil.virtual_memory()
                health_info["memory_used_mb"] = round(mem.used / (1024 * 1024), 1)
                health_info["memory_utilization_pct"] = round(mem.percent, 1)
            except Exception as e:
                app_logger.warning(f"Error reading psutil system metrics: {e}")

        return health_info


system_monitor = SystemMetricsMonitor()
