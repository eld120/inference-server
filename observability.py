from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("inference-server.observability")


def safe_read_file(path: Path | str) -> str | None:
    """Reads a file path and returns its stripped content, handling missing/busy devices."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        # OSError can happen (e.g., Errno 16 Device or resource busy) when GPU is suspended
        return None


def format_utc_timestamp(ts_str: str | None) -> str | None:
    """Normalizes a UTC timestamp string to ISO 8601 format with a 'Z' suffix (e.g. YYYY-MM-DDTHH:MM:SSZ)."""
    if not ts_str:
        return None
    # Split fractional seconds if present
    base = ts_str.split(".")[0]
    # Remove any existing Z or timezone offsets
    if base.endswith("Z"):
        base = base[:-1]
    if "+" in base:
        base = base.split("+")[0]
    if " " in base:
        base = base.replace(" ", "T")
    return base + "Z"


class HardwareMonitor:
    """Queries system hardware sensors from /sys/class/hwmon."""

    @staticmethod
    def get_pci_address(hwmon_dir: Path) -> str:
        """Resolves the PCI address of the device associated with this hwmon entry."""
        try:
            device_symlink = hwmon_dir / "device"
            if device_symlink.exists():
                return os.path.basename(os.path.realpath(device_symlink))
        except Exception:
            pass
        return "unknown"

    def collect(self) -> dict[str, Any]:
        """Collects current hardware metrics."""
        metrics: dict[str, Any] = {
            "cpu": {"tctl": None, "tccd1": None},
            "gpus": [],
            "ram_temps": [],
            "motherboard": {"fans": [], "temps": []},
        }

        hwmon_root = Path("/sys/class/hwmon")
        if not hwmon_root.exists():
            return metrics

        for hwmon_dir in hwmon_root.iterdir():
            if not hwmon_dir.name.startswith("hwmon"):
                continue

            name = safe_read_file(hwmon_dir / "name")
            if not name:
                continue

            # 1. CPU (k10temp)
            if name == "k10temp":
                self._collect_cpu(hwmon_dir, metrics["cpu"])

            # 2. GPU (amdgpu)
            elif name == "amdgpu":
                gpu_metrics = self._collect_gpu(hwmon_dir)
                metrics["gpus"].append(gpu_metrics)

            # 3. DDR5 RAM (spd5118)
            elif name == "spd5118":
                val = safe_read_file(hwmon_dir / "temp1_input")
                if val:
                    try:
                        metrics["ram_temps"].append(float(val) / 1000.0)
                    except ValueError:
                        pass

            # 4. Other (Motherboard fans, generic fallback)
            else:
                self._collect_generic(hwmon_dir, name, metrics["motherboard"])

        return metrics

    def _collect_cpu(self, hwmon_dir: Path, cpu_metrics: dict[str, Any]) -> None:
        # Loop through temp files to match labels
        for path in hwmon_dir.glob("temp*_input"):
            label_path = hwmon_dir / f"{path.name.replace('_input', '_label')}"
            label = safe_read_file(label_path)
            val_str = safe_read_file(path)
            if not val_str:
                continue
            try:
                temp_val = float(val_str) / 1000.0
                if label == "Tctl":
                    cpu_metrics["tctl"] = temp_val
                elif label == "Tccd1":
                    cpu_metrics["tccd1"] = temp_val
                elif not cpu_metrics["tctl"] and path.name == "temp1_input":
                    # Fallback if labels aren't populated
                    cpu_metrics["tctl"] = temp_val
            except ValueError:
                pass

    def _collect_gpu(self, hwmon_dir: Path) -> dict[str, Any]:
        pci_addr = self.get_pci_address(hwmon_dir)
        gpu_name = "Radeon Graphics"
        if "0000:03:00.0" in pci_addr:
            gpu_name = "Radeon AI PRO R9700"

        # Check if the device is suspended (D3cold)
        runtime_status_path = hwmon_dir / "device/power/runtime_status"
        status = safe_read_file(runtime_status_path) or "unknown"

        gpu_metrics: dict[str, Any] = {
            "name": gpu_name,
            "pci_addr": pci_addr,
            "status": status,
            "temps": {},
            "fan_rpm": None,
            "fan_percent": None,
            "power_w": None,
        }

        # If suspended, do not attempt to read other sysfs files (causes Device or resource busy)
        if status == "suspended":
            return gpu_metrics

        # Read temperatures
        for path in hwmon_dir.glob("temp*_input"):
            label_path = hwmon_dir / f"{path.name.replace('_input', '_label')}"
            label = safe_read_file(label_path) or path.name.replace("_input", "")
            val_str = safe_read_file(path)
            if val_str:
                try:
                    gpu_metrics["temps"][label] = float(val_str) / 1000.0
                except ValueError:
                    pass

        # Read fan RPM
        fan_val = safe_read_file(hwmon_dir / "fan1_input")
        if fan_val:
            try:
                gpu_metrics["fan_rpm"] = int(fan_val)
            except ValueError:
                pass

        # Read fan PWM/percentage
        pwm = safe_read_file(hwmon_dir / "pwm1")
        pwm_max = safe_read_file(hwmon_dir / "pwm1_max")
        if pwm and pwm_max:
            try:
                gpu_metrics["fan_percent"] = round((float(pwm) / float(pwm_max)) * 100.0, 1)
            except (ValueError, ZeroDivisionError):
                pass

        # Read power consumption
        power_val = safe_read_file(hwmon_dir / "power1_average")
        if power_val:
            try:
                # power1_average is in micro-Watts
                gpu_metrics["power_w"] = round(float(power_val) / 1000000.0, 2)
            except ValueError:
                pass

        return gpu_metrics

    def _collect_generic(self, hwmon_dir: Path, name: str, mb_metrics: dict[str, Any]) -> None:
        # Collect fans
        for path in hwmon_dir.glob("fan*_input"):
            label_path = hwmon_dir / f"{path.name.replace('_input', '_label')}"
            label = safe_read_file(label_path) or f"{name} {path.name.replace('_input', '')}"
            val_str = safe_read_file(path)
            if val_str:
                try:
                    rpm = int(val_str)
                    mb_metrics["fans"].append({"label": label, "rpm": rpm})
                except ValueError:
                    pass

        # Collect temperatures
        for path in hwmon_dir.glob("temp*_input"):
            label_path = hwmon_dir / f"{path.name.replace('_input', '_label')}"
            label = safe_read_file(label_path) or f"{name} {path.name.replace('_input', '')}"
            val_str = safe_read_file(path)
            if val_str:
                try:
                    temp_c = float(val_str) / 1000.0
                    mb_metrics["temps"].append({"label": label, "value": temp_c})
                except ValueError:
                    pass


class ObservabilityDB:
    """Manages SQLite-based recording of telemetry logs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._temp_path_to_clean = None
        self._pending_close_sessions = {}
        self._retry_lock = threading.Lock()
        self._active_retry_task = None
        if db_path is None:
            if "pytest" in sys.modules:
                import tempfile
                import uuid
                db_dir = Path(tempfile.gettempdir()) / "inference-server-test"
                db_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = db_dir / f"observability_{uuid.uuid4().hex}.db"
                self._temp_path_to_clean = self.db_path
            else:
                db_dir = Path.home() / ".config" / "inference-server"
                db_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = db_dir / "observability.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def close(self) -> None:
        if self._temp_path_to_clean and self._temp_path_to_clean.exists():
            try:
                self._temp_path_to_clean.unlink()
            except Exception:
                pass

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_tctl REAL,
                    cpu_tccd1 REAL,
                    gpu_active INTEGER,
                    gpu_temp_edge REAL,
                    gpu_temp_junction REAL,
                    gpu_temp_mem REAL,
                    gpu_fan_rpm INTEGER,
                    gpu_fan_percent REAL,
                    gpu_power_w REAL,
                    ram_temp_mean REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS serving_session (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME
                )
            """)
            # Recover/close any abnormally terminated serving sessions using ISO 8601 UTC format
            conn.execute("""
                UPDATE serving_session
                SET end_time = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE end_time IS NULL
            """)
            conn.commit()

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Extracts key metrics and logs them to the database."""
        cpu_tctl = metrics["cpu"].get("tctl")
        cpu_tccd1 = metrics["cpu"].get("tccd1")

        # Prioritize discrete R9700 GPU metrics
        gpu_active = 0
        gpu_temp_edge = None
        gpu_temp_junction = None
        gpu_temp_mem = None
        gpu_fan_rpm = None
        gpu_fan_percent = None
        gpu_power_w = None

        r9700_gpu = None
        for g in metrics["gpus"]:
            if "03:00.0" in g.get("pci_addr", "") or g.get("name") == "Radeon AI PRO R9700":
                r9700_gpu = g
                break

        # Fallback to any GPU if discrete not found
        target_gpu = r9700_gpu if r9700_gpu is not None else (metrics["gpus"][0] if metrics["gpus"] else None)

        if target_gpu:
            gpu_active = 1 if target_gpu.get("status") == "active" else 0
            if gpu_active:
                temps = target_gpu.get("temps", {})
                gpu_temp_edge = temps.get("edge") or temps.get("temp1")
                gpu_temp_junction = temps.get("junction") or temps.get("temp2")
                gpu_temp_mem = temps.get("mem") or temps.get("temp3")
                gpu_fan_rpm = target_gpu.get("fan_rpm")
                gpu_fan_percent = target_gpu.get("fan_percent")
                gpu_power_w = target_gpu.get("power_w")

        ram_temps = metrics.get("ram_temps", [])
        ram_temp_mean = sum(ram_temps) / len(ram_temps) if ram_temps else None

        with self._connection() as conn:
            conn.execute("""
                INSERT INTO telemetry_log (
                    cpu_tctl, cpu_tccd1, gpu_active,
                    gpu_temp_edge, gpu_temp_junction, gpu_temp_mem,
                    gpu_fan_rpm, gpu_fan_percent, gpu_power_w, ram_temp_mean
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cpu_tctl, cpu_tccd1, gpu_active,
                gpu_temp_edge, gpu_temp_junction, gpu_temp_mem,
                gpu_fan_rpm, gpu_fan_percent, gpu_power_w, ram_temp_mean
            ))
            conn.commit()

    def get_summary(self, window_hours: int = 24) -> dict[str, Any]:
        """Calculates min, max, average statistics over the specified hours window."""
        query = """
            SELECT 
                COUNT(*) as count,
                MIN(cpu_tctl) as cpu_tctl_min, MAX(cpu_tctl) as cpu_tctl_max, AVG(cpu_tctl) as cpu_tctl_avg,
                MIN(cpu_tccd1) as cpu_tccd1_min, MAX(cpu_tccd1) as cpu_tccd1_max, AVG(cpu_tccd1) as cpu_tccd1_avg,
                MIN(gpu_temp_edge) as gpu_edge_min, MAX(gpu_temp_edge) as gpu_edge_max, AVG(gpu_temp_edge) as gpu_edge_avg,
                MIN(gpu_temp_junction) as gpu_junc_min, MAX(gpu_temp_junction) as gpu_junc_max, AVG(gpu_temp_junction) as gpu_junc_avg,
                MIN(gpu_temp_mem) as gpu_mem_min, MAX(gpu_temp_mem) as gpu_mem_max, AVG(gpu_temp_mem) as gpu_mem_avg,
                MIN(gpu_fan_rpm) as gpu_fan_rpm_min, MAX(gpu_fan_rpm) as gpu_fan_rpm_max, AVG(gpu_fan_rpm) as gpu_fan_rpm_avg,
                MIN(gpu_fan_percent) as gpu_fan_pct_min, MAX(gpu_fan_percent) as gpu_fan_pct_max, AVG(gpu_fan_percent) as gpu_fan_pct_avg,
                MIN(gpu_power_w) as gpu_power_min, MAX(gpu_power_w) as gpu_power_max, AVG(gpu_power_w) as gpu_power_avg,
                MIN(ram_temp_mean) as ram_temp_min, MAX(ram_temp_mean) as ram_temp_max, AVG(ram_temp_mean) as ram_temp_avg
            FROM telemetry_log
            WHERE datetime(timestamp) >= datetime('now', ?)
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (f"-{window_hours} hours",))
            row = cursor.fetchone()
            if not row or row["count"] == 0:
                return {"window_hours": window_hours, "records_found": 0}

            def r(val: float | None) -> float | None:
                return round(val, 1) if val is not None else None

            return {
                "window_hours": window_hours,
                "records_found": row["count"],
                "cpu": {
                    "tctl": {"min": r(row["cpu_tctl_min"]), "max": r(row["cpu_tctl_max"]), "avg": r(row["cpu_tctl_avg"])},
                    "tccd1": {"min": r(row["cpu_tccd1_min"]), "max": r(row["cpu_tccd1_max"]), "avg": r(row["cpu_tccd1_avg"])},
                },
                "gpu": {
                    "temp_edge": {"min": r(row["gpu_edge_min"]), "max": r(row["gpu_edge_max"]), "avg": r(row["gpu_edge_avg"])},
                    "temp_junction": {"min": r(row["gpu_junc_min"]), "max": r(row["gpu_junc_max"]), "avg": r(row["gpu_junc_avg"])},
                    "temp_mem": {"min": r(row["gpu_mem_min"]), "max": r(row["gpu_mem_max"]), "avg": r(row["gpu_mem_avg"])},
                    "fan_rpm": {"min": row["gpu_fan_rpm_min"], "max": row["gpu_fan_rpm_max"], "avg": r(row["gpu_fan_rpm_avg"])},
                    "fan_percent": {"min": r(row["gpu_fan_pct_min"]), "max": r(row["gpu_fan_pct_max"]), "avg": r(row["gpu_fan_pct_avg"])},
                    "power_w": {"min": r(row["gpu_power_min"]), "max": r(row["gpu_power_max"]), "avg": r(row["gpu_power_avg"])},
                },
                "ram": {
                    "temp_mean": {"min": r(row["ram_temp_min"]), "max": r(row["ram_temp_max"]), "avg": r(row["ram_temp_avg"])},
                }
            }

    def start_serving_session(self, start_time: datetime | None = None) -> int:
        """Logs the start of a serving session and returns its ID."""
        resolved_start_time = start_time or datetime.now(timezone.utc)
        if resolved_start_time.tzinfo is None:
            resolved_start_time = resolved_start_time.replace(tzinfo=timezone.utc)
        else:
            resolved_start_time = resolved_start_time.astimezone(timezone.utc)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO serving_session (start_time) VALUES (?)",
                (resolved_start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),)
            )
            conn.commit()
            return cursor.lastrowid

    def end_serving_session(self, session_id: int, end_time: datetime | None = None) -> None:
        """Logs the end of a serving session by setting end_time to the specified timestamp."""
        resolved_end_time = end_time or datetime.now(timezone.utc)
        if resolved_end_time.tzinfo is None:
            resolved_end_time = resolved_end_time.replace(tzinfo=timezone.utc)
        else:
            resolved_end_time = resolved_end_time.astimezone(timezone.utc)
        try:
            with self._connection() as conn:
                conn.execute(
                    "UPDATE serving_session SET end_time = ? WHERE id = ?",
                    (resolved_end_time.strftime("%Y-%m-%dT%H:%M:%SZ"), session_id)
                )
                conn.commit()
            self._pending_close_sessions.pop(session_id, None)
        except Exception as exc:
            self._pending_close_sessions[session_id] = resolved_end_time
            raise exc

    def retry_pending_closes(self) -> None:
        if not self._pending_close_sessions:
            return
        acquired = self._retry_lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            if not self._pending_close_sessions:
                return
            to_retry = list(self._pending_close_sessions.items())
            for session_id, end_time in to_retry:
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                else:
                    end_time = end_time.astimezone(timezone.utc)
                try:
                    with self._connection() as conn:
                        conn.execute(
                            "UPDATE serving_session SET end_time = ? WHERE id = ?",
                            (end_time.strftime("%Y-%m-%dT%H:%M:%SZ"), session_id)
                        )
                        conn.commit()
                    self._pending_close_sessions.pop(session_id, None)
                except Exception as exc:
                    logger.exception("Failed to retry ending serving session %s in DB: %s", session_id, exc)
        finally:
            self._retry_lock.release()

    def trigger_retry(self, task_set: set | None = None) -> None:
        """Triggers a retry of pending closes in a background task if not already in progress."""
        if not self._pending_close_sessions:
            return
        if self._active_retry_task is None or self._active_retry_task.done():
            async def run_retry():
                try:
                    await asyncio.to_thread(self.retry_pending_closes)
                finally:
                    self._active_retry_task = None
            
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(run_retry())
                self._active_retry_task = task
                if task_set is not None:
                    task_set.add(task)
                    task.add_done_callback(task_set.discard)
            except RuntimeError:
                # Fallback to synchronous retry if there is no running event loop
                self.retry_pending_closes()

    def get_serving_sessions(self, window_hours: int = 24) -> list[dict[str, Any]]:
        """Returns serving sessions that overlap with or fall within the specified hours window."""
        query = """
            SELECT id, start_time, end_time
            FROM serving_session
            WHERE datetime(start_time) >= datetime('now', ?) OR datetime(end_time) >= datetime('now', ?) OR end_time IS NULL
            ORDER BY start_time DESC, id DESC
        """
        sessions = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (f"-{window_hours} hours", f"-{window_hours} hours"))
            rows = cursor.fetchall()
            for row in rows:
                start_str = format_utc_timestamp(row["start_time"])
                end_str = format_utc_timestamp(row["end_time"]) if row["end_time"] else None
                
                # Compute duration
                duration = None
                if end_str:
                    try:
                        s_clean = start_str.replace(" ", "T").replace("Z", "+00:00")
                        e_clean = end_str.replace(" ", "T").replace("Z", "+00:00")
                        s_dt = datetime.fromisoformat(s_clean)
                        e_dt = datetime.fromisoformat(e_clean)
                        duration = int((e_dt - s_dt).total_seconds())
                    except Exception:
                        pass
                
                sessions.append({
                    "id": row["id"],
                    "start_time": start_str,
                    "end_time": end_str,
                    "duration_seconds": duration,
                })
        return sessions

    def get_active_sessions(self, window_hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
        """Scans the database logs and groups consecutive active GPU rows into inference sessions."""
        query = """
            SELECT timestamp, cpu_tctl, cpu_tccd1, gpu_active, gpu_temp_edge, 
                   gpu_temp_junction, gpu_temp_mem, gpu_fan_rpm, gpu_fan_percent, gpu_power_w
            FROM telemetry_log
            WHERE datetime(timestamp) >= datetime('now', ?)
            ORDER BY timestamp ASC
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (f"-{window_hours} hours",))
            rows = cursor.fetchall()

        sessions: list[dict[str, Any]] = []
        current_session: dict[str, Any] | None = None

        def finalize_session(s: dict[str, Any]) -> dict[str, Any]:
            s_clean = s["start_time"].replace("Z", "+00:00")
            e_clean = s["end_time"].replace("Z", "+00:00")
            start_dt = datetime.fromisoformat(s_clean)
            end_dt = datetime.fromisoformat(e_clean)
            duration_sec = int((end_dt - start_dt).total_seconds())

            def avg_max(lst: list[float]) -> dict[str, float | None]:
                if not lst:
                    return {"avg": None, "max": None}
                return {"avg": round(sum(lst) / len(lst), 1), "max": round(max(lst), 1)}

            return {
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "duration_seconds": duration_sec,
                "cpu_tctl": avg_max(s["_cpu_tctl"]),
                "gpu_temp_edge": avg_max(s["_gpu_temp_edge"]),
                "gpu_temp_junction": avg_max(s["_gpu_temp_junction"]),
                "gpu_temp_mem": avg_max(s["_gpu_temp_mem"]),
                "gpu_power_w": avg_max(s["_gpu_power_w"]),
                "gpu_fan_percent": avg_max(s["_gpu_fan_percent"]),
            }

        for row in rows:
            is_active = row["gpu_active"] == 1
            ts_str = format_utc_timestamp(row["timestamp"])

            if is_active:
                if current_session is None:
                    current_session = {
                        "start_time": ts_str,
                        "end_time": ts_str,
                        "_cpu_tctl": [],
                        "_gpu_temp_edge": [],
                        "_gpu_temp_junction": [],
                        "_gpu_temp_mem": [],
                        "_gpu_power_w": [],
                        "_gpu_fan_percent": [],
                    }

                current_session["end_time"] = ts_str
                if row["cpu_tctl"] is not None:
                    current_session["_cpu_tctl"].append(row["cpu_tctl"])
                if row["gpu_temp_edge"] is not None:
                    current_session["_gpu_temp_edge"].append(row["gpu_temp_edge"])
                if row["gpu_temp_junction"] is not None:
                    current_session["_gpu_temp_junction"].append(row["gpu_temp_junction"])
                if row["gpu_temp_mem"] is not None:
                    current_session["_gpu_temp_mem"].append(row["gpu_temp_mem"])
                if row["gpu_power_w"] is not None:
                    current_session["_gpu_power_w"].append(row["gpu_power_w"])
                if row["gpu_fan_percent"] is not None:
                    current_session["_gpu_fan_percent"].append(row["gpu_fan_percent"])
            else:
                if current_session is not None:
                    # Session finished
                    sessions.append(finalize_session(current_session))
                    current_session = None

        if current_session is not None:
            sessions.append(finalize_session(current_session))

        # Return latest sessions first
        sessions.reverse()
        return sessions[:limit]


async def start_telemetry_logger(
    db: ObservabilityDB,
    task_set: set | None = None,
    interval_seconds: int = 30,
) -> None:
    """Async background task that periodically collects hardware metrics and writes them to SQLite."""
    monitor = HardwareMonitor()
    logger.info("Hardware telemetry logger started (interval: %ds)", interval_seconds)
    try:
        while True:
            try:
                metrics = await asyncio.to_thread(monitor.collect)
                await asyncio.to_thread(db.log_metrics, metrics)
                db.trigger_retry(task_set)
            except Exception as e:
                logger.error("Error in hardware telemetry logger: %s", e, exc_info=True)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("Hardware telemetry logger task cancelled")
