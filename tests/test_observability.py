from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from observability import HardwareMonitor, ObservabilityDB


@pytest.fixture
def mock_sysfs_suspended(tmp_path):
    """Creates a mock sysfs directory structure where the GPU is suspended."""
    hwmon_root = tmp_path / "sys" / "class" / "hwmon"
    hwmon_root.mkdir(parents=True)

    # CPU (k10temp)
    cpu_dir = hwmon_root / "hwmon2"
    cpu_dir.mkdir()
    (cpu_dir / "name").write_text("k10temp\n")
    (cpu_dir / "temp1_input").write_text("43000\n")
    (cpu_dir / "temp1_label").write_text("Tctl\n")
    (cpu_dir / "temp3_input").write_text("32000\n")
    (cpu_dir / "temp3_label").write_text("Tccd1\n")

    # GPU (amdgpu) - Suspended
    gpu_dir = hwmon_root / "hwmon0"
    gpu_dir.mkdir()
    (gpu_dir / "name").write_text("amdgpu\n")
    
    # Mock device symlink structure
    device_power_dir = gpu_dir / "device" / "power"
    device_power_dir.mkdir(parents=True)
    (device_power_dir / "runtime_status").write_text("suspended\n")

    # RAM (spd5118)
    ram_dir = hwmon_root / "hwmon4"
    ram_dir.mkdir()
    (ram_dir / "name").write_text("spd5118\n")
    (ram_dir / "temp1_input").write_text("31500\n")

    return hwmon_root


@pytest.fixture
def mock_sysfs_active(tmp_path):
    """Creates a mock sysfs directory structure where the GPU is active."""
    hwmon_root = tmp_path / "sys" / "class" / "hwmon"
    hwmon_root.mkdir(parents=True)

    # CPU (k10temp)
    cpu_dir = hwmon_root / "hwmon2"
    cpu_dir.mkdir()
    (cpu_dir / "name").write_text("k10temp\n")
    (cpu_dir / "temp1_input").write_text("45000\n")
    (cpu_dir / "temp1_label").write_text("Tctl\n")

    # GPU (amdgpu) - Active
    gpu_dir = hwmon_root / "hwmon0"
    gpu_dir.mkdir()
    (gpu_dir / "name").write_text("amdgpu\n")
    device_power_dir = gpu_dir / "device" / "power"
    device_power_dir.mkdir(parents=True)
    (device_power_dir / "runtime_status").write_text("active\n")

    # Sensors
    (gpu_dir / "temp1_input").write_text("38000\n")
    (gpu_dir / "temp1_label").write_text("edge\n")
    (gpu_dir / "temp2_input").write_text("45000\n")
    (gpu_dir / "temp2_label").write_text("junction\n")
    (gpu_dir / "fan1_input").write_text("1200\n")
    (gpu_dir / "pwm1").write_text("127\n")
    (gpu_dir / "pwm1_max").write_text("255\n")
    (gpu_dir / "power1_average").write_text("25000000\n") # 25W

    return hwmon_root


@patch("observability.Path")
def test_hardware_monitor_suspended(mock_path_class, mock_sysfs_suspended):
    # Route Path queries to use the temporary mock dir
    mock_path_class.return_value = mock_sysfs_suspended
    # Mock Path.exists and iterdir to look at the mock directory instead of host /sys
    mock_path_class.home.return_value = mock_sysfs_suspended

    with patch("observability.os.path.realpath") as mock_realpath:
        # Mock PCI Address resolution
        mock_realpath.return_value = "/sys/devices/pci0000:00/0000:03:00.0"
        
        monitor = HardwareMonitor()
        # Redirect the root path in collection
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "iterdir", return_value=list(mock_sysfs_suspended.iterdir())):
            metrics = monitor.collect()

        assert metrics["cpu"]["tctl"] == 43.0
        assert metrics["cpu"]["tccd1"] == 32.0
        assert len(metrics["gpus"]) == 1
        gpu = metrics["gpus"][0]
        assert gpu["name"] == "Radeon AI PRO R9700"
        assert gpu["status"] == "suspended"
        assert gpu["power_w"] is None  # Suspended skips reading active properties
        assert metrics["ram_temps"] == [31.5]


@patch("observability.Path")
def test_hardware_monitor_active(mock_path_class, mock_sysfs_active):
    mock_path_class.return_value = mock_sysfs_active
    mock_path_class.home.return_value = mock_sysfs_active

    with patch("observability.os.path.realpath") as mock_realpath:
        mock_realpath.return_value = "/sys/devices/pci0000:00/0000:03:00.0"
        
        monitor = HardwareMonitor()
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "iterdir", return_value=list(mock_sysfs_active.iterdir())):
            metrics = monitor.collect()

        assert metrics["cpu"]["tctl"] == 45.0
        assert len(metrics["gpus"]) == 1
        gpu = metrics["gpus"][0]
        assert gpu["name"] == "Radeon AI PRO R9700"
        assert gpu["status"] == "active"
        assert gpu["temps"]["edge"] == 38.0
        assert gpu["temps"]["junction"] == 45.0
        assert gpu["fan_rpm"] == 1200
        assert gpu["fan_percent"] == 49.8 # 127/255 * 100
        assert gpu["power_w"] == 25.0


def test_observability_db_operations(tmp_path):
    db_file = tmp_path / "test_observability.db"
    db = ObservabilityDB(db_path=db_file)

    # Test initial state
    summary = db.get_summary(window_hours=1)
    assert summary["records_found"] == 0

    # Insert mock metrics
    metrics = {
        "cpu": {"tctl": 50.0, "tccd1": 40.0},
        "gpus": [{
            "name": "Radeon AI PRO R9700",
            "pci_addr": "0000:03:00.0",
            "status": "active",
            "temps": {"edge": 40.0, "junction": 50.0, "mem": 45.0},
            "fan_rpm": 1000,
            "fan_percent": 30.0,
            "power_w": 30.0,
        }],
        "ram_temps": [30.0, 32.0],
    }

    db.log_metrics(metrics)

    summary = db.get_summary(window_hours=1)
    assert summary["records_found"] == 1
    assert summary["cpu"]["tctl"]["max"] == 50.0
    assert summary["gpu"]["temp_junction"]["avg"] == 50.0
    assert summary["ram"]["temp_mean"]["avg"] == 31.0

    # Log suspended metrics
    metrics_suspended = {
        "cpu": {"tctl": 40.0, "tccd1": 35.0},
        "gpus": [{
            "name": "Radeon AI PRO R9700",
            "pci_addr": "0000:03:00.0",
            "status": "suspended",
            "temps": {},
            "fan_rpm": None,
            "fan_percent": None,
            "power_w": None,
        }],
        "ram_temps": [28.0, 29.0],
    }
    db.log_metrics(metrics_suspended)

    summary = db.get_summary(window_hours=1)
    assert summary["records_found"] == 2
    # Check max and average calculation
    assert summary["cpu"]["tctl"]["max"] == 50.0
    assert summary["cpu"]["tctl"]["avg"] == 45.0 # (50 + 40) / 2
    assert summary["gpu"]["temp_junction"]["max"] == 50.0 # Suspended doesn't log GPU temps, so max is still 50
    assert summary["gpu"]["temp_junction"]["avg"] == 50.0 # Only active values averaged


def test_observability_active_sessions(tmp_path):
    from datetime import datetime, timedelta, timezone
    db_file = tmp_path / "test_sessions.db"
    db = ObservabilityDB(db_path=db_file)

    now = datetime.now(timezone.utc)
    def ts(offset_seconds):
        return (now + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Insert a sequence of records manually to control timestamps
    # Session 1: 3 active records
    # Gap: 2 inactive records
    # Session 2: 2 active records
    records = [
        (ts(0), 50.0, 1, 40.0, 45.0, 20.0, 30.0),
        (ts(30), 52.0, 1, 42.0, 48.0, 25.0, 32.0),
        (ts(60), 54.0, 1, 44.0, 50.0, 30.0, 35.0),
        (ts(90), 40.0, 0, None, None, None, None), # Suspended
        (ts(120), 38.0, 0, None, None, None, None), # Suspended
        (ts(150), 55.0, 1, 45.0, 51.0, 35.0, 40.0),
        (ts(180), 57.0, 1, 47.0, 53.0, 40.0, 45.0),
    ]

    conn = sqlite3.connect(db_file)
    try:
        for timestamp, cpu, active, edge, junc, pwr, fan in records:
            conn.execute("""
                INSERT INTO telemetry_log (
                    timestamp, cpu_tctl, gpu_active, gpu_temp_edge, 
                    gpu_temp_junction, gpu_power_w, gpu_fan_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, cpu, active, edge, junc, pwr, fan))
        conn.commit()
    finally:
        conn.close()

    sessions = db.get_active_sessions(window_hours=24)
    
    assert len(sessions) == 2
    
    # Sorted latest first (Session 2 starts at ts(150))
    latest_session = sessions[0]
    assert latest_session["start_time"] == ts(150)
    assert latest_session["end_time"] == ts(180)
    assert latest_session["duration_seconds"] == 30
    assert latest_session["gpu_temp_edge"]["max"] == 47.0
    assert latest_session["gpu_temp_edge"]["avg"] == 46.0 # (45 + 47) / 2
    assert latest_session["gpu_power_w"]["avg"] == 37.5 # (35 + 40) / 2

    # Older session (Session 1 starts at ts(0))
    old_session = sessions[1]
    assert old_session["start_time"] == ts(0)
    assert old_session["end_time"] == ts(60)
    assert old_session["duration_seconds"] == 60
    assert old_session["gpu_temp_edge"]["max"] == 44.0
    assert old_session["gpu_temp_junction"]["max"] == 50.0


def test_observability_window_filtering(tmp_path):
    from datetime import datetime, timedelta, timezone
    db_file = tmp_path / "test_filtering.db"
    db = ObservabilityDB(db_path=db_file)

    now = datetime.now(timezone.utc)
    # Session A: 2 hours ago (within 24 hour window)
    ts_a = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    # Intermediary suspended row: 12 hours ago
    ts_mid = (now - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
    # Session B: 25 hours ago (outside 24 hour window, but within 48 hour window)
    ts_b = (now - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_file)
    try:
        # Session B
        conn.execute("""
            INSERT INTO telemetry_log (timestamp, gpu_active, gpu_temp_edge)
            VALUES (?, 1, 60.0)
        """, (ts_b,))
        # Intermediary suspended row
        conn.execute("""
            INSERT INTO telemetry_log (timestamp, gpu_active, gpu_temp_edge)
            VALUES (?, 0, NULL)
        """, (ts_mid,))
        # Session A
        conn.execute("""
            INSERT INTO telemetry_log (timestamp, gpu_active, gpu_temp_edge)
            VALUES (?, 1, 50.0)
        """, (ts_a,))
        conn.commit()
    finally:
        conn.close()

    # Query 24 hours window -> should only find Session A
    sessions_24 = db.get_active_sessions(window_hours=24)
    assert len(sessions_24) == 1
    assert sessions_24[0]["gpu_temp_edge"]["max"] == 50.0

    # Query 48 hours window -> should find both
    sessions_48 = db.get_active_sessions(window_hours=48)
    assert len(sessions_48) == 2


def test_observability_serving_sessions(tmp_path):
    from datetime import datetime, timedelta
    db_file = tmp_path / "test_serving.db"
    db = ObservabilityDB(db_path=db_file)

    # Test initial state
    serving_sessions = db.get_serving_sessions(window_hours=24)
    assert len(serving_sessions) == 0

    # Start a serving session
    sess_id = db.start_serving_session()
    assert sess_id is not None

    serving_sessions = db.get_serving_sessions(window_hours=24)
    assert len(serving_sessions) == 1
    assert serving_sessions[0]["end_time"] is None
    assert serving_sessions[0]["duration_seconds"] is None

    # End the session
    db.end_serving_session(sess_id)
    serving_sessions = db.get_serving_sessions(window_hours=24)
    assert len(serving_sessions) == 1
    assert serving_sessions[0]["end_time"] is not None


@pytest.mark.asyncio
async def test_app_observability_endpoints(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from unittest.mock import MagicMock

    # Create app with fake manager
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )
    
    db_file = tmp_path / "endpoints.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(model_resource),
        observability_db_path=db_file,
    )

    # Patch HardwareMonitor.collect to return mock metrics and avoid /sys
    mock_metrics = {
        "cpu": {"tctl": 45.0, "tccd1": 35.0},
        "gpus": [
            {
                "name": "Radeon AI PRO R9700",
                "pci_addr": "0000:03:00.0",
                "status": "active",
                "temps": {"edge": 40.0},
                "fan_rpm": 1000,
                "fan_percent": 30.0,
                "power_w": 25.0,
            }
        ],
        "ram_temps": [31.0],
        "motherboard": {"fans": [], "temps": []}
    }

    with patch("observability.HardwareMonitor.collect", return_value=mock_metrics):
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # 1. Test GET /api/observability/metrics
                res = await client.get("/api/observability/metrics")
                assert res.status_code == 200
                data = res.json()
                assert data["cpu"]["tctl"] == 45.0
                assert data["gpus"][0]["name"] == "Radeon AI PRO R9700"

                # 2. Test GET /api/observability/history (with default response shape)
                res_history = await client.get("/api/observability/history")
                assert res_history.status_code == 200
                history_data = res_history.json()
                assert "summary" in history_data
                assert "active_sessions" in history_data
                assert "serving_sessions" in history_data
                assert "sessions" in history_data

                # 3. Test GET /api/observability/history?summary_only=true
                res_summary_only = await client.get("/api/observability/history?summary_only=true")
                assert res_summary_only.status_code == 200
                summary_data = res_summary_only.json()
                assert "summary" in summary_data
                assert "active_sessions" not in summary_data
                assert "serving_sessions" not in summary_data


@pytest.mark.asyncio
async def test_app_observability_lifespan_task(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    
    db_file = tmp_path / "lifespan.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )

    with patch("observability.HardwareMonitor.collect", return_value={"cpu": {}, "gpus": [], "ram_temps": [], "motherboard": {"fans": [], "temps": []}}):
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # Entering context manager starts the lifespan
                assert hasattr(app.state, "observability_db")
                assert app.state.observability_db is not None
                # Verify database table is initialized
                db = app.state.observability_db
                summary = db.get_summary()
                assert summary["records_found"] >= 0


@pytest.mark.asyncio
async def test_app_observability_request_serving(tmp_path):
    from httpx import ASGITransport, AsyncClient, Response as HTTPXResponse
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    import httpx
    
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )

    # Set up mock manager with custom proxy session
    fake_manager = FakeManager(model_resource)
    
    # Mock upstream response
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 200
    mock_upstream_response.headers = httpx.Headers({"content-type": "application/json"})
    
    async def mock_aread():
        return b'{"result": "ok"}'
    
    mock_upstream_response.aread = mock_aread
    
    from manager import ProxySession
    mock_client = MagicMock()
    
    async def mock_aclose():
        pass
        
    mock_client.aclose = mock_aclose
    mock_upstream_response.aclose = mock_aclose
    
    proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
    
    async def mock_open_proxy_session(*args, **kwargs):
        return proxy_session
        
    fake_manager.open_proxy_session = mock_open_proxy_session

    db_file = tmp_path / "serving.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=fake_manager,
        observability_db_path=db_file,
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Check initially no serving sessions (isolated database)
            db = app.state.observability_db
            assert len(db.get_serving_sessions()) == 0

            # Run non-streaming proxy request
            res = await client.post("/api/v1/chat/completions", json={"model": "primary"})
            assert res.status_code == 200
            
            # Verify a serving session was recorded and ended (end_time is not None)
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            sessions = db.get_serving_sessions()
            assert len(sessions) == 1
            assert sessions[0]["end_time"] is not None


@pytest.mark.asyncio
async def test_app_observability_request_serving_streaming(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    import httpx
    
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )

    fake_manager = FakeManager(model_resource)
    
    # Mock upstream streaming response
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 200
    mock_upstream_response.headers = httpx.Headers({"content-type": "text/event-stream"})
    
    async def mock_aiter_raw():
        yield b"data: chunk 1\n\n"
        yield b"data: chunk 2\n\n"
    
    mock_upstream_response.aiter_raw = mock_aiter_raw
    
    from manager import ProxySession
    mock_client = MagicMock()
    
    async def mock_aclose():
        pass
        
    mock_client.aclose = mock_aclose
    mock_upstream_response.aclose = mock_aclose
    
    proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
    
    async def mock_open_proxy_session(*args, **kwargs):
        return proxy_session
        
    fake_manager.open_proxy_session = mock_open_proxy_session

    db_file = tmp_path / "streaming.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=fake_manager,
        observability_db_path=db_file,
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            db = app.state.observability_db
            assert len(db.get_serving_sessions()) == 0

            # Stream request and read stream content
            async with client.stream("POST", "/api/v1/chat/completions", json={"model": "primary"}) as response:
                assert response.status_code == 200
                content = await response.aread()
                assert b"chunk" in content

            # Once response context exits, the stream finishes cleaning up
            # Verify serving session has successfully recorded and completed (end_time is set)
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            sessions = db.get_serving_sessions()
            assert len(sessions) == 1
            assert sessions[0]["end_time"] is not None


@pytest.mark.asyncio
async def test_app_observability_request_serving_failures(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    import httpx
    
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )

    fake_manager = FakeManager(model_resource)
    
    # 503 simulation: open_proxy_session raises RuntimeError
    async def mock_open_proxy_session_fail(*args, **kwargs):
        raise RuntimeError("Mock upstream connect error")
        
    db_file = tmp_path / "failures.db"
    app_503 = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=fake_manager,
        observability_db_path=db_file,
    )
    fake_manager.open_proxy_session = mock_open_proxy_session_fail

    async with app_503.router.lifespan_context(app_503):
        async with AsyncClient(transport=ASGITransport(app=app_503), base_url="http://test") as client:
            # 1. 400 Local Validation Failure (Unknown Model)
            res_400 = await client.post("/api/v1/chat/completions", json={"model": "unknown"})
            assert res_400.status_code == 400
            
            # Verify no serving session was logged
            db = app_503.state.observability_db
            assert len(db.get_serving_sessions()) == 0

            # 2. 503 Upstream Connection Failure
            res_503 = await client.post("/api/v1/chat/completions", json={"model": "primary"})
            assert res_503.status_code == 503
            
            # Verify no serving session was logged
            assert len(db.get_serving_sessions()) == 0


@pytest.mark.asyncio
async def test_app_observability_db_write_failures(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    import httpx
    
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )

    fake_manager = FakeManager(model_resource)
    
    # Non-streaming mock response
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 200
    mock_upstream_response.headers = httpx.Headers({"content-type": "application/json"})
    
    async def mock_aread():
        return b'{"result": "ok"}'
    
    mock_upstream_response.aread = mock_aread
    
    # Streaming mock response
    mock_streaming_response = MagicMock()
    mock_streaming_response.status_code = 200
    mock_streaming_response.headers = httpx.Headers({"content-type": "text/event-stream"})
    
    async def mock_aiter_raw():
        yield b"data: chunk 1\n\n"
        
    mock_streaming_response.aiter_raw = mock_aiter_raw

    from manager import ProxySession
    mock_client = MagicMock()
    async def mock_aclose():
        pass
    mock_client.aclose = mock_aclose
    mock_upstream_response.aclose = mock_aclose
    mock_streaming_response.aclose = mock_aclose

    db_file = tmp_path / "failures_writes.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=fake_manager,
        observability_db_path=db_file,
    )

    async with app.router.lifespan_context(app):
        db = app.state.observability_db
        
        # 1. Test non-streaming proxy request when start_serving_session raises exception
        with patch.object(db, "start_serving_session", side_effect=RuntimeError("DB Write Fail")):
            proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
            async def mock_open_proxy_session(*args, **kwargs):
                return proxy_session
            fake_manager.open_proxy_session = mock_open_proxy_session

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                res = await client.post("/api/v1/chat/completions", json={"model": "primary"})
                # Request should succeed despite DB write failure
                assert res.status_code == 200
                assert res.json() == {"result": "ok"}
                
                # Check that state remains consistent (inflight_requests is 0, session_id is None)
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                assert app.state.inflight_requests == 0
                assert app.state.current_serving_session_id_future is None

        # 2. Test streaming proxy request when start_serving_session raises exception
        with patch.object(db, "start_serving_session", side_effect=RuntimeError("DB Write Fail")):
            proxy_session_stream = ProxySession(client=mock_client, response=mock_streaming_response)
            async def mock_open_proxy_session_stream(*args, **kwargs):
                return proxy_session_stream
            fake_manager.open_proxy_session = mock_open_proxy_session_stream

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                async with client.stream("POST", "/api/v1/chat/completions", json={"model": "primary"}) as response:
                    assert response.status_code == 200
                    content = await response.aread()
                    assert b"chunk" in content

                # After stream consumption and context exit
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                assert app.state.inflight_requests == 0
                assert app.state.current_serving_session_id_future is None

        # 3. Test successful request when end_serving_session raises exception
        with patch.object(db, "end_serving_session", side_effect=RuntimeError("DB End Fail")):
            proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
            async def mock_open_proxy_session_end(*args, **kwargs):
                return proxy_session
            fake_manager.open_proxy_session = mock_open_proxy_session_end

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                res = await client.post("/api/v1/chat/completions", json={"model": "primary"})
                assert res.status_code == 200
                
                # Check that state remains consistent (inflight_requests resets to 0, session_id resets to None)
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                assert app.state.inflight_requests == 0
                assert app.state.current_serving_session_id_future is None


@pytest.mark.asyncio
async def test_app_observability_offloaded_to_thread(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import asyncio
    
    db_file = tmp_path / "offload.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    original_to_thread = asyncio.to_thread
    calls = []
    
    async def mock_to_thread(func, *args, **kwargs):
        calls.append(func)
        return await original_to_thread(func, *args, **kwargs)
        
    async with app.router.lifespan_context(app):
        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            # Trigger increment and decrement to check offloading
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
            # Trigger metrics endpoint logic
            mock_metrics = {"cpu": {}, "gpus": [], "ram_temps": []}
            mock_collect = None
            mock_collect_logger = None
            for route in app.router.routes:
                if route.path == "/api/observability/metrics":
                    with patch("observability.HardwareMonitor.collect", return_value=mock_metrics) as m_coll:
                        mock_collect = m_coll
                        await route.endpoint()
                elif route.path == "/api/observability/history":
                    await route.endpoint()
                    
            # Trigger start_telemetry_logger background task offloading
            from observability import start_telemetry_logger
            with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]):
                try:
                    with patch("observability.HardwareMonitor.collect", return_value=mock_metrics) as m_coll_logger:
                        mock_collect_logger = m_coll_logger
                        await start_telemetry_logger(app.state.observability_db, interval_seconds=1)
                except asyncio.CancelledError:
                    pass
                
    # Verify that the DB operations and metrics collect were offloaded to thread
    db = app.state.observability_db
    assert any(c == db.start_serving_session for c in calls)
    assert any(c == db.end_serving_session for c in calls)
    assert any(c == db.get_summary for c in calls)
    assert any(c == db.log_metrics for c in calls)
    assert mock_collect in calls
    assert mock_collect_logger in calls


@pytest.mark.asyncio
async def test_app_observability_db_write_recovery_and_non_blocking_latency(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from schemas import ModelResource, ModelConfig, ModelStatus, RuntimeConfig, ModelSource
    import httpx
    import time
    
    model_resource = ModelResource(
        name="primary",
        config=ModelConfig(
            name="primary",
            runtimes={
                "rocm": RuntimeConfig(
                    docker_image="ghcr.io/ggerganov/llama.cpp:server-rocm",
                    source=ModelSource(local_path=Path("/models/dummy.gguf")),
                )
            },
        ),
        status=ModelStatus(
            name="primary",
            state="running",
            model="repo/model.gguf",
            host="127.0.0.1",
            port=8080,
            base_url="http://127.0.0.1:8080",
            active=True,
            active_runtime="rocm",
        ),
    )

    fake_manager = FakeManager(model_resource)
    
    # Mock response that completes instantly
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 200
    mock_upstream_response.headers = httpx.Headers({"content-type": "application/json"})
    
    async def mock_aread():
        return b'{"result": "ok"}'
    mock_upstream_response.aread = mock_aread

    from manager import ProxySession
    mock_client = MagicMock()
    async def mock_aclose():
        pass
    mock_client.aclose = mock_aclose
    mock_upstream_response.aclose = mock_aclose

    db_file = tmp_path / "recovery.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=fake_manager,
        observability_db_path=db_file,
    )

    async with app.router.lifespan_context(app):
        db = app.state.observability_db
        
        # --- TEST 1: PROVE REQUEST IS NOT BLOCKED BY SLOW/STALLED DB WRITES ---
        original_start = db.start_serving_session
        original_end = db.end_serving_session
        
        def slow_start(*args, **kwargs):
            time.sleep(2.0)
            return original_start(*args, **kwargs)
            
        def slow_end(*args, **kwargs):
            time.sleep(2.0)
            return original_end(*args, **kwargs)
            
        with patch.object(db, "start_serving_session", side_effect=slow_start), \
             patch.object(db, "end_serving_session", side_effect=slow_end):
             
            proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
            async def mock_open_proxy_session(*args, **kwargs):
                return proxy_session
            fake_manager.open_proxy_session = mock_open_proxy_session

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                start_time = time.time()
                res = await client.post("/api/v1/chat/completions", json={"model": "primary"})
                end_time = time.time()
                
                assert res.status_code == 200
                # Request should return almost instantly (< 0.5s), even though DB calls take 4 seconds total
                assert (end_time - start_time) < 0.5

        # Cleanup slow background tasks
        if app.state.observability_tasks:
            await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)

        # --- TEST 2: RECOVERY AND RETRY OF FAILED END_SERVING_SESSION ---
        sessions = db.get_serving_sessions()
        assert len(sessions) >= 1
        for s in sessions:
            assert s["end_time"] is not None # Completed successfully in background
            
        # We will trigger a request where the database execute raises an exception during session close
        recovery_allowed = False
        original_connect = sqlite3.connect
        
        class MockConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if not recovery_allowed and sql.strip().startswith("UPDATE serving_session SET end_time"):
                    raise sqlite3.OperationalError("Mock database lock error")
                return super().execute(sql, *args, **kwargs)
        
        def mock_connect(*args, **kwargs):
            kwargs["factory"] = MockConnection
            return original_connect(*args, **kwargs)

        with patch("observability.sqlite3.connect", side_effect=mock_connect):
            proxy_session = ProxySession(client=mock_client, response=mock_upstream_response)
            async def mock_open_proxy_session2(*args, **kwargs):
                return proxy_session
            fake_manager.open_proxy_session = mock_open_proxy_session2

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                res = await client.post("/api/v1/chat/completions", json={"model": "primary"})
                assert res.status_code == 200
                
            # Wait for background tasks to complete execution
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)

            # Get latest session from DB
            sessions = db.get_serving_sessions()
            latest = sessions[0]
            # Since end_serving_session failed once, it is left open in DB (end_time is None)
            assert latest["end_time"] is None
            # But the session ID should be in the DB's pending close set!
            assert latest["id"] in db._pending_close_sessions

        # Now allow recovery and trigger a recovery run by calling retry_pending_closes
        recovery_allowed = True
        await asyncio.to_thread(db.retry_pending_closes)
        
        # Verify the session has been closed in the DB now
        sessions = db.get_serving_sessions()
        latest = sessions[0]
        assert latest["end_time"] is not None
        assert latest["id"] not in db._pending_close_sessions


@pytest.mark.asyncio
async def test_shutdown_waits_for_pending_writes(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    
    db_file = tmp_path / "shutdown_wait.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    original_end = ObservabilityDB.end_serving_session
    def slow_end(self, *args, **kwargs):
        time.sleep(1.0)
        return original_end(self, *args, **kwargs)
        
    with patch.object(ObservabilityDB, "end_serving_session", slow_end):
        async with app.router.lifespan_context(app):
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            assert len(app.state.observability_tasks) > 0
            start_shutdown = time.time()
            
    end_shutdown = time.time()
    # Lifespan shutdown must have waited for the slow end_serving_session write to complete
    assert (end_shutdown - start_shutdown) >= 0.9
    
    # Verify the session was indeed recorded as closed in the DB
    new_db = ObservabilityDB(db_path=db_file)
    sessions = new_db.get_serving_sessions()
    assert len(sessions) == 1
    assert sessions[0]["end_time"] is not None


@pytest.mark.asyncio
async def test_shutdown_timeout_behavior(tmp_path, caplog):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    import logging
    from unittest.mock import patch
    from observability import ObservabilityDB
    
    db_file = tmp_path / "shutdown_timeout.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    original_wait_for = asyncio.wait_for
    async def mock_wait_for(fut, timeout=None):
        # Force a short timeout of 0.1 seconds for testing
        return await original_wait_for(fut, timeout=0.1)
        
    original_end = ObservabilityDB.end_serving_session
    def slow_end(self, *args, **kwargs):
        time.sleep(1.0)
        return original_end(self, *args, **kwargs)
        
    with patch.object(ObservabilityDB, "end_serving_session", slow_end), \
         patch("asyncio.wait_for", side_effect=mock_wait_for), \
         caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
    warnings = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("Timeout waiting for pending observability tasks to finish on shutdown" in w for w in warnings)
    
    # Sleep to allow the slow background thread to eventually complete its write
    await asyncio.sleep(1.2)
    new_db = ObservabilityDB(db_path=db_file)
    sessions = new_db.get_serving_sessions()
    assert len(sessions) == 1
    assert sessions[0]["end_time"] is not None


@pytest.mark.asyncio
async def test_db_close_does_not_race_pending_tasks(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    
    db_file = tmp_path / "race_test.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    task_done_before_close = False
    
    original_close = ObservabilityDB.close
    def mock_close(self):
        nonlocal task_done_before_close
        tasks = getattr(app.state, "observability_tasks", set())
        task_done_before_close = all(t.done() for t in tasks)
        original_close(self)
        
    original_end = ObservabilityDB.end_serving_session
    def slow_end(self, *args, **kwargs):
        time.sleep(0.5)
        return original_end(self, *args, **kwargs)
        
    with patch.object(ObservabilityDB, "close", mock_close), \
         patch.object(ObservabilityDB, "end_serving_session", slow_end):
         
        async with app.router.lifespan_context(app):
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
    assert task_done_before_close is True


@pytest.mark.asyncio
async def test_retry_pending_closes_gating(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    import sqlite3
    
    db_file = tmp_path / "gating.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    async with app.router.lifespan_context(app):
        db = app.state.observability_db
        
        # Verify no retry scheduled when pending is empty
        retry_calls = 0
        original_retry = ObservabilityDB.retry_pending_closes
        def mock_retry(self):
            nonlocal retry_calls
            retry_calls += 1
            return original_retry(self)
            
        with patch.object(ObservabilityDB, "retry_pending_closes", mock_retry):
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                
            assert retry_calls == 0
            
        # Simulate a failed close to populate _pending_close_sessions
        original_connect = sqlite3.connect
        class MockConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql.strip().startswith("UPDATE serving_session SET end_time"):
                    raise sqlite3.OperationalError("Mock DB error on close")
                return super().execute(sql, *args, **kwargs)
                
        def mock_connect(*args, **kwargs):
            kwargs["factory"] = MockConnection
            return original_connect(*args, **kwargs)
            
        with patch("observability.sqlite3.connect", side_effect=mock_connect):
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                
        assert len(db._pending_close_sessions) == 1
        
        # Verify retry is scheduled and repeated transitions do not spawn multiple concurrent retry jobs
        retry_calls = 0
        retry_started = asyncio.Event()
        retry_finish = asyncio.Event()
        
        async def mock_retry_delayed():
            nonlocal retry_calls
            retry_calls += 1
            retry_started.set()
            await retry_finish.wait()
            await original_to_thread(original_retry, db)
            
        original_to_thread = asyncio.to_thread
        async def mock_to_thread(func, *args, **kwargs):
            if getattr(func, "__name__", None) == "retry_pending_closes":
                await mock_retry_delayed()
                return
            return await original_to_thread(func, *args, **kwargs)
            
        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            await increment_inflight_requests(app)
            await asyncio.wait_for(retry_started.wait(), timeout=2.0)
            
            # Trigger subsequent transitions while retry is in progress
            await increment_inflight_requests(app)
            await decrement_inflight_requests(app)
            
            retry_finish.set()
            
            if app.state.observability_tasks:
                await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
                
            assert retry_calls == 1


@pytest.mark.asyncio
async def test_serving_session_timestamps(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    from datetime import datetime, timedelta, timezone
    import sqlite3
    
    db_file = tmp_path / "timestamps.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    async with app.router.lifespan_context(app):
        db = app.state.observability_db
        
        # Test 1: Delayed start write with preserved original start_time
        original_start = ObservabilityDB.start_serving_session
        request_start_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        
        with patch("app.datetime") as mock_datetime:
            mock_datetime.now.return_value = request_start_time
            
            def slow_start(self, start_time=None):
                time.sleep(1.0)
                return original_start(self, start_time)
                
            with patch.object(ObservabilityDB, "start_serving_session", slow_start):
                await increment_inflight_requests(app)
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            
        sessions = db.get_serving_sessions()
        assert len(sessions) == 1
        db_start_str = sessions[0]["start_time"]
        expected_start_str = request_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert db_start_str == expected_start_str
        
        # Test 2: Delayed end write with preserved original end_time
        request_end_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        original_end = ObservabilityDB.end_serving_session
        
        with patch("app.datetime") as mock_datetime:
            mock_datetime.now.return_value = request_end_time
            
            def slow_end(self, session_id, end_time=None):
                time.sleep(1.0)
                return original_end(self, session_id, end_time)
                
            with patch.object(ObservabilityDB, "end_serving_session", slow_end):
                await decrement_inflight_requests(app)
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            
        sessions = db.get_serving_sessions()
        assert len(sessions) == 1
        db_end_str = sessions[0]["end_time"]
        expected_end_str = request_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert db_end_str == expected_end_str
        
        # Test 3: Failed end followed by retry, verifying the final persisted end_time matches original failed_end_time
        # Start a new session
        await increment_inflight_requests(app)
        if app.state.observability_tasks:
            await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            
        failed_end_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        
        original_connect = sqlite3.connect
        class MockConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql.strip().startswith("UPDATE serving_session SET end_time"):
                    raise sqlite3.OperationalError("Mock DB write error")
                return super().execute(sql, *args, **kwargs)
                
        def mock_connect(*args, **kwargs):
            kwargs["factory"] = MockConnection
            return original_connect(*args, **kwargs)
            
        with patch("app.datetime") as mock_datetime:
            mock_datetime.now.return_value = failed_end_time
            with patch("observability.sqlite3.connect", side_effect=mock_connect):
                await decrement_inflight_requests(app)
                if app.state.observability_tasks:
                    await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            
        sessions = db.get_serving_sessions()
        latest = sessions[0]
        assert latest["end_time"] is None
        assert latest["id"] in db._pending_close_sessions
        assert db._pending_close_sessions[latest["id"]] == failed_end_time
        
        # Run retry
        await asyncio.to_thread(db.retry_pending_closes)
        
        sessions = db.get_serving_sessions()
        latest = sessions[0]
        assert latest["end_time"] is not None
        assert latest["end_time"] == failed_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.asyncio
async def test_retry_telemetry_request_overlap(tmp_path):
    from app import create_app, increment_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    from datetime import datetime
    
    db_file = tmp_path / "overlap.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    async with app.router.lifespan_context(app):
        db = app.state.observability_db
        
        # Populate _pending_close_sessions
        db._pending_close_sessions[123] = datetime.now()
        
        retry_calls = 0
        original_retry = ObservabilityDB.retry_pending_closes
        retry_started = asyncio.Event()
        retry_finish = asyncio.Event()
        
        async def mock_retry_delayed():
            nonlocal retry_calls
            retry_calls += 1
            retry_started.set()
            await retry_finish.wait()
            await original_to_thread(original_retry, db)
            
        original_to_thread = asyncio.to_thread
        async def mock_to_thread(func, *args, **kwargs):
            if getattr(func, "__name__", None) == "retry_pending_closes":
                await mock_retry_delayed()
                return
            return await original_to_thread(func, *args, **kwargs)
            
        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            # 1. Trigger request-path retry
            await increment_inflight_requests(app)
            await asyncio.wait_for(retry_started.wait(), timeout=2.0)
            
            # 2. Trigger telemetry-loop retry tick while request retry is active
            db.trigger_retry(app.state.observability_tasks)
            
            # Finish the first retry
            retry_finish.set()
            
            # Verify only one retry run executed
            assert retry_calls == 1


@pytest.mark.asyncio
async def test_timezone_consistency(tmp_path):
    from app import create_app, increment_inflight_requests, decrement_inflight_requests
    from schemas import AppConfig
    from tests.test_app import FakeManager
    from datetime import datetime, timezone, timedelta
    from observability import ObservabilityDB
    import asyncio
    
    db_file = tmp_path / "timezone.db"
    
    # 1. Test explicit start/end timestamps round-trip correctly under UTC
    db = ObservabilityDB(db_path=db_file)
    start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    end_time = datetime.now(timezone.utc)
    
    session_id = db.start_serving_session(start_time)
    db.end_serving_session(session_id, end_time)
    
    sessions = db.get_serving_sessions(window_hours=1)
    assert len(sessions) == 1
    assert sessions[0]["start_time"] == start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert sessions[0]["end_time"] == end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert sessions[0]["duration_seconds"] == 300
    
    # 2. Test a completed serving session is still returned in a recent window (e.g. window_hours=1)
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    async with app.router.lifespan_context(app):
        # Trigger transition
        await increment_inflight_requests(app)
        await decrement_inflight_requests(app)
        
        if app.state.observability_tasks:
            await asyncio.gather(*list(app.state.observability_tasks), return_exceptions=True)
            
        history_route = None
        for route in app.router.routes:
            if route.path == "/api/observability/history":
                history_route = route
                break
                
        # Query with window_hours=1
        response = await history_route.endpoint(window_hours=1)
        # Should return both sessions
        assert len(response["serving_sessions"]) >= 2
        for s in response["serving_sessions"]:
            assert s["start_time"].endswith("Z")
            assert "T" in s["start_time"]
            if s["end_time"]:
                assert s["end_time"].endswith("Z")
                assert "T" in s["end_time"]
        
    # 3. Test startup recovery of an open session does not inflate duration due to timezone mismatch
    db_file_recovery = tmp_path / "recovery_tz.db"
    db_init = ObservabilityDB(db_path=db_file_recovery)
    
    # Start a session 10 seconds ago (UTC)
    start_time_rec = datetime.now(timezone.utc) - timedelta(seconds=10)
    db_init.start_serving_session(start_time_rec)
    
    # Startup recovery runs when the new ObservabilityDB is instantiated
    db_recovered = ObservabilityDB(db_path=db_file_recovery)
    sessions_recovered = db_recovered.get_serving_sessions(window_hours=1)
    
    assert len(sessions_recovered) == 1
    # Duration should be close to 10 seconds (e.g. between 5 and 15 seconds)
    assert 5 <= sessions_recovered[0]["duration_seconds"] <= 15


@pytest.mark.asyncio
async def test_telemetry_retry_shutdown(tmp_path):
    from app import create_app
    from schemas import AppConfig
    from tests.test_app import FakeManager
    import time
    from unittest.mock import patch
    from observability import ObservabilityDB
    from datetime import datetime, timezone
    import asyncio
    
    db_file = tmp_path / "telemetry_shutdown.db"
    app = create_app(
        app_config=AppConfig(models=[], api_prefix="/api", hf_cache_dir=tmp_path),
        manager=FakeManager(None),
        observability_db_path=db_file,
    )
    
    original_retry = ObservabilityDB.retry_pending_closes
    def slow_retry(self):
        time.sleep(1.0)
        return original_retry(self)
        
    with patch.object(ObservabilityDB, "retry_pending_closes", slow_retry):
        async with app.router.lifespan_context(app):
            db = app.state.observability_db
            
            # Start a real session to obtain a valid session_id
            session_id = db.start_serving_session()
            
            # Populate pending close
            db._pending_close_sessions[session_id] = datetime.now(timezone.utc)
            
            # Telemetry loop triggers retry, adding the task to app.state.observability_tasks
            db.trigger_retry(app.state.observability_tasks)
            assert len(app.state.observability_tasks) > 0
            
            # Start shutdown, it should block until slow_retry completes
            start_shutdown = time.time()
            
    end_shutdown = time.time()
    # It must have waited for the retry task
    assert (end_shutdown - start_shutdown) >= 0.9
    
    # Verify the session is successfully closed in the DB
    new_db = ObservabilityDB(db_path=db_file)
    sessions = new_db.get_serving_sessions()
    assert len(sessions) == 1
    assert sessions[0]["end_time"] is not None



