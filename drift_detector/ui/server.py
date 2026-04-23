#!/usr/bin/env python3
"""
DriftDetector v2 - FastAPI Dashboard Server

Provides REST API for drift monitoring UI.
Endpoints: GET /api/chain, POST /api/drift, GET /api/health, GET /api/config
"""

import os
import logging
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

logger = logging.getLogger(__name__)

from drift_detector.core import DriftDetectorAgent
from drift_detector.core.drift_detector_agent import AgentConfig, SessionSnapshot

# Optional session storage (can be disabled)
try:
    from drift_detector.core.session_storage import SessionStorage
    SESSION_STORAGE_AVAILABLE = True
except ImportError:
    SESSION_STORAGE_AVAILABLE = False

# ============================================================================
# Detector Lifecycle Management (Thread-Safe)
# ============================================================================

_detector_instance: Optional[DriftDetectorAgent] = None
_session_storage: Optional['SessionStorage'] = None
_current_session_id: Optional[str] = None
_enable_persistence = os.getenv("PERSIST_SESSIONS", "false").lower() == "true"

def _init_detector() -> DriftDetectorAgent:
    """Initialize detector (called once at startup)"""
    global _detector_instance, _session_storage, _current_session_id
    try:
        config = AgentConfig(
            agent_id="ui_detector",
            drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.6")),
            signal_threshold=float(os.getenv("SIGNAL_THRESHOLD", "0.7")),
        )
        _detector_instance = DriftDetectorAgent(config)
        logger.info("DriftDetector initialized: %s", config.agent_id)

        # Optional session storage
        if _enable_persistence and SESSION_STORAGE_AVAILABLE:
            db_path = os.getenv("SESSION_DB_PATH", "drift_sessions.db")
            _session_storage = SessionStorage(db_path)
            _current_session_id = _session_storage.create_session()
            logger.info("Session storage enabled: %s", db_path)
            logger.info("Current session: %s", _current_session_id)
        else:
            logger.info("Session storage disabled (in-memory only)")

        return _detector_instance
    except (ValueError, TypeError) as e:
        logger.warning("Failed to initialize DriftDetector: %s", e)
        raise RuntimeError(f"DriftDetector config error: {e}") from e
    except Exception as e:
        logger.warning("Unexpected error in DriftDetector init: %s", e)
        raise RuntimeError(f"DriftDetector initialization failed: {e}") from e

def get_detector() -> DriftDetectorAgent:
    """Dependency injection for detector (thread-safe via lifespan)"""
    if _detector_instance is None:
        raise RuntimeError("Detector not initialized. Check app startup.")
    return _detector_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifecycle: startup → shutdown"""
    # Startup
    _init_detector()
    yield
    # Shutdown (if needed)
    global _detector_instance
    _detector_instance = None

# ============================================================================
# FastAPI Setup
# ============================================================================

app = FastAPI(
    title="DriftDetector v2 API",
    description="Real-time behavioral drift monitoring",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS - Restrict to allowed origins in production
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
# allow_credentials=True + allow_origins=["*"] is an invalid combo (browser rejects it)
_allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def limit_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 256_000:
        return JSONResponse({"detail": "payload too large"}, status_code=413)
    return await call_next(request)

# ============================================================================
# Request/Response Models
# ============================================================================

class SnapshotRequest(BaseModel):
    """Request to measure drift"""
    before_text: str
    after_text: str
    tool_calls_before: Optional[List[str]] = None
    tool_calls_after: Optional[List[str]] = None
    agent_id: Optional[str] = "ui_session"

class DriftResponse(BaseModel):
    """Drift measurement response"""
    combined_drift_score: float
    ghost_loss: float
    behavior_shift: float
    agreement_score: float
    stagnation_score: float
    is_drifting: bool
    loop_detected: bool
    timestamp: str

class ChainData(BaseModel):
    """Full chain history response"""
    total_reports: int
    reports: List[Dict]
    trend: Optional[Dict]
    average_drift: float
    max_drift: float
    min_drift: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    detector_id: str
    drift_history_count: int
    uptime: str

class ConfigResponse(BaseModel):
    """Configuration response"""
    drift_threshold: float
    signal_threshold: float
    version: str

# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check(detector: DriftDetectorAgent = Depends(get_detector)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        detector_id=detector.agent_id,
        drift_history_count=len(detector.drift_history),
        uptime="running"
    )

@app.get("/api/config", response_model=ConfigResponse)
async def get_config(detector: DriftDetectorAgent = Depends(get_detector)):
    """Get current configuration"""
    return ConfigResponse(
        drift_threshold=detector.config.drift_threshold,
        signal_threshold=detector.config.signal_threshold,
        version="2.0.0"
    )

@app.post("/api/drift", response_model=DriftResponse)
async def measure_drift(request: SnapshotRequest, detector: DriftDetectorAgent = Depends(get_detector)):
    """Measure drift between two text snapshots"""
    try:

        # Create snapshots
        before = detector.snapshot(
            agent_id=request.agent_id,
            response_text=request.before_text,
            tool_calls=request.tool_calls_before or []
        )

        after = detector.snapshot(
            agent_id=request.agent_id,
            response_text=request.after_text,
            tool_calls=request.tool_calls_after or []
        )

        # Measure drift
        report = detector.measure_drift(before, after)

        return DriftResponse(
            combined_drift_score=report.combined_drift_score,
            ghost_loss=report.ghost_loss,
            behavior_shift=report.behavior_shift,
            agreement_score=report.agreement_score,
            stagnation_score=report.stagnation_score,
            is_drifting=report.is_drifting,
            loop_detected=report.loop_report.is_looping if report.loop_report else False,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift measurement failed: {str(e)}")

@app.get("/api/chain", response_model=ChainData)
async def get_chain_data(
    limit: int = 500,
    offset: int = 0,
    detector: DriftDetectorAgent = Depends(get_detector)
):
    """Get drift history with trends (paginated via limit/offset)"""
    try:

        if len(detector.drift_history) == 0:
            return ChainData(
                total_reports=0,
                reports=[],
                trend=None,
                average_drift=0.0,
                max_drift=0.0,
                min_drift=0.0
            )

        # Convert reports to dict format (paginated)
        history_page = detector.drift_history[offset: offset + limit]
        reports = []
        scores = []

        for i, report in enumerate(history_page, offset + 1):
            score = report.combined_drift_score
            scores.append(score)

            reports.append({
                "step_id": f"report_{i}",
                "step_number": i,
                "timestamp": report.timestamp,
                "combined_drift_score": score,
                "ghost_loss": report.ghost_loss,
                "behavior_shift": report.behavior_shift,
                "agreement_score": report.agreement_score,
                "stagnation_score": report.stagnation_score,
                "is_drifting": report.is_drifting,
            })

        # Calculate statistics
        avg_drift = sum(scores) / len(scores) if scores else 0.0
        max_drift = max(scores) if scores else 0.0
        min_drift = min(scores) if scores else 0.0

        # Get trend
        trend_data = detector.get_drift_trend(window_size=5)

        return ChainData(
            total_reports=len(detector.drift_history),
            reports=reports,
            trend=trend_data,
            average_drift=avg_drift,
            max_drift=max_drift,
            min_drift=min_drift
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chain data: {str(e)}")

@app.get("/")
async def root():
    """Serve frontend HTML dashboard (v4 with sessions, theme, help)"""
    # Try v4 first, then v3, then v2
    frontend_path = Path(__file__).parent / "frontend_v4_sessions.html"
    if not frontend_path.exists():
        frontend_path = Path(__file__).parent / "frontend_v3_modern.html"
    if not frontend_path.exists():
        frontend_path = Path(__file__).parent / "frontend.html"

    if frontend_path.exists():
        return FileResponse(str(frontend_path), media_type="text/html")
    return {"message": "Dashboard not found. Check /docs for API"}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("UI_HOST", "127.0.0.1")
    port = int(os.getenv("UI_PORT", "8000"))

    print(f"Starting DriftDetector UI on {host}:{port}")
    print(f"Dashboard: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
