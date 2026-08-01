import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from configs.settings import settings
from src.api.schemas import (
    BatchDemandPredictionRequest,
    BatchDemandPredictionResponse,
    DemandPredictionRequest,
    DemandPredictionResponse,
    DriftReportResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.mlops.prediction_logger import prediction_logger
from src.mlops.registry import model_registry
from src.models.forecaster import forecaster
from src.models.train_pipeline import get_bike_data, run_training_pipeline
from src.monitoring.drift_detector import drift_detector
from src.monitoring.system_metrics import SystemMetricsMonitor
from src.retraining.retrainer import retrainer
from src.utils.logger import app_logger

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False
    limiter = None

app = FastAPI(
    title="🚲 Enterprise Bike Rental Demand Forecasting API",
    description="Production-grade MLOps REST microservice for bike rental demand forecasting, quantile uncertainty estimation, and fleet rebalancing recommendations.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

if HAS_SLOWAPI:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    request.state.request_id = request_id

    response = await call_next(request)

    process_time_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = str(process_time_ms)

    app_logger.info(
        f"Request [{request_id}] | {request.method} {request.url.path} | Status: {response.status_code} | Latency: {process_time_ms}ms"
    )
    return response


@app.on_event("startup")
def startup_event():
    app_logger.info("Initializing Enterprise FastAPI Microservice...")
    try:
        forecaster.bundle = model_registry.load_champion()
    except Exception as e:
        app_logger.warning(
            f"No Champion model loaded on startup: {e}. Executing training pipeline..."
        )
        run_training_pipeline()
        forecaster.bundle = model_registry.load_champion()


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health & Status"])
def health_check():
    return HealthResponse(
        status="healthy",
        app_version=settings.app_version,
        environment=settings.environment,
    )


@app.get("/live", tags=["Health & Status"])
def liveness_probe():
    return {"status": "alive"}


@app.get("/ready", tags=["Health & Status"])
def readiness_probe():
    if forecaster.bundle is not None:
        return {"status": "ready", "model_loaded": True}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not_ready", "model_loaded": False},
    )


@app.get("/version", tags=["Health & Status"])
def get_version():
    meta = model_registry.load_metadata()
    return {
        "app_version": settings.app_version,
        "model_version": meta.get("version", "3.0.0"),
        "champion_model": meta.get("champion_model", "CatBoost"),
        "git_commit": meta.get("git_commit", "local_dev"),
    }


@app.get("/system", tags=["Health & Status"])
def get_system_metrics():
    return SystemMetricsMonitor.get_system_health()


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model Registry"])
def get_model_info():
    meta = model_registry.load_metadata()
    return ModelInfoResponse(
        champion_model=meta.get("champion_model", "CatBoost"),
        version=meta.get("version", "3.0.0"),
        git_commit=meta.get("git_commit", "local_dev"),
        training_timestamp=meta.get("training_timestamp", "N/A"),
        champion_metrics=meta.get("champion_metrics", {}),
        feature_names=meta.get("feature_names", []),
    )


@app.get("/metrics", tags=["Model Registry"])
def get_metrics():
    meta = model_registry.load_metadata()
    return {
        "champion_metrics": meta.get("champion_metrics", {}),
        "challenger_metrics": meta.get("challenger_metrics", {}),
    }


@app.post(
    "/predict", response_model=DemandPredictionResponse, tags=["Inference Engine"]
)
def predict_demand(payload: DemandPredictionRequest, request: Request):
    t0 = time.time()
    try:
        input_df = pd.DataFrame(
            [
                {
                    "season": payload.season,
                    "yr": payload.yr,
                    "mnth": payload.mnth,
                    "hr": payload.hr,
                    "holiday": payload.holiday,
                    "weekday": payload.weekday,
                    "workingday": payload.workingday,
                    "weathersit": payload.weather,
                    "temp": payload.temp,
                    "hum": payload.hum,
                    "windspeed": payload.windspeed,
                }
            ]
        )

        result = forecaster.predict_single(input_df)
        latency_ms = round((time.time() - t0) * 1000, 2)

        # Audit prediction logging
        req_id = getattr(request.state, "request_id", "local")
        meta = model_registry.load_metadata()
        prediction_logger.log_prediction(
            request_id=req_id,
            inputs=payload.model_dump(),
            predicted_demand=result["predicted_demand"],
            q10_bound=result["q10_demand_bound"],
            q90_bound=result["q90_demand_bound"],
            revenue=result["estimated_revenue_usd"],
            model_version=meta.get("champion_model", "CatBoost"),
            latency_ms=latency_ms,
        )

        return DemandPredictionResponse(**result)
    except Exception as e:
        app_logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post(
    "/predict_batch",
    response_model=BatchDemandPredictionResponse,
    tags=["Inference Engine"],
)
def predict_batch(payload: BatchDemandPredictionRequest):
    try:
        results = []
        for item in payload.inputs:
            input_df = pd.DataFrame(
                [
                    {
                        "season": item.season,
                        "yr": item.yr,
                        "mnth": item.mnth,
                        "hr": item.hr,
                        "holiday": item.holiday,
                        "weekday": item.weekday,
                        "workingday": item.workingday,
                        "weathersit": item.weather,
                        "temp": item.temp,
                        "hum": item.hum,
                        "windspeed": item.windspeed,
                    }
                ]
            )
            res = forecaster.predict_single(input_df)
            results.append(DemandPredictionResponse(**res))

        return BatchDemandPredictionResponse(
            total_predictions=len(results), predictions=results
        )
    except Exception as e:
        app_logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/drift", response_model=DriftReportResponse, tags=["MLOps Monitoring"])
def get_drift_report():
    try:
        ref_df = get_bike_data()
        curr_df = ref_df.sample(min(200, len(ref_df)), random_state=42)
        summary = drift_detector.run_drift_analysis(ref_df, curr_df)
        return DriftReportResponse(**summary)
    except Exception as e:
        app_logger.error(f"Drift report generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/retrain", tags=["MLOps Actions"])
def trigger_retraining():
    try:
        res = retrainer.execute_retraining()
        forecaster.bundle = model_registry.load_champion()
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/rollback", tags=["MLOps Actions"])
def trigger_rollback():
    try:
        res = retrainer.rollback()
        forecaster.bundle = model_registry.load_champion()
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
