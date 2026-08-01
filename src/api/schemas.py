from typing import Any

from pydantic import BaseModel, Field


class DemandPredictionRequest(BaseModel):
    season: str = Field(
        ..., example="summer", description="Season name: springer, summer, fall, winter"
    )
    yr: str = Field(..., example="2012", description="Year: 2011 or 2012")
    mnth: float = Field(
        ..., ge=1, le=12, example=6.0, description="Month of year (1-12)"
    )
    hr: int = Field(..., ge=0, le=23, example=18, description="Hour of day (0-23)")
    holiday: str = Field(..., example="No", description="Is holiday: Yes or No")
    weekday: int = Field(
        ..., ge=0, le=6, example=2, description="Day of week (0=Sun, 6=Sat)"
    )
    workingday: str = Field(
        ...,
        example="Working Day",
        description="Working day status: Working Day or No work",
    )
    weather: str = Field(
        ...,
        example="Clear",
        description="Weather condition: Clear, Mist, Light Snow, Heavy Rain",
    )
    temp: float = Field(
        ..., ge=0.0, le=1.0, example=0.65, description="Normalized temperature (0-1)"
    )
    hum: float = Field(
        ..., ge=0.0, le=1.0, example=0.45, description="Normalized humidity (0-1)"
    )
    windspeed: float = Field(
        ..., ge=0.0, le=1.0, example=0.15, description="Normalized windspeed (0-1)"
    )


class DemandPredictionResponse(BaseModel):
    predicted_demand: int = Field(..., description="Median expected bike demand")
    q10_demand_bound: int = Field(..., description="10th percentile lower demand bound")
    q90_demand_bound: int = Field(..., description="90th percentile upper demand bound")
    confidence_interval_pct: int = Field(
        90, description="Confidence interval percentage"
    )
    estimated_revenue_usd: float = Field(
        ..., description="Estimated trip rental revenue"
    )
    fleet_utilization_pct: float = Field(
        ..., description="Expected fleet utilization percentage"
    )
    trucks_recommended: int = Field(..., description="Recommended rebalancing trucks")
    staff_recommended: int = Field(..., description="Recommended staffing count")
    stockout_risk_level: str = Field(
        ..., description="Stockout risk level: Low, Moderate, High"
    )
    operational_recommendations: list[str] = Field(
        ..., description="Operational action directives"
    )
    disclaimer: str = Field(..., description="Calculation basis statement")


class BatchDemandPredictionRequest(BaseModel):
    inputs: list[DemandPredictionRequest]


class BatchDemandPredictionResponse(BaseModel):
    total_predictions: int
    predictions: list[DemandPredictionResponse]


class HealthResponse(BaseModel):
    status: str
    app_version: str
    environment: str


class ModelInfoResponse(BaseModel):
    champion_model: str
    version: str
    git_commit: str
    training_timestamp: str
    champion_metrics: dict[str, Any]
    feature_names: list[str]


class DriftReportResponse(BaseModel):
    dataset_drift_detected: bool
    number_of_drifted_features: int
    total_features_evaluated: int
    report_path: str
