"""JSON inference endpoints for the car price prediction service."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from car_price_prediction.model import predict as prediction_service
from car_price_prediction.schemas import CarFeatures, PredictionResponse


router = APIRouter(prefix="/api", tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict used car price",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The trained model artifact is not available."
        }
    },
)
async def predict_api(features: CarFeatures) -> PredictionResponse:
    """Predict car price from a validated JSON payload."""

    try:
        return await asyncio.to_thread(prediction_service.predict_price, features)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
