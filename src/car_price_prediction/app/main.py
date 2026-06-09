from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from car_price_prediction import config
from car_price_prediction.app.api import router as api_router
from car_price_prediction.app.forms import form_fields
from car_price_prediction.model import predict as prediction_service
from car_price_prediction.schemas import CarFeatures, HealthResponse, PredictionResponse


APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    prediction_service.warm_model_bundle()
    form_fields()
    yield


app = FastAPI(title="Car Price Prediction", version="0.1.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)


def empty_form_values() -> dict[str, Any]:
    return {field["name"]: field.get("default", "") for field in form_fields()}


def template_context(
    request: Request,
    values: dict[str, Any] | None = None,
    prediction: PredictionResponse | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "request": request,
        "fields": form_fields(),
        "values": values or empty_form_values(),
        "prediction": prediction,
        "errors": errors or [],
        "model_available": prediction_service.model_available(),
    }


def validation_errors_to_text(error: ValidationError) -> list[str]:
    messages = []
    for item in error.errors():
        field = item.get("loc", ["field"])[0]
        message = item.get("msg", "Invalid value")
        messages.append(f"{field}: {message}")
    return messages


def build_features_from_form(
    condition: str,
    vehicle_brand: str,
    production_year: int,
    mileage_km: int,
    power_hp: int,
    displacement_cm3: int,
    fuel_type: str,
    drive: str,
    transmission: str,
    body_type: str,
    doors_number: int,
) -> CarFeatures:
    return CarFeatures(
        Condition=condition,
        Vehicle_brand=vehicle_brand,
        Production_year=production_year,
        Mileage_km=mileage_km,
        Power_HP=power_hp,
        Displacement_cm3=displacement_cm3,
        Fuel_type=fuel_type,
        Drive=drive,
        Transmission=transmission,
        Type=body_type,
        Doors_number=doors_number,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        name="index.html",
        context=template_context(request),
        request=request,
    )


@app.post("/form/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    condition: str = Form(..., alias="Condition"),
    vehicle_brand: str = Form(..., alias="Vehicle_brand"),
    production_year: int = Form(..., alias="Production_year"),
    mileage_km: int = Form(..., alias="Mileage_km"),
    power_hp: int = Form(..., alias="Power_HP"),
    displacement_cm3: int = Form(..., alias="Displacement_cm3"),
    fuel_type: str = Form(..., alias="Fuel_type"),
    drive: str = Form(..., alias="Drive"),
    transmission: str = Form(..., alias="Transmission"),
    body_type: str = Form(..., alias="Type"),
    doors_number: int = Form(..., alias="Doors_number"),
) -> HTMLResponse:
    form_values = {
        "Condition": condition,
        "Vehicle_brand": vehicle_brand,
        "Production_year": production_year,
        "Mileage_km": mileage_km,
        "Power_HP": power_hp,
        "Displacement_cm3": displacement_cm3,
        "Fuel_type": fuel_type,
        "Drive": drive,
        "Transmission": transmission,
        "Type": body_type,
        "Doors_number": doors_number,
    }

    try:
        features = build_features_from_form(
            condition,
            vehicle_brand,
            production_year,
            mileage_km,
            power_hp,
            displacement_cm3,
            fuel_type,
            drive,
            transmission,
            body_type,
            doors_number,
        )
        prediction = await asyncio.to_thread(prediction_service.predict_price, features)
    except ValidationError as error:
        return templates.TemplateResponse(
            name="index.html",
            context=template_context(
                request,
                values=form_values,
                errors=validation_errors_to_text(error),
            ),
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    except FileNotFoundError as error:
        return templates.TemplateResponse(
            name="index.html",
            context=template_context(request, values=form_values, errors=[str(error)]),
            request=request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return templates.TemplateResponse(
        name="index.html",
        context=template_context(request, values=form_values, prediction=prediction),
        request=request,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_available=prediction_service.model_available(),
        model_path=str(config.MODEL_PATH.relative_to(config.REPO_ROOT)),
    )


def serve() -> None:
    uvicorn.run(
        "car_price_prediction.app.main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=config.APP_RELOAD,
    )


if __name__ == "__main__":
    serve()
