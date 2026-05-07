from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from config import DATA_PATH
from forecast_service import ForecastingService

service = ForecastingService(data_path=DATA_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.ensure_ready()
    yield


app = FastAPI(title="State Sales Forecast API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/states")
def states() -> dict[str, list[str]]:
    return {"states": service.list_states()}


@app.get("/forecast/{state}")
def forecast_state(state: str, horizon: int = Query(default=8, ge=1, le=52)) -> dict:
    try:
        return service.predict_state(state, horizon=horizon)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/summary")
def summary() -> dict:
    result = service.training_summary()
    return {"rows": result.to_dict(orient="records")}


@app.post("/forecast/bulk")
def bulk_forecast(horizon: int = Query(default=8, ge=1, le=52)) -> dict:
    return {"results": service.bulk_forecast(horizon=horizon)}
