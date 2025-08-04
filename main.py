import os
import sys
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from contextlib import asynccontextmanager
from mind import Mind

# --- Constants ---
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
APP_ID = os.getenv("DERIV_APP_ID", "1089")
SYMBOL = "stpRNG"
GRANULARITY = 60
SEQUENCE_LENGTH = 21  # Fetch 21 candles, use last 20 starting at index 1

# --- Logger Setup ---
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# --- Model Reload Task ---
async def reload_model_daily(app: FastAPI):
    while True:
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (next_midnight - now).total_seconds()
        logger.info(f"⏳ Waiting {wait_seconds / 3600:.2f} hours until midnight model reload...")
        await asyncio.sleep(wait_seconds)

        try:
            app.state.mind = Mind(sequence_length=SEQUENCE_LENGTH - 1, download_on_init=True)
            logger.info("🔁 Model reloaded successfully at midnight.")
        except Exception as e:
            logger.error(f"❌ Failed to reload model: {str(e)}")

# --- Lifespan Context ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚙️ Waking Mind... Attempting to load model from the cloud.")
    try:
        app.state.mind = Mind(sequence_length=SEQUENCE_LENGTH - 1, download_on_init=True)
        logger.success("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {str(e)}")
        sys.exit(1)

    asyncio.create_task(reload_model_daily(app))
    yield
    # Optional cleanup

# --- App Initialization ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Candle Fetch ---
async def get_candles():
    payload = {
        "ticks_history": SYMBOL,
        "count": SEQUENCE_LENGTH,
        "granularity": GRANULARITY,
        "end": "latest",
        "style": "candles"
    }

    for attempt in range(3):
        try:
            async with websockets.connect(f"{DERIV_WS_URL}?app_id={APP_ID}") as ws:
                await ws.send(json.dumps(payload))
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(response)

                if "candles" in data and isinstance(data["candles"], list):
                    candles = [[c["high"], c["low"]] for c in data["candles"]]
                    if len(candles) >= SEQUENCE_LENGTH:
                        return candles
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Candle fetch failed - {str(e)}")
            await asyncio.sleep(1)

    raise HTTPException(status_code=504, detail="Failed to fetch candle data")

# --- API Endpoints ---
@app.get("/health")
async def health_check(request: Request):
    mind = request.app.state.mind
    return {"status": "ok", "model_loaded": mind.model_loaded_successfully}

@app.get("/predict")
async def predict(request: Request):
    mind = request.app.state.mind

    if not mind.model_loaded_successfully:
        raise HTTPException(status_code=503, detail="Model not loaded")

    candles = await get_candles()
    if len(candles) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=422, detail="Not enough candle data")

    try:
        input_sequence = candles[1:]  # Use candles from index 1 to 20
        last_candle = input_sequence[0]  # candles[1]
        prediction = mind.predict(input_sequence)

        return {
            "predicted_high": prediction["Predicted High"],
            "predicted_low": prediction["Predicted Low"],
            "last_candle_high": last_candle[0],
            "last_candle_low": last_candle[1]
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# --- Run the app if needed ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
