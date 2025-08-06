# @title 🧠 Training and Testing on Deriv Candle Data by Sanne Karibo
from google.colab import files
import os, asyncio, json, time
import importlib.util
import logging
logging.basicConfig(level=logging.INFO, format="🔧 %(message)s")

print("📤 Upload your `Mind.py` file...")
uploaded = files.upload()

# Save the uploaded Python file
for filename in uploaded:
    if filename.endswith(".py"):
        model_filename = filename
        break
else:
    raise RuntimeError("❌ No valid .py file uploaded.")

print(f"✅ Uploaded: {model_filename}")

# ✅ Install dependencies
!pip install -q loguru torch numpy requests cloudinary websockets nest_asyncio

# === 🧠 Load Mind dynamically ===
spec = importlib.util.spec_from_file_location("Mind_module", model_filename)
Mind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Mind)

# === 🔁 Async Setup ===
import nest_asyncio
nest_asyncio.apply()
import websockets
import numpy as np

# === 🌐 Deriv getCandles() WebSocket function ===
async def getCandles(count=100020, granularity=60, symbol="stpRNG"):
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    candles = []
    end = "latest"
    remaining = count

    async with websockets.connect(uri) as ws:
        while remaining > 0:
            batch_size = min(5000, remaining)
            await ws.send(json.dumps({
                "ticks_history": symbol,
                "end": end,
                "count": batch_size,
                "style": "candles",
                "granularity": granularity
            }))
            res = json.loads(await ws.recv())
            if "error" in res:
                raise RuntimeError(f"❌ Deriv API Error: {res['error']['message']}")

            data = res.get("candles", [])
            if not data:
                break

            candles = data + candles
            remaining -= len(data)
            end = data[0]["epoch"] - 1  # Go backward
            await asyncio.sleep(0.1)

    if not candles:
        raise ValueError("❌ No candle data fetched.")

    return list(reversed(candles[-count:]))  # ✅ REVERSED: newest candle is at index 0

# === 🔃 Fetch ~300,000 candles for training
candle_data = asyncio.run(getCandles(300000))
print(f"🕯️ Received {len(candle_data)} candles ✔️")

# === 🧪 Prepare HL pairs for Mind model ===
ohlc_sanne = np.stack([[c["high"], c["low"]] for c in candle_data], axis=0)
print(f"🥂 Prepared data for training, sample: {ohlc_sanne[0]}")

# 🧠 Initialize Mind model
mind = Mind.Mind(sequence_length=20, download_on_init=True)
logging.info(f"🧠 Loaded model. Continuing training:= {mind.model_loaded_successfully}")

# 🏋️ Train model
result = mind.learn(ohlc_sanne, epochs=100, lr=0.001, batch_size=64)
mind.sleep()
print(f"✅ Mind has been trained and uploaded to Cloudinary.\n ✊ Result = {result}")

# === 🧪 Live Prediction Test: fetch latest 21 candles
latest_21 = asyncio.run(getCandles(21))
test_data = np.stack([[c["high"], c["low"]] for c in latest_21], axis=0)

# ✅ Predict most recent candle[0] using candle[1] to candle[20]
input_seq = test_data[1:]      # candles[1] to [20] (older 20 candles)
true_candle = test_data[0]     # candle[0] = most recent, actual future

# 🧠 Predict
pred = mind.predict(input_seq)

# 🧾 Display results
print("\n🧠 Prediction vs Actual:")
print(f"🔮 Predicted High: {pred['Predicted High']:.5f}, Low: {pred['Predicted Low']:.5f}")
print(f"📈 Actual    High: {true_candle[0]:.5f}, Low: {true_candle[1]:.5f}")
