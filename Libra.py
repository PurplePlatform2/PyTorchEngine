import os
import time
import zipfile
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from loguru import logger
import cloudinary
import cloudinary.uploader
import cloudinary.api

# =============================
#   Cloudinary Config
# =============================
cloudinary.config(
    cloud_name="dj4bwntzb",
    api_key="354656419316393",
    api_secret="M-Trl9ltKDHyo1dIP2AaLOG-WPM",
    secure=True
)

# =============================
#   LSTM Classifier Model
# =============================
class Libra_ModelJane(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.3):
        super(Libra_ModelJane, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.layernorm = LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        normed = self.layernorm(last_hidden)
        return self.fc(normed)

# =============================
#   Libra Manager Class
# =============================
class Libra:
    def __init__(self, sequence_length=10, input_size=5, device=None, download_on_init=False):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Libra_ModelJane(input_size=input_size).to(self.device)
        self.norm_params = {}
        self.model_loaded_successfully = False

        # Paths & Cloud IDs
        self.MODEL_PUBLIC_ID = "libra_jane_model_v1"
        self.MODEL_LOCAL_PATH = "/tmp/libra_model.pt"
        self.ZIP_LOCAL_PATH = "/tmp/libra_model.zip"

        logger.info(f"📈 Libra initialized on device: {self.device}")

        if download_on_init:
            self.awaken()
        if not self.model_loaded_successfully:
            logger.warning("⚠ No pre-trained model loaded. Please train first.")

    # -----------------------------
    #   Normalization Helpers
    # -----------------------------
    def _normalize(self, data, is_training=False):
        if is_training or not self.norm_params:
            self.norm_params['min'] = np.min(data, axis=0)
            self.norm_params['max'] = np.max(data, axis=0)
        return (data - self.norm_params['min']) / (self.norm_params['max'] - self.norm_params['min'] + 1e-8)

    def _denormalize(self, data):
        return data * (self.norm_params['max'] - self.norm_params['min'] + 1e-8) + self.norm_params['min']

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length - 1):
            seq = data[i:i+self.sequence_length]
            target = 1.0 if data[i+self.sequence_length][3] > data[i+self.sequence_length-1][3] else 0.0
            X.append(seq)
            y.append(target)
        return np.array(X), np.array(y)

    # -----------------------------
    #   Training
    # -----------------------------
    def learn(self, data, epochs=50, lr=0.001, batch_size=32, patience=5):
        data_scaled = self._normalize(data, is_training=True)
        X, y = self._create_sequences(data_scaled)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("⏹ Early stopping triggered.")
                    break

        logger.success("✅ Training complete.")
        return best_loss

    # -----------------------------
    #   Prediction
    # -----------------------------
    def predict(self, last_candles):
        if len(last_candles) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} candles, got {len(last_candles)}")
        arr = np.array(last_candles, dtype=np.float32)
        scaled = self._normalize(arr, is_training=False)
        tensor_input = torch.tensor(scaled[None], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            prob_up = self.model(tensor_input).item()
        return {"up": prob_up, "down": 1 - prob_up}

    # -----------------------------
    #   Cloud Save
    # -----------------------------
    def sleep(self, max_attempts=3):
        if not self.norm_params:
            logger.error("❌ Cannot save: No normalization params found.")
            return False
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'norm_params': self.norm_params
        }, self.MODEL_LOCAL_PATH)

        try:
            with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(self.MODEL_LOCAL_PATH, os.path.basename(self.MODEL_LOCAL_PATH))
        except Exception as e:
            logger.error(f"Zip creation failed: {e}")
            return False

        for attempt in range(max_attempts):
            try:
                cloudinary.uploader.upload(
                    self.ZIP_LOCAL_PATH,
                    resource_type='raw',
                    public_id=self.MODEL_PUBLIC_ID,
                    overwrite=True
                )
                logger.success("☁ Model saved to Cloudinary.")
                return True
            except Exception as e:
                logger.warning(f"Upload attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("All Cloudinary uploads failed.")
        return False

    # -----------------------------
    #   Cloud Load
    # -----------------------------
    def awaken(self, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                url = cloudinary.utils.cloudinary_url(
                    f"{self.MODEL_PUBLIC_ID}.zip",
                    resource_type='raw',
                    sign_url=True
                )[0]
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(self.ZIP_LOCAL_PATH, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))

                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state['model_state_dict'])
                self.norm_params = state['norm_params']
                self.model.eval()
                self.model_loaded_successfully = True
                logger.success("✅ Model loaded from Cloudinary.")
                return True
            except Exception as e:
                logger.warning(f"Load attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("❌ Could not load model from Cloudinary.")
        return False
