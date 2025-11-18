from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()

# Load model once
model = tf.keras.models.load_model("model.h5")

class InputPayload(BaseModel):
    data: dict    # values MUST be numeric, in correct order

def to_np(data_dict):
    # Convert dict → np array in order of keys as sent by client
    arr = np.array(list(data_dict.values()), dtype=float)
    return arr.reshape(1, -1)

@app.post("/predict")
def predict(payload: InputPayload):
    x = to_np(payload.data)
    y = model.predict(x)
    return {"prediction": float(y[0][0])}

@app.get("/")
def root():
    return {"status": "model loaded"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
