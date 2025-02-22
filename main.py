from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()

#save the model
model.save("Effinetb7.h5")

# Load the trained model
model = tf.keras.models.load_model("Effinetb7.h5")

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
    return img_array

# Endpoint to handle image uploads and make predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")  # Open image
    processed_image = preprocess_image(image)

    # Run prediction
    prediction = model.predict(processed_image)
    result = "Diabetic Retinopathy Detected" if np.argmax(prediction) == 0 else "No Diabetic Retinopathy"

    return {"prediction": result, "confidence": float(np.max(prediction))}

