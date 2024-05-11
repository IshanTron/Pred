import os
import shutil
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("brain_model_updated.h5")

# Create a folder to store uploaded images
UPLOAD_FOLDER = "uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    # Get the image path from the request
    data = request.json
    image_path = data["image_path"]

    # Generate a new filename
    filename = os.path.basename(image_path)
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, filename)

    # Copy the image to the uploaded_images folder
    shutil.copy(image_path, uploaded_image_path)

    # Load and preprocess the copied image
    img = cv2.imread(uploaded_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis=0)

    # Perform classification
    predictions = model.predict(img)
    class_names = ["No Tumor", "Tumor"]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]

    # Return the classification result
    return jsonify(
        {"result": f"{predicted_class_name} (Confidence: {confidence_score:.2f})"}
    )


if __name__ == "__main__":
    app.run(debug=True)
