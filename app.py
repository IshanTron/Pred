"""from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf


app = Flask(__name__, static_url_path="/static")

# Load the trained model
model = tf.keras.models.load_model("brain_model_updated.h5")


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

    # Remove the first "/" from the image path
    if image_path.startswith("/"):
        image_path = image_path[1:]

    # Load and preprocess the image
    img = cv2.imread(image_path)
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
    app.run(debug=True)"""

"""from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_url_path="/static")


from flask import Flask, request, Response

app = Flask(__name__)


# Load the trained model
model = tf.keras.models.load_model("brain_model_updated.h5")

# Define the upload folder and allowed extensions
# Define the upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to render the home page
@app.route("/")
def index():
    return render_template("index2.html")


# Route for image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    # Get the image path from the request
    data = request.json
    image_path = data.get("image_path")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    # Load and preprocess the image
    img = cv2.imread(image_path)
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


@app.route("/upload", methods=["POST"])
def upload_image():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            try:
                # Secure filename and remove leading slash
                filename = secure_filename(file.filename.lstrip("/"))
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                return (
                    jsonify(
                        {"message": "Image uploaded successfully", "filename": filepath}
                    ),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"Upload failed: {str(e)}"}), 400
        else:
            return jsonify({"error": "Invalid file type"}), 400

    return jsonify({"error": "Upload failed"}), 400


if __name__ == "__main__":
    app.run(debug=True)
"""


from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_url_path="/static")

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the default model (Brain Tumor)
model_path = "brain_model_updated.h5"
model = tf.keras.models.load_model(model_path)


# Route to render the home page
@app.route("/")
def index():
    return render_template("index3.html")


# Route for image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    # Get the image path and selected disease from the request
    data = request.json
    image_path = data.get("image_path")
    disease = data.get("disease")
    app.logger.info("Image path: %s, Disease: %s", image_path, disease)

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if disease == "Brain Tumor":
        img = cv2.resize(img, (300, 300))
    elif disease == "Chest Cancer":
        img = cv2.resize(img, (224, 224))
    else:
        return jsonify({"error": "Invalid disease"}), 400

    img = np.expand_dims(img, axis=0)

    # Load the appropriate model based on the selected disease
    if disease == "Brain Tumor":
        model_path = "brain_model_updated.h5"
        class_names = ["No Tumor", "Tumor"]
    elif disease == "Chest Cancer":
        model_path = "chest-model.h5"
        class_names = ["Normal", "Cancer"]
    else:
        return jsonify({"error": "Invalid disease"}), 400

    model = tf.keras.models.load_model(model_path)

    # Perform classification
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]

    # Return the classification result
    return jsonify(
        {"result": f"{predicted_class_name} (Confidence: {confidence_score:.2f})"}
    )


# Route for handling image upload
@app.route("/upload", methods=["POST"])
def upload_image():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            try:
                # Secure filename and remove leading slash
                filename = secure_filename(file.filename.lstrip("/"))
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                return (
                    jsonify(
                        {"message": "Image uploaded successfully", "filename": filepath}
                    ),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"Upload failed: {str(e)}"}), 400
        else:
            return jsonify({"error": "Invalid file type"}), 400

    return jsonify({"error": "Upload failed"}), 400


if __name__ == "__main__":
    app.run(debug=True)
