from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
model = load_model("image_recognition_model.keras")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    return predicted_class

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            image_path = "temp_image.jpg"
            image_file.save(image_path)

            try:
                predicted_class = predict_class(image_path)
                data["predicted_class"] = int(predicted_class)
                data["success"] = True
            except Exception as e:
                data["error"] = str(e)

    return jsonify(data)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True) 
