import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flask import Flask, render_template, request, send_from_directory
from PIL import Image

app = Flask(__name__)

classifier = None

# Upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Food information
food_info = {
    "pizza": {
        "calories": "266 kcal",
        "type": "Italian"
    },

    "burger": {
        "calories": "295 kcal",
        "type": "Fast Food"
    },

    "samosa": {
        "calories": "262 kcal",
        "type": "Indian Snack"
    },

    "chicken_curry": {
        "calories": "243 kcal",
        "type": "Indian"
    },

    "garlic_bread": {
        "calories": "350 kcal",
        "type": "Italian"
    }
}

# Prediction history
history = []

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

# Main route
@app.route("/", methods=["GET", "POST"])
def home():

    predictions = []
    image_path = None
    food_data = None

    if request.method == "POST":

        file = request.files["image"]

        # Save image
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # Open image
        image = Image.open(filepath)

        # Lazy load model
        global classifier

        if classifier is None:
            from transformers import pipeline

            classifier = pipeline(
                "image-classification",
                model="nateraw/food"
            )

        # Predict
        result = classifier(image)

        # Top 3 predictions
        for item in result[:3]:

            label = item["label"]

            score = round(
                item["score"] * 100,
                2
            )

            predictions.append(
                {
                    "label": label,
                    "score": score
                }
            )

        # Top prediction
        top_food = predictions[0]["label"]

        # Save history
        history.insert(0, top_food)

        # Keep only last 5
        history[:] = history[:5]

        # Food details
        food_data = food_info.get(
            top_food,
            {
                "calories": "Not Available",
                "type": "Unknown"
            }
        )

        # Image path
        image_path = file.filename

    return render_template(
        "index.html",
        predictions=predictions,
        image_path=image_path,
        food_data=food_data,
        history=history
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)