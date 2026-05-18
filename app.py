import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import gc

app = Flask(__name__)

classifier = None

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

food_info = {
    "pizza": {
        "calories": "266 kcal",
        "type": "Italian"
    },
    "burger": {
        "calories": "295 kcal",
        "type": "Fast Food"
    }
}

history = []

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

@app.route("/", methods=["GET", "POST"])
def home():

    predictions = []
    image_path = None
    food_data = None

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        image = Image.open(filepath)

        global classifier

        if classifier is None:
            from transformers import pipeline

            classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50"
            )

        try:
            result = classifier(image)

        except Exception as e:
            return f"Prediction Error: {str(e)}"

        gc.collect()

        for item in result[:1]:

            label = item["label"]

            score = round(
                item["score"] * 100,
                2
            )

            predictions.append({
                "label": label,
                "score": score
            })

        if predictions:
            top_food = predictions[0]["label"]
        else:
            top_food = "Unknown"

        history.insert(0, top_food)
        history[:] = history[:5]

        food_data = food_info.get(
            top_food,
            {
                "calories": "Not Available",
                "type": "Unknown"
            }
        )

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