from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import json

app = Flask(__name__)

# Load YOLO Model
model = YOLO("best.pt")

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/")
def home():

    try:
        with open("results.json", "r") as f:
            history = json.load(f)
    except:
        history = []

    total = len(history)

    stats = {}

    for item in history:

        category = item["category"].lower().strip()

        # Merge similar classes
        if category in ["plastic"]:
            category = "Plastic"

        elif category in ["red apple", "green apple", "banana", "organic"]:
            category = "Organic"

        elif category in ["metal", "can"]:
            category = "Metal"

        elif category == "glass":
            category = "Glass"

        elif category == "paper":
            category = "Paper"

        elif category == "cardboard":
            category = "Cardboard"

        elif category in ["e-waste", "ewaste"]:
            category = "E-Waste"

        elif category == "medical waste":
            category = "Medical Waste"

        if category in stats:
            stats[category] += 1
        else:
            stats[category] = 1

    return render_template(
        "index.html",
        total=total,
        stats=stats
    )


# -------------------------------
# PREDICT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    if image.filename == "":
        return "Please select an image."

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    # AI Prediction
    results = model.predict(image_path, conf=0.25)

    if len(results[0].boxes) > 0:

        box = results[0].boxes[0]

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        category = model.names[class_id]

        confidence = f"{confidence*100:.2f}%"

    else:

        category = "No Waste Detected"

        confidence = "0%"

    # ---------------------------
    # Disposal Suggestion
    # ---------------------------

    category_lower = category.lower()

    if "plastic" in category_lower:

        suggestion = "🥤 Dispose in the Plastic Recycling Bin."

    elif ("organic" in category_lower or
          "apple" in category_lower or
          "banana" in category_lower or
          "food" in category_lower):

        suggestion = "🌱 Dispose in the Compost Bin."

    elif ("metal" in category_lower or
          "can" in category_lower):

        suggestion = "🥫 Dispose in the Metal Recycling Bin."

    elif "glass" in category_lower:

        suggestion = "🍾 Dispose in the Glass Recycling Bin."

    elif "paper" in category_lower:

        suggestion = "📄 Dispose in the Paper Recycling Bin."

    elif "cardboard" in category_lower:

        suggestion = "📦 Recycle with Paper/Cardboard."

    elif ("e-waste" in category_lower or
          "ewaste" in category_lower):

        suggestion = "💻 Take to an E-Waste Collection Center."

    elif "medical" in category_lower:

        suggestion = "⚠️ Dispose at an Authorized Medical Waste Facility."

    else:

        suggestion = "♻️ Dispose according to local recycling guidelines."

    # ---------------------------
    # Save History
    # ---------------------------

    prediction = {

        "image": image.filename,
        "category": category,
        "confidence": confidence

    }

    try:

        with open("results.json", "r") as f:

            history = json.load(f)

    except:

        history = []

    history.append(prediction)

    with open("results.json", "w") as f:

        json.dump(history, f, indent=4)

    return render_template(

        "result.html",

        image=image.filename,

        category=category,

        confidence=confidence,

        suggestion=suggestion

    )


# -------------------------------
# HISTORY
# -------------------------------
@app.route("/history")
def history():

    try:

        with open("results.json", "r") as f:

            data = json.load(f)

    except:

        data = []

    return render_template(

        "history.html",

        history=data

    )


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":

    app.run(debug=True)