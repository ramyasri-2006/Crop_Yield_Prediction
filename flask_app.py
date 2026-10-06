from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("/home/venumadhavi/mysite/crop_yield.pkl", "rb") as f:
    model = pickle.load(f)

# Load Area LabelEncoder
with open("/home/venumadhavi/mysite/area_encoder.pkl", "rb") as f:
    area_encoder = pickle.load(f)

# Load Item LabelEncoder
with open("/home/venumadhavi/mysite/item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Read form data
        country = request.form["area"]
        crop = request.form["item"]

        year = float(request.form["year"])
        rainfall = float(request.form["rainfall"])
        pesticides = float(request.form["pesticides"])
        avg_temp = float(request.form["avg_temp"])

        # Encode country and crop
        area = area_encoder.transform([country])[0]
        item = item_encoder.transform([crop])[0]

        # Prepare input
        new_data = np.array([[area,
                              item,
                              year,
                              rainfall,
                              pesticides,
                              avg_temp]])

        # Prediction
        prediction = model.predict(new_data)

        return render_template(
            "result.html",
            prediction=round(prediction[0], 2),
            country=country,
            crop=crop,
            year=year
        )

    # Display all countries and crops
    return render_template(
        "index.html",
        countries=area_encoder.classes_,
        crops=item_encoder.classes_
    )