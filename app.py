from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnclassifier.utils.common import decodeImage
from cnnclassifier.pipeline.predict import PredictionPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "input_image.jpg"
        self.classifier = PredictionPipeline(filepath = self.filename)

@app.route("/", methods = ["GET"])
@cross_origin()
def home():
    return render_template("index.html")

# @app.route("/train", methods = ["GET", "POST"])
# @cross_origin()
# def trainRoute():
#     os.system("main.py")
#     return "Training Completed Successfully"

@app.route("/predict", methods = ["POST"])
@cross_origin()
def predictRoute():
    image = request.json["image"]
    decodeImage(image, clApp.filename)
    prediction, confidence = clApp.classifier.predict()
    return jsonify({"prediction": prediction, "confidence": confidence})

if __name__=="__main__":
    clApp = ClientApp()
    app.run(host = "0.0.0.0", port = 8080)