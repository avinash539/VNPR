from wsgiref import  simple_server
from flask import Flask
import cv2
from predict_image import RecognizeNumberPlate
from modules.getNumberPlateValues import numberPlateOcr


app = Flask(__name__)

imagePath = r"E:\Education\Computer\Data Science\Deep Learning CV & NLP\Project\Computer Vision\VNPR\data\image\car8.png"
croppedImagePath = "data/detected/croppedImage1.jpg"

# Select the library for Ocr 0:- Easy OCR and 1:- Tesseract OCR
ocrLib = 1

class Vnpr:
    def __init__(self):
        # The file path of the model
        self.modelPath = "inference_graph/frozen_inference_graph.pb"
        self.labelPath = "inference_graph/classes.pbtxt"
        self.num_classes = 1
        self.min_confidence = 0.5
        self.recognizeNumberPlate = RecognizeNumberPlate()

@app.route("/")
def welcome():
    return "Welcome to flask"

@app.route("/predict", methods=["GET"])
def getPrediction():
    try:
        labelledImage = vnpr.recognizeNumberPlate.predictImages(imagePath, croppedImagePath, vnpr.recognizeNumberPlate)
        if labelledImage is not None:
            numberPlateValue = numberPlateOcr(croppedImagePath, ocrLib)
            numberPlateValue = numberPlateValue[0] if ocrLib == 0 else numberPlateValue
            return "Licence Number is: {}".format(numberPlateValue)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    vnpr = Vnpr()
    # app.run(debug=True)
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

