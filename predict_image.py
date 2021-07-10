import cv2
from PIL import Image
import tensorflow as tf
from modules.utils import label_map_util
from modules.Predictor import Predictor
from modules.licence_plate_finder import LicencePlateFinder


class RecognizeNumberPlate:
    def __init__(self):

        # The file path of the model
        self.modelPath = "inference_graph/frozen_inference_graph.pb"
        self.labelPath = "inference_graph/classes.pbtxt"
        self.num_classes = 1
        self.min_confidence = 0.5

        # Initialize the model
        self.model = tf.Graph()

        with self.model.as_default():
            # Initialize the graph defination
            self.graphDef = tf.GraphDef()

            # Load graph
            with tf.gfile.GFile(self.modelPath, "rb") as f:
                self.serializedGraph = f.read()
                self.graphDef.ParseFromString(self.serializedGraph)
                tf.import_graph_def(self.graphDef, name="")
                print('End of loading model...')

        # Load the labels
        self.labelMap = label_map_util.load_labelmap(self.labelPath)
        self.category = label_map_util.convert_label_map_to_categories(
            self.labelMap, max_num_classes=self.num_classes,
            use_display_name=True
        )
        self.categoryIdx = label_map_util.create_category_index(self.category)

        # Instance of licence plate finder class
        self.licencePlateFinder = LicencePlateFinder(self.categoryIdx, self.min_confidence)



    def predictImages(self, imagePath, croppedImagePath, recognizeNumberPlate):

        # Create a session to perform inference
        with recognizeNumberPlate.model.as_default():
            with tf.Session(graph=recognizeNumberPlate.model) as sess:
                predictor = Predictor(recognizeNumberPlate.model, sess, recognizeNumberPlate.categoryIdx)

                # Loading Image
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Performing Inference and get the licence plate
                boxes, scores, labels = predictor.predictNumberPlates(image)

                # Finding localized number plate
                numberPlateFound, predictedBoxes, predictedScores = self.licencePlateFinder.findNumberPlate(boxes, scores, labels)
                imageLabelled = self.getBoundingBox(image, predictedBoxes, imagePath, croppedImagePath)

                return imageLabelled


    def getBoundingBox(self, image, plateBoxes, imagePath, croppedImagePath):

        (H,W) = image.shape[:2]
        for plateBox in plateBoxes:
            # Draw the rectangular box in red colour
            # scale the bounding box from the range [0, 1 to  [W, H]
            (startY, startX, endY, endX) = plateBox
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)

            try:
                image_obj = Image.open(imagePath)
                cropped_image = image_obj.crop((startX, startY, endX, endY))
                cropped_image = cropped_image.convert("L")
                cropped_image.save(croppedImagePath)
                return cropped_image
            except Exception as e:
                print(e)


