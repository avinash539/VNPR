import cv2
import numpy as np
import tensorflow as tf
from modules.utils import label_map_util

num_classes = 1
modelPath = "inference_graph/frozen_inference_graph.pb"
labelPath = "inference_graph/classes.pbtxt"

# Initializing the model
model = tf.Graph()

# Creating a context manager
with model.as_default():
    # Initialinzing the graph defination
    grapgDef = tf.GraphDef()

    # Loading the graph
    with tf.gfile.GFile(modelPath, 'rb') as f:
        serializedGraph = f.read()
        grapgDef.ParseFromString(serializedGraph)
        tf.import_graph_def(grapgDef, name="")

labelMap = label_map_util.load_labelmap(labelPath)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=num_classes, use_display_name=True)


class Predictor:
    def __init__(self, model, sess, categoryIdx):
        self.sess = sess
        self.categoryIdx = categoryIdx
        self.imageTensor = model.get_tensor_by_name("image_tensor:0")
        self.boxesTensor = model.get_tensor_by_name("detection_boxes:0")
        self.scoresTensor = model.get_tensor_by_name("detection_scores:0")
        self.classesTensor = model.get_tensor_by_name("detection_classes:0")
        self.numDetections = model.get_tensor_by_name("num_detections:0")


    def predictNumberPlates(self, image):

        # Preparing the image for inference input
        # inputImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputImage = np.expand_dims(image, axis=0)
        (boxes, scores, labels, num) = self.sess.run(
            [self.boxesTensor, self.scoresTensor, self.classesTensor, self.numDetections],
            feed_dict={self.imageTensor: inputImage}
        )

        # squeezing the lists into a single dimension
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)

        return boxes, scores, labels