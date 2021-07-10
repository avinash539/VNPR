
class LicencePlateFinder:

    def __init__(self, categoryIdx, minConfidence):
        self.categoryidx = categoryIdx
        self.minConfidence = minConfidence

    def findNumberPlate(self, boxes, scores, labels):
        numberPlateFound = False
        plateBoxes = []
        plateScores = []

        # loop to get all the boxes, scores and labels
        for (i, (box, score, label)) in enumerate(zip(boxes, scores, labels)):
            if score < self.minConfidence:
                continue
            label = self.categoryidx[label]
            label = "{}".format(label["name"])

            if label == "licence":
                numberPlateFound = True
                plateBoxes.append(box)
                plateScores.append(score)

        return numberPlateFound, plateBoxes, plateScores
