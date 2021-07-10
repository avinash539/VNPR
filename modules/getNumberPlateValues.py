import cv2
import easyocr
import os
import re
import time
import numpy as np
import pytesseract

# labelledImage = r"E:\Education\Computer\Data Science\Deep Learning CV & NLP\Project\Computer Vision\VNPR\data\detected\croppedImage.jpg"
filePath = r"data/detected/licence_number.txt"
pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def numberPlateOcr(croppedImagePath, ocrLib):

    if ocrLib == 0:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(croppedImagePath, detail=False)
        with open(filePath, 'a+') as f:
            f.write(result[0])
        return result
    else:
        plate_num = ""
        image = cv2.imread(croppedImagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilation = cv2.dilate(thresh, rect_kern, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        im2 = gray.copy()
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = im2.shape
            charHeight = height / float(h)
            if charHeight > 6 or charHeight < 1.2: continue
            ratio = h / float(w)
            if ratio < 1.5 or ratio > 10: continue
            charWidth = width / float(w)
            if charWidth > 15: continue
            area = h * w
            if area < 100: continue
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text

        # cv2.imshow("Licence Plate", rect)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        with open(filePath, 'a+') as f:
            f.write(plate_num)
            f.write("\n")
        return plate_num
