import cv2
import numpy as np
import io
import os
import sys
import shutil
import subprocess

from PIL import Image
import pytesseract

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

def recognize_text(imageFilepath):
    text = pytesseract.image_to_string(Image.open(imageFilepath))
    text = text.replace(' ', '')
    text = text.replace('\t', '')
    lines = text.splitlines()
    return lines

def showImage(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)

def extractBordingPxs(numBlackPixVert, threshold):
    bordingRows = []
    prevVal = 0
    counter = 0
    for px in numBlackPixVert:
        if px > threshold and prevVal <= threshold:
            bordingRows.append(counter)
        elif px <= threshold and prevVal > threshold:
            bordingRows.append(counter)
        prevVal = px
        counter = counter + 1
    return bordingRows

def getLineRects(image, boardingPxs, width):
    rects = []
    for i in range(len(boardingPxs)-1):
        if i % 2 == 0:
            rects.append(image[boardingPxs[i]:boardingPxs[i + 1], 0:width])
    return rects


def cutRows(image, threshold):
    height, width = image.shape
    numBlackPixVert = []
    for i in range(0, height):
        numBlackPixInRow = 0
        for j in range(0, width):
            if image[i, j] ==  0:
                numBlackPixInRow = numBlackPixInRow + 1
        numBlackPixVert.append(numBlackPixInRow)
    bordingPxs = extractBordingPxs(numBlackPixVert, threshold)
    return getLineRects(image, bordingPxs, width)

def rotate(image, angle, center=None, scale=1.0):
    #showImage(image)
    (h, w) = image.shape[:2]

    if center is None:
        (cX, cY) = (w // 2, h // 2)
    else:
        (cX, cY) = center

    # Perform the rotation
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    newWidth = int((h * sin) + (w * cos))
    newHeight = int((h * cos) + (w * sin))

    M[0, 2] += (newWidth / 2) - cX
    M[1, 2] += (newHeight / 2) - cY

    rotated = cv2.warpAffine(image, M, (newWidth, newHeight))
    return rotated

def ocrImageInCloudVision(imagePath):
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = imagePath

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print("Label from GCV:")
        print(label.description)




if __name__ == "__main__":
    if len(sys.argv) == 2:
        receipt_dir = sys.argv[1]
        if not os.path.exists(receipt_dir):
            os.makedirs(receipt_dir)
        else:
            shutil.rmtree(receipt_dir, ignore_errors=True)
            os.makedirs(receipt_dir)
        image = cv2.imread(receipt_dir+".png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        lines = cutRows(thresh, 10)
        j=0
        #recognized_lines = recognize_text(receipt_dir)
        for line in lines:
            rotatedLine = rotate(line, -90)
            #showImage(rotatedLine)
            chars = cutRows(rotatedLine, 1)
            print str(chars)
            i=0
            #recognized_chars_in_line = list(recognized_lines[j])
            for char in chars:
                unrotatedChar = rotate(char, 90)
                cv2.imwrite(receipt_dir+'/line_'+str(j)+'_index_'+str(i)+'_char_'+str(i)+'.png', unrotatedChar)
                i = i + 1
            j = j +1
        #ocrImageInCloudVision('char0_23.png')
        subprocess.call("find " + receipt_dir + " -name \"*.png\" -exec ./convert_for_nn.sh {} {} \;", shell=True)
