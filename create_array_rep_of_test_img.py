#!/bin/python
import numpy as np
import os
from PIL import Image

SAMPLES_DIR = "/Users/Marta/Desktop/STUDIA/PRACA_MAGISTERSKA/tensorflow_15102018/test_folder2/sample1"

def readFileImageToNumpyArray(im):
    image_array = np.empty([28, 28])
    w,h = im.size
    image_list = list()
    for i in range(h):
        row = list() #array.array('i')
        for j in range(w):
            #r,g,b = im.getpixel((i,j))
            d = im.getpixel((i,j))
            row.append(d)
        #print "np.array(row):"
        #print np.array(row)
        #image_array = np.append(image_array, np.array(row))
        image_list.append(row)
        #print "image_list:"
        #print image_list
    #print "Image array:"
    #print image_array
    return image_list #image_array

def readImageForTest(image_path):
    if image_path is not None:
        image_out = list()
        im = Image.open(image_path)
        image_out = readFileImageToNumpyArray(im)
        return np.array(image_out)
    else:
        return None


if __name__ == "__main__":
    np.set_printoptions(precision=0, suppress=True)
    image_for_test_path = os.path.join(SAMPLES_DIR, "line_1_index_0_char_S.png")
    image_for_test = readImageForTest(image_for_test_path)
    im = Image.open(image_for_test_path)
    im.show()
    np.save('image_for_test.npy', [image_for_test])
