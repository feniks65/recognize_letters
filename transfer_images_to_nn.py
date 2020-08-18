import numpy as np
import os
import array
import subprocess
import pytesseract
import pickle

from PIL import Image, ImageStat, ImageChops

SAMPLES_DIR = "/Users/Marta/Desktop/STUDIA/PRACA_MAGISTERSKA/zebrane_do_uczenia_sieci_nn/przeksztalcone/do_czyszczenia/do_nauczenia"

#COUNT_IMAGES = 2

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def checkIfImageHasProperSize(im):
    width, height = im.size
    if width == 28 and height == 28:
        return True
    return False

def checkIfImageIsGreyScaled(im):
    im.convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum)/3 == stat.sum[0]:
        return True
    return False

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode == "1":
        return True

    if im.mode not in ("L", "RGB"):
        return False
        #raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0:
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0:
            return False
    return True

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


def findAllPngsInDir(dir_path):
    #subprocess.call("find " + SAMPLES_DIR + " -name \"*.png\" -exec ./convert_for_nn.sh {} {} \;", shell=True)
    labels = list() #np.empty([COUNT_IMAGES])
    images = list() #np.empty([COUNT_IMAGES, 28, 28])
    i = 1
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".png"):
            im = Image.open(os.path.join(dir_path, file_name))
            if checkIfImageHasProperSize(im):
                if is_greyscale(im):
                    print "Plik dla ktorego szukam litery " + file_name
                    im.show()
                    text_label =  raw_input('Podaj jaka jest litera rozpoznana dla tego obrazka?')#pytesseract.image_to_string(im)#raw_input('Podaj jaka jest litera rozpoznana dla tego obrazka?')
                    #label = ord(text_label)
                    #if text_label.isdigit():
                    if len(text_label) > 0:
                        label = ord(text_label)
                        if label in range(0,128):
                            i = i + 1
                            print i
                            labels.append(label)
                            images.append(readFileImageToNumpyArray(im))
                            try:
                                with open('labels_tmp.lst', 'wb') as labels_tmp_fd, open('images_tmp.lst', 'wb') as images_tmp_fd:
                                    pickle.dump(labels, labels_tmp_fd)
                                    pickle.dump(images, images_tmp_fd)
                            except Exception as e:
                                continue
                    #labels = np.concatenate(labels, label)
                    #print label
                    #labels.append(label)
                    #print "Image z funkcji:"
                    #print readFileImageToNumpyArray(im)
                    #np.append(images, readFileImageToNumpyArray(im))
                    #images.append(readFileImageToNumpyArray(im))
    print i
    return np.array(labels), np.array(images)

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
    #image_for_test_path = os.path.join(SAMPLES_DIR, "line_0_index_0_char_G.png")
    #image_for_test = readImageForTest(image_for_test_path)
    #np.save('image_for_test.npy', [image_for_test])

    labels, images = findAllPngsInDir(SAMPLES_DIR)
    print "Images:\n"
    #print images
    np.save('images_11_receipts.npy', images)
    print "Labels:\n"
    #print labels
    np.save('labels_11_receipts.npy', labels)
