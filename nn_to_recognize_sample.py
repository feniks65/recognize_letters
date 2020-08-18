import tensorflow as tf
import numpy as np
import sys
import os
import re
from PIL import Image
mnist = tf.keras.datasets.mnist

def readFileImageIntoList(image_path):
    im = Image.open(image_path)
    #image_array = np.empty([28, 28])
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
    return [image_list] #image_array

def printNonNones(lines):
    output_text = ""
    for line in lines:
        if line != None:
            #print "\n",
            output_text = output_text + "\n"
            for char in line:
                if char != None:
                    #print char,
                    output_text = output_text + char
    return output_text

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.load('images_11_receipts.npy')
y_train = np.load('labels_11_receipts.npy')
x_test = np.load('images_11_receipts.npy')
y_test = np.load('labels_11_receipts.npy')

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
#model.save_weights('model_11_receipts_weights.h5')
#print "Labels from file:"
#print y_train
#print "Recognized labels:"

if len(sys.argv) == 2:
    folder_path = sys.argv[1]
    lines = [None] * 256
    for i in range(0, 256):
        lines[i] = [None] * 256
    for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                lineMatchObj = re.match( r'line_([0-9]*)_', file_name)
                charMatchObj = re.match( r'.*index_([0-9]*)_*.', file_name)
                line_index = lineMatchObj.group(1)
                char_index = charMatchObj.group(1)
                line_index = int(line_index)
                char_index = int(char_index)
                test_img = np.array(readFileImageIntoList(os.path.join(folder_path, file_name)))
                #print "Below test image:"
                #print test_img
                test_img = test_img / 255.0
                #print "test_img:"
                #print test_img
                try:
                    classes = model.predict_classes(test_img)
                except Exception as e:
                    continue
                (lines[line_index])[char_index] = str(unichr(classes[0]))
                #f = open(os.path.join(dir_path, file_name, ".txt", "w"))
                #f.write(str(unichr(classes[0])))
                #f.close()
    print "Recognized lines:"
    text = printNonNones(lines)

    print "Text:"
    print text.rstrip('\n\n')
    output_fd = open(os.path.join(folder_path, 'output.txt'), "w")
    output_fd.write(text.rstrip('\n\n'))
    output_fd.close()
