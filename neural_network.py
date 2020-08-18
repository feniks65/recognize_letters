import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

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
model.save_weights('model_11_receipts_weights.h5')
print "Labels from file:"
print y_train
print "Recognized labels:"
test_img = np.load('image_for_test.npy')
test_img = test_img / 255.0
print "test_img:"
print test_img
classes = model.predict_classes(test_img)
print "Rozpoznane klasy:"
for klass in classes:
    print str(unichr(klass))
#print model.predict(x_train)
#model.evaluate(x_test, y_test)
