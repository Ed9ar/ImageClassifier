from keras.datasets import cifar10
import keras.utils as utils
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
(_, _), (x_test, y_test) =cifar10.load_data()


x_test = x_test.astype('float32')/ 255.0
y_test = utils.to_categorical(y_test)

model = load_model("Image_Classifier.h5")

'''
results = model.evaluate(x=x_test,y=y_test)
print("Train loss", results[0])
print("Test accuracy", results[1])
'''

test_img = np.asarray([x_test[0]])
pred = model.predict(x=test_img)
max_index = np.argmax(pred[0])
print(labels[max_index])