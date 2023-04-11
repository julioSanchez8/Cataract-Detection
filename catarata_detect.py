import os

import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

path='/Users/pc/Downloads/Datas/test/Sano/NL_110.png'#/test/Catara/161_right.jpg   /test/Sano/NL_111.png
img=image.load_img(path, target_size=(150, 150))

x=image.img_to_array(img)
image=np.expand_dims(x, axis=0)

model = load_model('Eyes_categorical.h5')

classes = model.predict (image)
print (classes)
plt.imshow(img)

# model = load_model('Eyes_categorical.h5')
# test_dir = os.path.join(path, 'test')
# test_catara_dir = os.path.join(test_dir, 'Catara')
# test_sano_dir = os.path.join(test_dir, 'Sano')
# test_predict = []
# for i in range(len(test_catara_dir)):
#     test_predict.append(test_catara_dir[i])
# for i in range(len(test_sano_dir)):
#     test_predict.append(test_sano_dir[i])
# test_expected = []
# for i in range(138):
#     test_expected.append([])
#     if i < 75:
#
#         test_expected[i].append(1)
#     else:
#         test_expected[i].append(0)
# test_expected.append(os.listdir(test_sano_dir))
#
#
# test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
#
# test_generator = test_datagen.flow_from_directory(test_dir,
#                                                   batch_size=5,
#                                                   class_mode='categorical',
#                                                   target_size=(150, 150))
#
# classes = model.predict(test_generator)
# # print(test_generator)##------------------------------
# print(classes)
# for i in range(len(classes)):
#     classes[i] = (int(classes[i]))
# print(test_expected)
# #%matplotlib inline
# import matplotlib.image as mpimg
#
# # Confution Matrix and Classification Report
# from sklearn.metrics import classification_report, confusion_matrix
# Y_pred = model.predict(test_generator)
# y_pred = []
# # print((Y_pred[0]))
# for i in Y_pred:
#     y_pred.append(int(i))
# print('Confusion Matrix')
# print(test_generator.filenames)
# print(y_pred)
# print(confusion_matrix(test_generator.classes, y_pred))
# print('Classification Report')
# target_names = ['Catara', 'Sano']
# y_test = test_generator.classes
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))
#
# from sklearn.metrics import ConfusionMatrixDisplay
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels =target_names )
# plt.show()
#

if classes>0:
    print("Sano")
else:
    print("Catarata")

plt.show()



print(model)