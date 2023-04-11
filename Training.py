import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json
import cv2

print(tf.__version__)


def prepocessing_fun(filename):
    input(filename)
    original = cv2.imread(filename)
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GRISES", gris)
    # cv2.waitKey(0)
    return gris


# Definición de los directorios del dataset
base_dir = '\\Users\\pc\\Downloads\\Datas'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Directorio con las imagenes de training
train_catara_dir = os.path.join(train_dir, 'Catara')
train_sano_dir = os.path.join(train_dir, 'Sano')

# Directorio con las imagenes de validation
validation_catara_dir = os.path.join(validation_dir, 'Catara')
validation_sano_dir = os.path.join(validation_dir, 'Sano')

# Directorio con las imagenes de test
test_catara_dir = os.path.join(test_dir, 'Catara')
test_sano_dir = os.path.join(test_dir, 'Sano')

# print(os.path.normcase(train_catara_dir))
# print(validation_catara_dir)
# print(test_catara_dir)

# print(os.listdir("../Proyecto/Datas"))

# image = Image.open(base_dir + '/train/Sano/NL_112.png')
# imgplot = plt.imshow(image)
# plt.show()

# Confección de la lista de imagenes
# train_catara_fnames = os.listdir(train_catara_dir)
# print(train_catara_fnames[:5])
#
# train_sano_fnames = os.listdir(train_sano_dir)
# print(train_sano_fnames[:5])
#
# validation_catara_fnames = os.listdir(validation_catara_dir)
# print(validation_catara_fnames[:5])
#
# validation_sano_fnames = os.listdir(validation_sano_dir)
# print(validation_sano_fnames[:5])
#
# test_catara_fnames = os.listdir(test_catara_dir)
# print(test_catara_fnames[:5])
#
# test_sano_fnames = os.listdir(test_sano_dir)
# print(test_sano_fnames[:5])

print('total training catarata images :', len(os.listdir(train_catara_dir)))
print('total training sano images :', len(os.listdir(train_sano_dir)))

print('total validation Cataratas images :', len(os.listdir(validation_catara_dir)))
print('total validation Ojo sano images :', len(os.listdir(validation_sano_dir)))

print('total test cataratas images :', len(os.listdir(test_catara_dir)))
print('total test ojo sano images :', len(os.listdir(test_sano_dir)))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),  # optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
# print("HOLA")
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)



train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=5,
                                                    # color_mode="grayscale",
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    )

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=5,
                                                              # color_mode="grayscale",
                                                              class_mode='binary',
                                                              target_size=(150, 150),
                                                              )

test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                  batch_size=5,
                                                  # color_mode="grayscale",
                                                  class_mode='binary',
                                                  target_size=(150, 150),
                                                  )

batch_size = 5
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

print(steps_per_epoch)
print(validation_steps)
# input(type(train_generator[0]))
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=180,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    verbose=2)

# Guardar el modelo entrenado
model.save('Eyes_categorical.h5')

print(train_generator.class_indices)

a_file = open("Eyes_indices.json", "w")
a_file = json.dump(train_generator.class_indices, a_file)

# print(model.class_indices)
history_dict = history.history
print(history_dict.keys())

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1, 1)  # obtener número de epochs del eje X

plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.show()

plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.show()

test_steps = test_generator.n // batch_size
print(test_steps)
test_lost, test_acc = model.evaluate(test_generator, steps=test_steps)
print("Test Accuracy:", test_acc)
