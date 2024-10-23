import numpy as np
from tensorflow.keras.preprocessing import image

import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(train_images[0],cmap=plt.cm.binary)
plt.show()
print(train_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

def load_custom_image(img_path):
    # Загружаем изображение и приводим его к нужному формату (28x28)
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем ось для batch
    img_array = img_array / 255.0  # Нормализуем

    return img_array

# Использование 1
custom_img = load_custom_image('./img/1.png')
prediction = model.predict(custom_img)
print('Predicted class:', np.argmax(prediction))

# Использование 2
custom_img = load_custom_image('./img/2.png')
prediction = model.predict(custom_img)
print('Predicted class:', np.argmax(prediction))

# Использование 3
custom_img = load_custom_image('./img/3.png')
prediction = model.predict(custom_img)
print('Predicted class:', np.argmax(prediction))

# Использование 4
custom_img = load_custom_image('./img/4.png')
prediction = model.predict(custom_img)
print('Predicted class:', np.argmax(prediction))

# Использование 9
custom_img = load_custom_image('./img/9.png')
prediction = model.predict(custom_img)
print('Predicted class:', np.argmax(prediction))
