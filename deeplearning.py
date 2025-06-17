import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(trainX, trainY), (testX, testY) = datasets.cifar10.load_data()
trainX, testX = trainX / 255.0, testX / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY))
