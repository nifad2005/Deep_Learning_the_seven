from tensorflow import keras 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train.shape, y_train.shape)
X_train = X_train/255.0
X_test = X_test/255.0

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)


predictions = model.predict(X_test_flattened)
print(np.argmax(predictions[0]))
