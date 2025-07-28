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

model.fit(X_train_flattened, y_train, epochs=20)

test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)


predictions = model.predict(X_test_flattened)
for i in range(5):
    predicted_label = np.argmax(predictions[i]) # সর্বোচ্চ সম্ভাব্যতা সহ লেবেল
    true_label = y_test[i]
    print(f"ছবি {i+1}: প্রেডিক্টেড লেবেল: {predicted_label}, আসল লেবেল: {true_label}")

    # ছবিটি প্লট করে দেখা (ঐচ্ছিক)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.show()
