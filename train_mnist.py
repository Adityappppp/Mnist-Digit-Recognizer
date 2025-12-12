import numpy as np
from keras import layers
from tensorflow import keras
from keras import layers

# 1. Load MNIST dataset (60,000 train, 10,000 test)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalize pixel values to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Flatten from (28,28) → (784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 4. Create model architecture: 784 → 128 → 64 → 32 → 10
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.1)

# 7. Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 8. Save weights to .npz (NumPy archive)
weights = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        w, b = layer.get_weights()
        weights[f"W{i+1}"] = w
        weights[f"b{i+1}"] = b
np.savez("weights_mnist.npz", **weights)
print("✅ Saved trained weights to weights_mnist.npz")
