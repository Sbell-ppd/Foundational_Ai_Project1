import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp import MultilayerPerceptron, Layer, Relu, Softmax, CrossEntropyLoss
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / 255.0
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
# Load the MNIST dataset
train_x = load_mnist_images('train-images-idx3-ubyte.gz')
train_y = load_mnist_labels('train-labels-idx1-ubyte.gz')
test_x = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# Split the training set into 80% training and 20% validation
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

#one hot encode the labels
train_y = np.eye(10)[train_y]
val_y = np.eye(10)[val_y]
test_y = np.eye(10)[test_y]

# Define the MLP architecture
layers = [
    Layer(fan_in=784, fan_out=128, activation_function=Relu()),
    Layer(fan_in=128, fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=10, activation_function=Softmax())
]

mlp = MultilayerPerceptron(layers=layers)

# Train the MLP model
training_losses, validation_losses = mlp.train(
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    loss_func=CrossEntropyLoss(),
    learning_rate=1E-3,
    batch_size=32,
    epochs=50,
    rmsprop=True,
)

# Print the training and validation losses at each epoch
for epoch in range(len(training_losses)):
    print(f"Epoch {epoch + 1}: Training Loss = {training_losses[epoch]}, Validation Loss = {validation_losses[epoch]}")

# Plot the loss curves
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Evaluate the model on the testing set
test_predictions = mlp.forward(test_x)
test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(test_y, axis=1))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Select one sample for each class (0-9) form the testing set and show these samples along with the predicted class
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    sample_index = np.where(np.argmax(test_y, axis = 1) == i)[0][0]
    sample_image = test_x[sample_index].reshape(28, 28)
    predicted_class = np.argmax(test_predictions[sample_index])
    axes[i // 5, i % 5].imshow(sample_image, cmap='gray')
    axes[i // 5, i % 5].set_title(f'Predicted: {predicted_class}')
    axes[i // 5, i % 5].axis('off')
plt.show()