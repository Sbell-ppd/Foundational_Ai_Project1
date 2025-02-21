import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlp import MultilayerPerceptron, Layer,Relu, Linear,SquaredError
#downlaod and load the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
data = pd.read_csv(url, names=column_names, na_values='?',comment='\t', sep=' ', skipinitialspace=True)

# Drop rows with missing values

data = data.dropna()

#split the data into features and target

X = data.drop('MPG', axis=1).values
Y = data['MPG'].values.reshape(-1, 1)

#split the data into training,validation and testing sets

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Define the MLP architecture
layers = [
    Layer(fan_in=X_train.shape[1], fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=32, activation_function=Relu()),
    Layer(fan_in=32, fan_out=1, activation_function=Linear())
]
mlp = MultilayerPerceptron(layers=layers)

#Train the MLP model
training_losses, validation_losses = mlp.train(
    train_x = X_train,
    train_y = Y_train,
    val_x = X_val,
    val_y = Y_val,
    loss_func = SquaredError(),
    learning_rate = 1E-3,
    batch_size = 16,
    epochs = 100,
    rmsprop = True,
)
#  Print the training and validation losses at each epoch
for epoch in range(len(training_losses)):
    print(f"Epoch {epoch + 1}: Training Loss = {training_losses[epoch]}, Validation Loss = {validation_losses[epoch]}")

    # plot the loss curves
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

#Evaluate the model on the test set
test_predictions = mlp.forward(X_test)
test_loss = np.mean(SquaredError().loss(Y_test, test_predictions))
print(f'Total Test Loss: {test_loss}')

#Select 10 different samples form the testing set and report the predicted MPG against the true MPG
sample_indices = np.random.choice(len(X_test), 10, replace=False)
sample_X_test = X_test[sample_indices]
sample_Y_test = Y_test[sample_indices]
sample_predictions = mlp.forward(sample_X_test)

# create a DataFrame to display the results
results_df = pd.DataFrame({
    'True MPG': sample_Y_test.flatten(),
    'Predicted MPG': sample_predictions.flatten()
})
print(results_df)



