import numpy as np
import csv

#Initialize weights
def initialize_neural_network(weight_sizes, total_layers):
    weights = []
    for i in range(total_layers - 1):
        weight_matrix = np.random.randn(weight_sizes[i], weight_sizes[i + 1]) * 0.01#weights b/w 2 layers
        weights.append(weight_matrix)#adds this matrix into weights
    return weights

#Initialize biases with zero values
def initialize_bias(weight_sizes, total_layers):
    biases = []
    for i in range(1, total_layers):
        bias = np.zeros((weight_sizes[i], 1))#bias b/w 2 layers
        biases.append(bias)#adds this matrix into biases
    return biases

#Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

#Softmax function
def softmax(outputs):
    exp_outputs = np.exp(outputs - np.max(outputs))  #positivity for every output
    return exp_outputs / exp_outputs.sum(axis=0, keepdims=True)#keepdims makes sure dimension is not changed

#Forward_propagation
def forward_propagation(inputs, neurons, weights, biases, total_layers, weight_sizes):
    neurons[0] = inputs
    for i in range(1, total_layers):
        z = np.dot(weights[i - 1].T, neurons[i - 1]) + biases[i - 1]#.T is transpose
        if i<total_layers-1:
            neurons[i] = sigmoid(z)#activation function for hidden layers
        else:
            neurons[i] = softmax(z)#activation function for output layer(most probable one)
    return neurons

#Backpropagation
def backpropagation(neurons, weights, biases, total_layers, weight_sizes, target):
    deltas = [None] * (total_layers - 1) #gradients

    for i in range(total_layers - 2, -1, -1): #reverse for loop i.e. total_layers-2 to 0
        if i == total_layers - 2:
            deltas[i] = neurons[i + 1] - target #loss
        else:
            deltas[i] = np.dot(weights[i + 1], deltas[i + 1]) * sigmoid_derivative(neurons[i + 1])#scalar product

    grad_weights = [None] * (total_layers - 1)
    grad_biases = [None] * (total_layers - 1)
    for i in range(total_layers - 1):
        grad_weights[i] = np.dot(neurons[i], deltas[i].T) #finding gradient for weights from activation fns
        grad_biases[i] = deltas[i] #finding gradient for biases from activation fns
    
    return grad_weights, grad_biases

#Updating weights and biases
def update_parameters(weights, biases, grad_weights, grad_biases, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grad_weights[i] #updating weights
        biases[i] -= learning_rate * grad_biases[i] #updating biases
    return weights, biases

#Prediction function
def prediction(input_data, neurons, weights, biases, total_layers, weight_sizes):
    neurons = forward_propagation(input_data, neurons, weights, biases, total_layers, weight_sizes)
    return np.argmax(neurons[-1])

#Reading CSV files
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data, dtype=int) #dtype converts this string into a int:) 

#adding predicted values into CSV file
def save_predictions(predictions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for prediction in predictions:
            writer.writerow([prediction])

#Training the neural network
def train_neural_network(training_data, training_labels, test_data, test_labels, weight_sizes, total_layers, epochs, learning_rate):
    weights = initialize_neural_network(weight_sizes, total_layers)
    biases = initialize_bias(weight_sizes, total_layers)

    neurons = [np.zeros((weight_sizes[i], 1)) for i in range(total_layers)]

    for epoch in range(epochs):
        for i in range(len(training_data)):
            input_data = training_data[i].reshape(-1, 1)  # Reshape to column vector
            target = np.zeros((weight_sizes[-1], 1))
            target[int(training_labels[i].item())] = 1  # Ensure labels are scalar integers

            # Forward pass
            neurons = forward_propagation(input_data, neurons, weights, biases, total_layers, weight_sizes)

            # Backpropagation
            grad_weights, grad_biases = backpropagation(neurons, weights, biases, total_layers, weight_sizes, target)

            # Update weights and biases
            weights, biases = update_parameters(weights, biases, grad_weights, grad_biases, learning_rate)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed")

    # After training, evaluate the model and save predictions
    predictions = []
    for i in range(len(test_data)):
        input_data = test_data[i].reshape(-1, 1)  # Reshape to column vector
        prediction_result = prediction(input_data, neurons, weights, biases, total_layers, weight_sizes)
        predictions.append(prediction_result)
    
    # Save the predictions to a CSV file
    save_predictions(predictions, 'Digits_Recognition/predictions.csv')

    # Calculate accuracy
    correct_predictions = sum([1 for i in range(len(test_data)) if predictions[i] == int(test_labels[i].item())])
    accuracy = correct_predictions / len(test_data) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    return weights, biases

if __name__ == '__main__':#For code reusability(Generally used for test code or example/ Cannot be imported)

    weight_sizes = [784, 10, 10] 
    total_layers = len(weight_sizes)
    epochs = 100
    learning_rate = 0.01

    # Dataset
    training_data = read_csv('Digits_Recognition/training_data.csv')  #Shape: (60000, 784)
    training_labels = read_csv('Digits_Recognition/training_label.csv')  #Shape: (60000,1)
    test_data = read_csv('Digits_Recognition/test_data.csv')  #Shape: (10000, 784)
    test_labels = read_csv('Digits_Recognition/test_label.csv')  #Shape: (10000,1)


    # Train the neural network
    train_neural_network(training_data, training_labels, test_data, test_labels, weight_sizes, total_layers, epochs, learning_rate)
