# Perceptron from Scratch.

This project is a simple implementation of a neural network from scratch. The neural network is capable of performing binary classification tasks. It uses the concept of artificial neurons and backpropagation to learn from input data and make predictions.

## Getting Started

To get started with this project, you can follow the instructions below.

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy

### Installation

1. Clone the repository:

   ```shell
   git clone <repository_url>
   ```

2. Navigate to the project directory:

   ```shell
   cd neural-network-from-scratch
   ```

3. Install the required dependencies:

   ```shell
   pip install numpy
   ```

### Usage

To use the neural network, follow these steps:

1. Import the necessary modules:

   ```python
   import math
   import numpy as np
   ```

2. Copy the code for the `Connection`, `Neuron`, and `Network` classes into your Python script or notebook.

3. Create an instance of the `Network` class with the desired dimensions for the neural network:

   ```python
   network = Network([input_size, hidden_size, output_size])
   ```

4. Set the input values for the network:

   ```python
   network.setInput(inputs)
   ```

5. Activate the network:

   ```python
   network.activate()
   ```

6. Perform backpropagation by providing the target output values:

   ```python
   network.backpropagate(targets)
   ```

7. Access the predicted output values:

   ```python
   predictions = network.getPredictions()
   ```

8. Continue training the network by iterating over the data and adjusting the weights until the desired loss is achieved.

## Example

Here's an example of how to use the neural network:

```python
# Import the necessary modules
import math
import numpy as np

# Copy the code for the Connection, Neuron, and Network classes

# Create an instance of the Network class
network = Network([2, 1])

# Set the input values
inputs = [0, 1]
network.setInput(inputs)

# Activate the network
network.activate()

# Perform backpropagation
targets = [1]
network.backpropagate(targets)

# Get the predicted output values
predictions = network.getPredictions()

# Print the predictions
print(predictions)
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
