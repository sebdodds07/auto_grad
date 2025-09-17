import numpy as np
from typing import List
from autograd import Tensor, Parameter, Module
from autograd.optim import SGD
from autograd.function import tanh

def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]  # FizzBuzz
    elif x % 5 == 0:
        return [0, 0, 1, 0]  # Buzz
    elif x % 3 == 0:
        return [0, 1, 0, 0]  # Fizz
    else:
        return [1, 0, 0, 0]  # number
    
x_train = Tensor([binary_encode(i) for i in range(101, 1024)])
y_train = Tensor([fizz_buzz_encode(i) for i in range(101, 1024)])

class FizzBuzzModel(Module):
    def __init__(self, num_hidden: int = 50) -> None:
        self.w1 = Parameter(10, num_hidden)  # input layer to hidden layer
        self.b1 = Parameter(num_hidden,)      # bias for hidden layer
        self.w2 = Parameter(num_hidden, 4)    # hidden layer to output layer
        self.b2 = Parameter(4,)        # bias for output layer

    def predict(self, inputs: Tensor) -> Tensor:
        # inputs will be (batch_size, 10)
        x1 = inputs @ self.w1 + self.b1  # (batch_size, num_hidden)
        x2 = tanh(x1)               # (batch_size, num_hidden)
        x3 = x2 @ self.w2 + self.b2      # (batch_size, 4)
        
        return x3
    
model = FizzBuzzModel()
optimizer = SGD(lr=0.001)
batch_size = 32

starts = np.arange(0, x_train.data.shape[0], batch_size)

for epoch in range(5000):
    epoch_loss = 0.0

    np.random.shuffle(starts)

    for start in starts:
        end = start + batch_size

        model.zero_grad()

        inputs = x_train[start:end]
    
        predicted = model.predict(inputs)  # (batch_size, 4)
        actual = y_train[start:end]        # (batch_size, 4)
        error = predicted - actual         # (batch_size, 4)
        loss = (error*error).sum()         # MSE

        loss.backward()

        epoch_loss += loss.data

        optimizer.step(model)

    print(epoch, epoch_loss)

num_correct = 0
for x in range(1, 101):
    inputs = Tensor([binary_encode(x)])
    predicted = model.predict(inputs)[0]
    predicted_idx = np.argmax(predicted.data)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    
    if predicted_idx == actual_idx:
        num_correct += 1
    
    print(x, labels[predicted_idx], labels[actual_idx])

print("Final accuracy:", num_correct, "/ 100")