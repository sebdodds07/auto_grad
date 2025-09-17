import numpy as np
from autograd import Tensor, Parameter, Module
from autograd.optim import SGD

x_data = Tensor(np.random.randn(100,3))
coef = Tensor(np.array([-1, +3, -2]))
y_data = x_data @ coef + 5 # remove noise for simplicity and linearity  

class Model(Module):
    def __init__(self):
        # coefficients
        self.w = Parameter(3) # tensor of shape (3,), requires_grad=True, initialized randomly
        self.b = Parameter()  # bias term, shape=()
    
    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b  # (N,3) @ (3,) + () = (N,)

optimizer = SGD(lr=0.001)
batch_size = 32
model = Model()

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        model.zero_grad()

        inputs = x_data[start:end]
    
        predicted = inputs @ model.w + model.b
        actual = y_data[start:end]  # (100,3) @ (3,) + () = (100,)
        error = predicted - actual  # (100,) - (100,) = (100,)
        loss = (error*error).sum()  # MSE

        loss.backward()

        epoch_loss += loss.data

        optimizer.step(model)
        # Alternatively, without optimizer:
        #model.w -= model.w.grad * learning_rate
        #model.b -= model.b.grad * learning_rate

    print(epoch, epoch_loss)

