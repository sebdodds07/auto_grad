"""
Want to use library to minimize a function.
for example, minimize f(x) = (x-3)^2 + 4 or simply x**2
"""

from autograd.tensor import Tensor #, add, sub, mul, tensor_sum

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)
sum_of_squares = (x*x).sum()  # f(x) = x^2 and is a 0-tensor

# want to minimize sum_of_squares
for i in range(100):
    x.zero_grad()  # reset gradients to zero
    
    sum_of_squares = (x*x).sum()  # f(x) = x^2 and is a 0-tensor
    sum_of_squares.backward()  # compute gradients

    delta_x = 0.1 * x.grad # learning rate of 0.1
    x -= delta_x # now doesn't have a zero grad so zero it out above in next loop
    
    # = Tensor(x.data - delta_x.data, requires_grad=True)  # update x

    print(i, sum_of_squares)