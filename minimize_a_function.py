"""
Want to use library to minimize a function.
for example, minimize f(x) = (x-3)^2 + 4 or simply x**2
"""

from autograd.tensor import Tensor, add, sub, mul, tensor_sum

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)
sum_of_squares = mul(x, x).sum()  # f(x) = x^2 and is a 0-tensor

# want to minimize sum_of_squares
for i in range(100):
    sum_of_squares = tensor_sum(mul(x, x))  # f(x) = x^2 and is a 0-tensor
    sum_of_squares.backward()  # compute gradients

    delta_x = mul(x.grad, Tensor(0.1))  # learning rate of 0.1
    x = Tensor(x.data - delta_x.data, requires_grad=True)  # update x

    print(i, sum_of_squares)