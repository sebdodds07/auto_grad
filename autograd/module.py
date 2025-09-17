import inspect
from typing import Iterator
from autograd.tensor import Tensor
from autograd.parameter import Parameter

#collection of parameters knows how to zero_grad
class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
    
    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    