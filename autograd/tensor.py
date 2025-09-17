from typing import List, NamedTuple, Callable, Optional, Union

import numpy as np


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(self,
                data: arrayable,
                requires_grad: bool = False,
                depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))
    
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> "Tensor":
        return tensor_sum(self)
    
def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    containing the sum of all its elements.
    """

    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor (a scalar),
            so each element contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)

def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Idea: [1,2,3] + [4 + e,5,6] = [5 + e,7,9]
            # gradient will be appleid the same

            # Handle broadcasting
            # Sum out the added dimensions
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0) # sum over the added dimensions
                
            # Sum across dimensions that were broadcasted (but non-added dimsions)
            # (2, 3) + (1, 3) = (2, 3), grad (2, 3) but want(1, 3)

            for i, dim in enumerate(t1.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad


        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Handle broadcasting
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0) # sum over the added dimensions

            # Sum across dimensions that were broadcasted (but non-added dimsions)
            # (2, 3) + (1, 3) = (2, 3), grad (2, 3) but want(1, 3)

            for i, dim in enumerate(t2.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

            

        depends_on.append(Dependency(t2, grad_fn2))


    return Tensor(data,
                  requires_grad,
                  depends_on)

def mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = (t1 +eps) * t2 = t1 * t2 + (eps * t2)

    have dL/dy, want dL/dt1 and dL/dt2
    dL/dt1 = dL/dy * dy/dt1 = dL/dy * t2
    dL/dt2 = dL/dy * dy/dt2 = dL/dy * t1
    Note: need to handle broadcasting
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            # Handle broadcasting
            # Sum out the added dimensions
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0) # sum over the added dimensions
                
            # Sum across dimensions that were broadcasted (but non-added dimsions)
            # (2, 3) + (1, 3) = (2, 3), grad (2, 3) but want(1, 3)

            for i, dim in enumerate(t1.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad


        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            # Handle broadcasting
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0) # sum over the added dimensions

            # Sum across dimensions that were broadcasted (but non-added dimsions)
            # (2, 3) + (1, 3) = (2, 3), grad (2, 3) but want(1, 3)

            for i, dim in enumerate(t2.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

            

        depends_on.append(Dependency(t2, grad_fn2))


    return Tensor(data,
                  requires_grad,
                  depends_on)

def neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda grad: -grad)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)

def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return add(t1, neg(t2))
