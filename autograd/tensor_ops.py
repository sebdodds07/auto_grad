from typing import List
import numpy as np
from autograd.tensor import Tensor, Dependency


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

def _add(t1: Tensor, t2: Tensor) -> Tensor:
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

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
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

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda grad: -grad)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2
