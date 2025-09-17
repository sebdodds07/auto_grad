import unittest
from autograd.tensor import Tensor, _matmul
import numpy as np

class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self):
        t1 = Tensor([[1,2], [3,4], [5,6]], requires_grad=True) 
        t2 = Tensor([[10], [20]], requires_grad=True) 

        t3 = t1 @ t2 

        assert t3.data.tolist() == [[50], [110], [170]]

        grad = Tensor([[-1], [-2], [-3]])
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data, grad.data @ t2.data.T)
        #np.testing.assert_array_equal(t1.grad.data, )

        assert t1.grad.data.tolist() == [[-10, -20], [-20, -40], [-30, -60]]
        assert t2.grad.data.tolist() == [[-22], [-28]]



       