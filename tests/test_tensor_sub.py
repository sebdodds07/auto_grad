import unittest

from autograd.tensor import Tensor

class TestTensorSub(unittest.TestCase):
    def test_simple_sub(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = Tensor([4,5,6], requires_grad=True)
        
        t3 = t1 - t2

        assert t3.data.tolist() == [-3,-3,-3]

        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]
        assert t2.grad.data.tolist() == [+1,+2,+3]

        t1 -= 0.1
        assert t1.grad is None  # because did in-place operation so invalidate grad
        assert t1.data.tolist() == [0.9, 1.9, 2.9]
    
    def test_broadcast_sub(self):
        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True) # (2,3) shape
        t2 = Tensor([7,8,9], requires_grad=True) # (3,) shape

        t3 = t1 - t2
        
        assert t3.data.tolist() == [[-6,-6,-6], [-3,-3,-3]] # (2,3) shape

        t3.backward(Tensor([[1,1,1], [1,1,1]])) # (2,3) shape

        assert t1.grad.data.tolist() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.tolist() == [-2, -2, -2]  # because of broadcasting, each element is subtracted twice
    
    def test_broadcast_sub_2(self):
        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True) # (2,3) shape
        t2 = Tensor([[7, 8, 9]], requires_grad=True) # (1,3) shape

        t3 = t1 - t2 # (2,3) shape

        assert t3.data.tolist() == [[-6,-6,-6], [-3,-3,-3]] # (2,3) shape

        t3.backward(Tensor([[1,1,1], [1,1,1]])) # (2,3) shape

        assert t1.grad.data.tolist() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.tolist() == [[-2,-2,-2]]  # because of broadcasting, each element is subtracted twice
        