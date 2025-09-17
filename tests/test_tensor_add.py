import unittest

from autograd.tensor import Tensor, add

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = Tensor([4,5,6], requires_grad=True)
        
        t3 = add(t1, t2)

        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]
        assert t2.grad.data.tolist() == [-1,-2,-3]

    def test_broadcast_add(self):
        """
        What it means to broadcast:
        [1,2,3] + [4] = [5,6,7]

        applying the same logic to gradients:
        [1,2,3] + [4 + e] = [5 + e, 6 + e, 7 + e]

        and we may not want this to happen to the gradients as now lead to more calcualtions after
        """

        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True) # (2,3) shape
        t2 = Tensor([7,8,9], requires_grad=True) # (3,) shape

        t3 = add(t1, t2) # (2,3) shape
        t3.backward(Tensor([[1,1,1], [1,1,1]])) # (2,3) shape

        assert t1.grad.data.tolist() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.tolist() == [2,2,2]  # because of broadcasting, each element is added twice

    def test_broadcast_add_2(self):
        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True) # (2,3) shape
        t2 = Tensor([[7, 8, 9]], requires_grad=True) # (1,3) shape

        t3 = add(t1, t2) # (2,3) shape
        t3.backward(Tensor([[1,1,1], [1,1,1]])) # (2,3) shape

        assert t1.grad.data.tolist() == [[1,1,1], [1,1,1]]
        assert t2.grad.data.tolist() == [[2,2,2]]  # because of broadcasting, each element is added twice