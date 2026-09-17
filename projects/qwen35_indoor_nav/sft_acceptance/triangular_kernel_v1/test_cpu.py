import importlib.util
from pathlib import Path
import unittest
import torch

s=importlib.util.spec_from_file_location('triangular',Path(__file__).with_name('triangular.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(2)

class Tests(unittest.TestCase):
    def compare(self,n,dtype):
        torch.manual_seed(1109)
        leaf=(torch.randn(2,3,n,n,dtype=dtype)*.05).requires_grad_()
        a=leaf.tril(-1)
        y=m.reference(a);z=m.triangular(a)
        tol=1e-10 if dtype==torch.float64 else 1e-5
        torch.testing.assert_close(y,z,rtol=tol,atol=tol)
        weight=torch.randn_like(y)
        g=torch.autograd.grad((y*weight).sum(),leaf,retain_graph=True)[0]
        h=torch.autograd.grad((z*weight).sum(),leaf)[0]
        torch.testing.assert_close(g,h,rtol=tol*10,atol=tol*10)
    def test_n1(self):self.compare(1,torch.float64)
    def test_n8(self):self.compare(8,torch.float64)
    def test_n64_float64(self):self.compare(64,torch.float64)
    def test_n64_float32(self):self.compare(64,torch.float32)
    def test_zero(self):
        a=torch.zeros(2,64,64)
        torch.testing.assert_close(m.triangular(a),torch.eye(64).expand_as(a))
    def test_shape_guard(self):
        with self.assertRaises(ValueError):m.triangular(torch.ones(2,3))

if __name__=='__main__':unittest.main()
