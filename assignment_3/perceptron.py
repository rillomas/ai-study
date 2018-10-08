import numpy as np
import unittest
import pdb

def AND(l, r):
    x = np.array([l, r])
    w = np.array([0.5, 0.5])
    b = -0.7
    s = np.sum(w*x) + b
    return 0 if s <= 0 else 1

def NAND(l, r):
    x = np.array([l, r])
    w = np.array([-0.5, -0.5])
    b = 0.7
    s = np.sum(w*x) + b
    return 0 if s <= 0 else 1

def OR(l, r):
    x = np.array([l, r])
    w = np.array([0.5, 0.5])
    b = -0.2
    s = np.sum(w*x) + b
    return 0 if s <= 0 else 1

def NOR(l, r):
    x = np.array([l, r])
    w = np.array([-0.5, -0.5])
    b = 0.3
    s = np.sum(w*x) + b
    return 0 if s <= 0 else 1

def XOR(l, r):
    ll = OR(l, r)
    rr = NAND(l, r)
    return AND(ll, rr)

def XOR2(l, r):
    ll = NOR(l, r)
    rr = AND(l, r)
    return NOR(ll, rr)

def HADD(l, r):
    c = AND(l, r)
    s = XOR(l, r)
    return c, s

class TestPerceptron(unittest.TestCase):
    def test_and(self):
        self.assertEqual(AND(0,0), 0)
        self.assertEqual(AND(1,0), 0)
        self.assertEqual(AND(0,1), 0)
        self.assertEqual(AND(1,1), 1)

    def test_or(self):
        self.assertEqual(OR(0,0), 0)
        self.assertEqual(OR(1,0), 1)
        self.assertEqual(OR(0,1), 1)
        self.assertEqual(OR(1,1), 1)

    def test_nor(self):
        self.assertEqual(NOR(0,0), 1)
        self.assertEqual(NOR(1,0), 0)
        self.assertEqual(NOR(0,1), 0)
        self.assertEqual(NOR(1,1), 0)

    def test_xor(self):
        self.assertEqual(XOR(0,0), 0)
        self.assertEqual(XOR(1,0), 1)
        self.assertEqual(XOR(0,1), 1)
        self.assertEqual(XOR(1,1), 0)

    def test_xor2(self):
        self.assertEqual(XOR2(0,0), 0)
        self.assertEqual(XOR2(1,0), 1)
        self.assertEqual(XOR2(0,1), 1)
        self.assertEqual(XOR2(1,1), 0)

    def test_hadd(self):
        self.assertEqual(HADD(0,0), (0, 0))
        self.assertEqual(HADD(1,0), (0, 1))
        self.assertEqual(HADD(0,1), (0, 1))
        self.assertEqual(HADD(1,1), (1, 0))

if __name__ == "__main__":
    unittest.main()
