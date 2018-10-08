import numpy as np
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

if __name__ == "__main__":
    assert AND(0,0) is 0
    assert AND(1,0) is 0
    assert AND(0,1) is 0
    assert AND(1,1) is 1
    assert OR(0,0) is 0
    assert OR(1,0) is 1
    assert OR(0,1) is 1
    assert OR(1,1) is 1
    assert NOR(0,0) is 1
    assert NOR(1,0) is 0
    assert NOR(0,1) is 0
    assert NOR(1,1) is 0
    assert XOR(0,0) is 0
    assert XOR(1,0) is 1
    assert XOR(0,1) is 1
    assert XOR(1,1) is 0
    assert XOR2(0,0) is 0
    assert XOR2(1,0) is 1
    assert XOR2(0,1) is 1
    assert XOR2(1,1) is 0

