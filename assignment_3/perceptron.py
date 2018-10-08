import numpy as np
import pdb

def NOR(l, r):
    x = np.array([l, r])
    w = np.array([0.5, 0.5])
    b = -0.3
    s = np.sum(w*x) + b
    return 0 if s >= 0 else 1

def XOR(l, r):
    #ll = NOR(l, r)
    return 0

if __name__ == "__main__":
    assert NOR(0,0) is 1
    assert NOR(1,0) is 0
    assert NOR(0,1) is 0
    assert NOR(1,1) is 0
    assert XOR(0,0) is 0
    assert XOR(1,0) is 1
    assert XOR(0,1) is 1
    assert XOR(1,1) is 0

