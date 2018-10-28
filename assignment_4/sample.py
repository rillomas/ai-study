from __future__ import print_function

import torch

if __name__ == "__main__":
    x = torch.rand(5, 3)
    print(x)
    print("CUDA available:", torch.cuda.is_available())
    device = 0
    print(torch.cuda.get_device_capability(device))
    print(torch.cuda.get_device_name(device))
