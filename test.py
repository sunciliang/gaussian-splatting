import torch

A = torch.tensor([True,False,False,True,False],dtype=torch.bool)
B = torch.tensor([[1,2,3],
                 [4,5,6],
                 [7,8,9],
                 [10,11,12],
                 [13,14,15]],dtype=torch.float32)
print(B)
C = B[A] *3
B[A] *= 3
print(C)