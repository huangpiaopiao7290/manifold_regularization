# import torch
#
# labels = torch.tensor([-1,-1,-1,-1,-1,0,3,-1,9,2,6,-1,-1,2,3])
# labeled_mask = (labels != -1)
# print(labeled_mask)
#
# if labeled_mask.any():
#     targets = labels[labeled_mask]
#     print(targets)
# else:
#     print("no label\n")
import os
directory = os.getcwd()
print(directory)


