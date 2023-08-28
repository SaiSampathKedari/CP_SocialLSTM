import torch
import torch.nn as nn
import numpy as nn


# class SocialLSTMmodel(nn.Module):
    
#     def __init__(self, args, infer=False):
#         '''
#         Initializer function
#         params:
#         args: Training arguments
#         infer: Training or test time (true if test time)
#         '''
#         super(SocialLSTMmodel, self).__init__()

#         self.args = args
#         self.infer = infer
#         self.use_cuda = args.use_cuda

#         if infer:
#             # Test time
#             self.seq_length = 1
#         else:
#             # Training time
#             self.seq_length = args.seq_length