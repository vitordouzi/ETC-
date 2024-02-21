import torch
import torch.nn as nn

class SimMatrix(nn.Module):
    def __init__(self, eps=1e-8):
        super(SimMatrix, self).__init__()
        self.eps = eps

    def forward(self, a, b):
        a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
        a_norm = a / torch.clamp(a_n, min=self.eps)
        b_norm = b / torch.clamp(b_n, min=self.eps)
        return torch.einsum('bhid,bhjd->bhij', [a_norm, b_norm])
    
class InnerMatrix(nn.Module):
    def __init__(self):
        super(InnerMatrix, self).__init__()
    def forward(self, a, b):
        return torch.einsum('bhid,bhjd->bhij', [a, b]) / np.sqrt(a.shape[-1])
    
class DistMatrix(nn.Module):
    def __init__(self, eps=1e-8):
        super(DistMatrix, self).__init__()
        self.bias = nn.parameter.Parameter(torch.Tensor([1.]))
        self.eps = eps
    def forward(self, a, b):
        return ( self.bias + self.eps ) / ( torch.cdist(a, b) + self.bias + self.eps )