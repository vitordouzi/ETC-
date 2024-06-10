import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def removeNaN(data):
    zeros   = torch.zeros_like(data)
    isnan   = torch.isnan(data)
    return torch.where(isnan, zeros, data)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, labels):
        assert logits.size(0) == len(labels), "Batch with diff sizes"
        assert logits.size(1) > labels.max(), "labels with wrong label"
        
        index1 = labels.long()
        index0 = (torch.ones_like(labels).cumsum(0) - 1).long()
        values = torch.ones_like(labels).float()
        
        target = torch.zeros_like(logits).float()
        target = target.index_put_((index0, index1), values)
        
        return sigmoid_focal_loss(logits, target, alpha=self.alpha,
                                     gamma=self.gamma, reduction=self.reduction)
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
class ETCAttention(nn.Module):
    def __init__(self, ndim, nheads, nclass, p=.3):
        super(ETCAttention, self).__init__()
        self.H = nheads
        self.D = ndim
        self.C = nclass
        self.norm      = nn.LayerNorm(self.H)
        self.drop      = nn.Dropout(p=p)
        self.lin_out   = nn.Sequential( nn.Linear(self.D, self.C), nn.Softmax(dim=-1) )
        self.sim_fn    = SimMatrix()
        self.init_params()
    def init_params(self):
        torch.nn.init.kaiming_normal_(self.lin_out[0].weight)
    def getHidden(self, hidd):
        B, L, D = hidd.shape
        hidd = hidd.reshape( B, L, self.H, D//self.H )
        hidd = hidd.transpose(1,2)
        return hidd
    def catHiddens(self, hidd):
        B, H, L, d = hidd.shape
        hidd = hidd.transpose(1,2)
        hidd = hidd.reshape(B, L, H*d)
        return hidd
    
    def lnormalize(self, co_weights):
        B,H,L,_     = co_weights.shape
        nco_weights = co_weights.view(B,H,L*L).transpose(-1,-2)
        nco_weights = self.norm(nco_weights)
        return nco_weights.transpose(-1,-2).view(B,H,L,L)
    
    def forward(self, hiddens, mask2d):
        B,L,D = hiddens.shape
        hiddens = hiddens.reshape(B,L,3,D//3)
        
        Q = self.getHidden(hiddens[:,:,0,:])  # Q:[B,L,D] -> [B,H,L,D//H]
        K = self.getHidden(hiddens[:,:,1,:])  # K:[B,L,D] -> [B,H,L,D//H]
        
        att = self.sim_fn(K, Q)               # Q:[B,H,L,D//H] K:[B,H,L,D//H] -> [B,H,L,L]
        att = self.lnormalize(att)            # att:[B,H,L,L] -> [B,H,L,L]
        #att[mask2d] = 0                       # att:[B,H,L,L] mask:[B,H,L,L] -> [B,H,L,L] (Avg on z-norm)
        att = torch.softmax(att, dim=-1)
        att = removeNaN(att)
        
        V = self.getHidden(hiddens[:,:,2,:])  # V:[B,L,D] -> [B,H,L,D//H]
        V = self.catHiddens(att @ V)          # att:[B,H,L,L] V:[B,H,L,D//H] -> [B,L,D]
        
        return self.lin_out(self.drop(V)), att

class ETCModel(nn.Module):
    def __init__(self, ndim_in, ndim_out, nheads, vocab_size, nclass, p=.3):
        super(ETCModel, self).__init__()
        self.H       = nheads
        
        self.lin_inH = nn.Sequential(nn.Linear(ndim_in, 3*ndim_out), nn.LeakyReLU())
        self.attH    = ETCAttention(ndim_out, nheads, nclass, p=p)
        
        self.lin_inT = nn.Sequential(nn.Embedding(vocab_size, ndim_out), nn.LeakyReLU(),
                                    nn.Linear(ndim_out, 3*ndim_out), nn.LeakyReLU())
        self.attT    = ETCAttention(ndim_out, nheads, nclass, p=p)
        
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = FocalLoss(reduction='sum')
        self.init_params()
    def init_params(self):
        torch.nn.init.kaiming_normal_(self.lin_inH[0].weight)
        torch.nn.init.kaiming_normal_(self.lin_inT[0].weight)
        torch.nn.init.kaiming_normal_(self.lin_inT[2].weight)
        
    def getWeights(self, att, masks):
        W = att.mean(dim=1).sum(dim=1)
        W = masks * W
        return torch.softmax(W, dim=-1).unsqueeze(-1)
        
    
    def forward(self, input_ids, wv, labels=None):
        masks  = input_ids != 0  # [B,L]
        mask2d = masks.unsqueeze(1).logical_and(masks.unsqueeze(2)).unsqueeze(1) # B,1,L,L
        mask2d = mask2d.repeat(1,self.H,1,1).logical_not()
        
        probs_t, att_t = self.attT(self.lin_inT(input_ids), mask2d)
        probs_t = (self.getWeights(att_t, masks) * probs_t).sum(dim=1)
        
        probs_h, att_h = self.attH(self.lin_inH(wv), mask2d)
        probs_h = (self.getWeights(att_h, masks) * probs_h).sum(dim=1)
        
        probs = (probs_t+probs_h)/2.
        result = { 'logits': probs  }
        if labels is not None:
            loss_h = self.loss_fn(probs_h, labels)
            loss_t = self.loss_fn(probs_t, labels)
            result['loss'] = (loss_h + loss_t)/2.
        return result