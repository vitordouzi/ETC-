from .similarities import *
from .embedding import ETCEmbedding
from .focalloss import FocalLoss
from ..utils import removeNaN
import torch.nn.functional as F

class ETCContext(nn.Module):
    def __init__(self, heads, sim_func, drop, leaky:float=-1.):
        super(ETCContext, self).__init__()
        self.H     = heads
        self.norm  = nn.LayerNorm(heads)
        self.drop_ = drop
        self.leaky = leaky
        if sim_func.lower() == "inner":
            self.dist_func = InnerMatrix()
        if sim_func.lower() == "dist":
            self.dist_func = DistMatrix()
        if sim_func.lower() == "sim":
            self.dist_func = SimMatrix()
    def getHidden(self, hidd):
        B, L, D = hidd.shape
        hidd = hidd.view( B, L, self.H, D//self.H )
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
        nco_weights = nco_weights + float(self.leaky > 0.)*F.leaky_relu(nco_weights, negative_slope=self.leaky)
        return nco_weights.transpose(-1,-2).view(B,H,L,L)
    def forward(self, K, Q, V, doc_sizes, pad_mask, bx_packed):
        K = self.getHidden(K) # K:[B,L,D] -> K:[B,H,L,D//H]
        Q = self.getHidden(Q) # Q:[B,L,D] -> Q:[B,H,L,D//H]
        V = self.getHidden(V) # V:[B,L,D] -> V:[B,H,L,D//H]
        pad_mask = pad_mask.unsqueeze(1).repeat([1, self.H, 1, 1]) # pm:[B, L, L] -> pm:[B, H, L, L]
        
        co_weights  = self.dist_func( K, Q ) # SIMILARITY(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> W:[B,H,L,L]
        co_weights  = self.lnormalize(co_weights)
        
        co_weights[pad_mask.logical_not()] = float('-inf')
        
        co_weights = torch.softmax(co_weights, dim=-1)                     # co:[B, H, L, L]
        co_weights = removeNaN(co_weights)
        
        weights = co_weights.sum(axis=-2).mean(axis=-2) # co_weights:[B,(H,L),L'] -> weights:[B,L']
        weights = weights / doc_sizes                   # weights:[B,L] / d_sizes:[B,1]
        weights[bx_packed] = float('-inf')
        weights = torch.softmax(weights, dim=-1)        # softmax(weights:[B,L]) -> weights:[B,L]
        weights = removeNaN(weights)                    # weights:[B,L] -> weights:[B,L,1]
        
        V = self.drop_(co_weights @ V)                  # W:[B,H,L,L] @ V:[B,H,L,D//H] -> V':[B,H,L,D//H]
        V = self.catHiddens(V)
        
        return V, weights.unsqueeze(-1)
class ETCModel(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, maxF: int=20, nheads: int=6,
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', drop: float = .5,
                att_model: str ='AA', sim_func='dist', use_tf:bool=True, use_df:bool=True, norep=2):
        super(ETCModel, self).__init__()
        self.D      = hiddens    # number of   (D)imensions
        self.C      = nclass     # number of   (C)lass
        self.H      = nheads     # number of   (H)eads on multhead
        self.V      = vocab_size # size of the (V)ocabulary
        self.S      = norep      # number of   (S)capes 
        self.drop_  = nn.Dropout(drop)
        self.fc     = nn.Sequential( nn.Linear(self.D, self.C+self.S), nn.Softmax(dim=-1) )
        self.cntx   = ETCContext(self.H, sim_func, self.drop_)
        self.emb_   = ETCEmbedding(self.V, self.D, maxF=maxF, use_tf=use_tf, use_df=use_df, drop=self.drop_, att_model=att_model)
        self.loss_f = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc[0].weight.data)
    
    def forward(self, doc_tids, TFs, DFs, labels=None):
        embedds = self.emb_(doc_tids, TFs, DFs) 
        V, weights = self.cntx(**embedds)
        
        V_lgts = self.fc(V)                    # FC(V':[B,L,D])-> V_lgs:[B,L,C+p]
        logits = V_lgts * weights              # V_lgs:[B,L,C+p] * weights:[B,L,1] -> logits:[B,L,C+p]
        logits = logits.sum(dim=-2)            # logits:[B,L,C+p] -> logits:[B,C+p]
        logits = logits[:,:self.C]             # V_lgts:logits:[B,L,C+b] -> logits:[B,C]
        logits = torch.softmax(logits, dim=-1) # softmax(logits:[B,C]) -> logits:[B,C]
         
        result_ = { 't_probs': V_lgts, 'logits': logits}
        if labels is not None:
            result_['loss'] = self.loss_f(logits, labels)
            
        return result_