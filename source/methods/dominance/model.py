
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import copy
from .etcembedding import ETCEmbedding
from .utils import seed_everything
from .tokenizer import DominanceTokenizer

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

class ETCContext(nn.Module):
    def __init__(self, heads, sim_func, drop, leaky:float=torch.e):
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
        #nco_weights = nco_weights + float(self.leaky > 0.)*F.leaky_relu(nco_weights, negative_slope=self.leaky)######################################################################
        return nco_weights.transpose(-1,-2).view(B,H,L,L)
    def forward(self, K, Q, V, doc_sizes, pad_mask, bx_packed):
        K = self.getHidden(K) # K:[B,L,D] -> K:[B,H,L,D//H]
        Q = self.getHidden(Q) # Q:[B,L,D] -> Q:[B,H,L,D//H]c
        V = self.getHidden(V) # V:[B,L,D] -> V:[B,H,L,D//H]
        pad_mask = pad_mask.unsqueeze(1).repeat([1, self.H, 1, 1]) # pm:[B, L, L] -> pm:[B, H, L, L]
        
        co_weights  = self.dist_func( K, Q ) # SIMILARITY(Q:[B,H,L,D//H], Q:[B,H,L,D//H]) -> W:[B,H,L,L]
        co_weights  = self.lnormalize(co_weights)
        
        co_weights[pad_mask.logical_not()] = float('-inf') #float('-inf')
        
        co_weights = torch.softmax(co_weights, dim=-1)                     # co:[B, H, L, L]
        co_weights = removeNaN(co_weights)
        
        weights = co_weights.sum(axis=-2).sum(axis=-2)  # co_weights:[B,(H,L),L'] -> weights:[B,L']
        weights = weights / doc_sizes                   # weights:[B,L] / d_sizes:[B,1]
        weights[bx_packed] = float('-inf')
        weights = torch.softmax(weights, dim=-1)        # softmax(weights:[B,L]) -> weights:[B,L]
        weights = removeNaN(weights)                    # weights:[B,L] -> weights:[B,L,1]
        
        V = self.drop_(co_weights @ V)                  # W:[B,H,L,L] @ V:[B,H,L,D//H] -> V':[B,H,L,D//H]
        V = self.catHiddens(V)
        
        return V, weights.unsqueeze(-1)

class ETCModel(nn.Module):
    def __init__(self, vocab_size: int, hiddens: int, nclass: int, nheads: int=6, left: bool=True, form: str='left',
                 alpha: float = 0.25, gamma: float = 3., reduction: str = 'sum', drop: float = .5,
                att_model: str ='AA', mode:str='sum', sim_func='dist', temperature=1):
        super(ETCModel, self).__init__()
        self.D      = hiddens    # number of   (D)imensions
        self.C      = nclass     # number of   (C)lass
        self.H      = nheads     # number of   (H)eads on multhead
        self.V      = vocab_size # size of the (V)ocabulary
        self.alpha  = alpha
        self.form   = form
        self.drop_  = nn.Dropout(drop)
        self.fc     = nn.Linear(self.D, self.C)
        self.cntx   = ETCContext(self.H, sim_func, self.drop_)
        self.emb_   = ETCEmbedding(self.V, self.D, drop=self.drop_, att_model=att_model)
        #self.loss_f = FocalLoss(gamma=gamma, alpha=.25, reduction=reduction)
        self.loss_f = nn.CrossEntropyLoss()
        self.sig    = nn.Sigmoid()
        self.temperature = temperature
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight.data)
    
    def get_inverse_fc(self, dom):
        W = self.fc.weight
        b = self.fc.bias
        
        if self.form == 'left':
            W2 = torch.linalg.inv(W.T @ W)
            W  = W2 @ W.T   # W^-1  = W^T @ W^-2
        elif self.form == 'right':
            W2 = torch.linalg.inv(W @ W.T)
            W  = W.T @ W2   # W^-1  = W^T @ W^-2
        elif self.form is None:
            W  = W.T
        return (dom-b)@W.T
        #return (W@((dom-b).transpose(-1,-2))).transpose(-1,-2)
    
    def forward(self, input_ids, dom, labels=None):
        dom = torch.pow(torch.softmax(dom, dim=-1), self.temperature)
        dom = self.get_inverse_fc(dom)
        
        embedds = self.emb_(input_ids)
        
        V, weights = self.cntx(**embedds)
        V      = self.alpha*V + (1.-self.alpha)*dom
        V_lgts = self.fc(V)                  # FC(V':[B,L,D])-> V_lgs:[B,L,C+p]
        logits = V_lgts * weights              # V_lgs:[B,L,C+p] * weights:[B,L,1] -> logits:[B,L,C+p]
        logits = logits.sum(dim=-2)            # logits:[B,L,C+p] -> logits:[B,C+p]
        logits = torch.softmax(logits, dim=-1) # softmax(logits:[B,C]) -> logits:[B,C]
         
        result_ = { 't_probs': V_lgts, 'logits': logits}
        if labels is not None:
            result_['loss'] = self.loss_f(logits, labels)
            
        return result_


class ETCClassifier():
    def __init__(self, tknz, model, nepochs:int=50,
                max_drop:float=.75, batch_size:int=16, min_f1=.97, seed=42, 
                weight_decay:float = 5e-3, lr:float = 5e-3, update_drop:bool=True, device='cuda'):
        super(ETCClassifier, self).__init__()
        seed_everything(seed)
        self.seed          = seed
        self.model         = model
        self.transfom_conf = model
        self.tknz          = tknz
        self.min_f1        = min_f1
        self.device        = device
        self.batch_size    = batch_size
        self.max_drop      = max_drop
        self.nepochs       = nepochs
        self.weight_decay  = weight_decay
        self.lr            = lr
        self.update_drop   = update_drop
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.tknz      = DominanceTokenizer(**self.tknz).fit(X_train, y_train)
        self.transfom_conf['vocab_size'] = len(self.tknz.vect.vocabulary_)+1
        self.transfom_conf['nclass']     = len(set(y_val + y_train))
        self.model     = ETCModel(**self.transfom_conf).to(self.device)
        self.optimizer = AdamW( self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=.95, patience=3, verbose=False)
        best = 99999.
        trained_f1 = (0.,0.)
        counter = 1
        dl_val = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.tknz.collate_val)
        N_train = len(y_train)

        logging_ = []
        with tqdm(total=self.nepochs, position=3, desc="First epoch") as e_pbar:
            with tqdm(total=N_train+len(y_val), position=4, smoothing=0., desc=f"First batch") as b_pbar:
                for e in range(self.nepochs):
                    b_pbar.reset(total=N_train+len(y_val))
                    dl_train = DataLoader(list(zip(X_train, y_train)),
                                          batch_size=self.batch_size, shuffle=True,
                                          collate_fn=self.tknz.collate_train)
                        
                    loss_train  = 0.
                    total = 0.
                    correct  = 0.
                    self.model.train()
                    y_true = []
                    y_preds = []
                    for i, data in enumerate(dl_train):
                        data = { k: v.to(self.device) for (k,v) in data.items() }

                        result = self.model( **data )
                        loss   = result['loss']

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        loss_train    += result['loss'].item()

                        y_pred         = result['logits'].argmax(axis=-1)
                        correct       += (y_pred == data['labels']).sum().item()
                        total         += len(data['labels'])

                        y_true.extend(list(data['labels'].cpu()))
                        y_preds.extend(list(y_pred.cpu()))
                        
                        if self.update_drop:
                            self.model.drop_.p  = (correct/total)*self.max_drop
                        b_pbar.desc = f"--ACC: {(correct/total):.3} ({trained_f1[0]:.2},{trained_f1[1]:.2}) L={(loss_train/(i+1)):.6} b={i+1}"
                        b_pbar.update( len(data['labels']) )
                        del result, data

                    f1_ma = f1_score(y_true, y_preds, average='macro')
                    f1_mi = f1_score(y_true, y_preds, average='micro')
                    trained_f1 = (f1_mi, f1_ma)
                    b_pbar.desc = f"t-F1: ({f1_mi:.3}/{f1_ma:.3}) L={(loss_train/(i+1)):.6}"
                    loss_train = loss_train/(i+1)
                    total = 0.
                    correct  = 0.
                    loss_val = 0.
                    self.model.eval()
                    y_true  = []
                    y_preds = [] 
                    for i, data in enumerate(dl_val):
                        data = { k: v.to(self.device) for (k,v) in data.items() }
                        result = self.model( **data )

                        loss_val   += result['loss'].item()
                        y_pred      = result['logits'].argmax(axis=-1)
                        correct    += (y_pred == data['labels']).sum().item()
                        total      += len(data['labels'])
                        b_pbar.update( len(data['labels']) )

                        y_true.extend(list(data['labels'].cpu()))
                        y_preds.extend(list(y_pred.cpu()))

                        del result, data
                    f1_ma  = f1_score(y_true, y_preds, average='macro')
                    f1_mi  = f1_score(y_true, y_preds, average='micro')
                    metric = (loss_val/(i+1)) / ( f1_ma + f1_mi )
                    logging_.append({'f1_mi': f1_mi, 'f1_ma': f1_ma, 'loss_val': loss_val/(i+1), 'metric': metric})
                    self.scheduler.step(loss_val)

                    if best-metric > 0.0001:
                        best = metric
                        counter = 1
                        best_acc = correct/total
                        best_model = copy.deepcopy(self.model).to('cpu')
                        print(f"*-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}")
                        b_pbar.desc = f"*-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                        e_pbar.desc = f"v-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                    elif counter > 10:
                        break
                    elif trained_f1[0] > self.min_f1 and trained_f1[1] > self.min_f1:
                        counter += 1
                    e_pbar.update(1)
                    b_pbar.update(-(N_train+len(y_val)))
        self.model = best_model.to(self.device)
        return logging_
        
    def predict(self, X):
        dl_test = DataLoader(X, batch_size=self.batch_size, shuffle=False, collate_fn=self.tknz.collate)
        model = self.model.eval()
        y_preds = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dl_test)):
                data = { k: v.to(self.device) for (k,v) in data.items() }
                result = model( **data )
                y_preds.extend(result['logits'].argmax(axis=-1).long().cpu().tolist())
        return y_preds