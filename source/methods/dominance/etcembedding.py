import torch
import torch.nn as nn

class ETCEmbedding(nn.Module):
    def __init__(self, vocab_size, hiddens, drop, att_model='aa'):
        super(ETCEmbedding, self).__init__()
        self.hiddens        = hiddens
        self.drop_          = drop
        self.att_model      = att_model.lower()
        if self.att_model == 'aa':
            self.k = 3
        elif self.att_model == 'ca':
            self.k = 2
        elif self.att_model == 'sa':
            self.k = 1        
        self.emb        = nn.Embedding(vocab_size, self.k*hiddens, scale_grad_by_freq=True, padding_idx=0)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.emb.weight.data)
        self.emb.weight.data[0] = 0
        
    def _get_hidden_(self, input_ids, emb_bag):
        *Bn, b = input_ids.shape
        return emb_bag(input_ids.reshape(prod(Bn), b)).reshape(*Bn, self.hiddens)
    
    def forward(self, doc_tids):
        # doc_tids: [B,L,2]
        batch_size = doc_tids.size(0)
        bx_packed  = doc_tids == 0 # doc_tids: [B,L]
        pad_mask   = bx_packed.logical_not()
        doc_sizes  = pad_mask.sum(dim=1).view(batch_size, 1)
        pad_mask   = pad_mask.view(*bx_packed.shape, 1)
        pad_mask   = pad_mask.logical_and(pad_mask.transpose(1, 2))
        
        H = self.emb( doc_tids ) # B,L,self.k*D
        B,L,D = H.shape
        H = H.reshape(B,L,self.k,D//self.k)
        
        if self.att_model == 'ca':
            K,V = H[:,:,0,:], H[:,:,1,:]
            Q = V
        elif self.att_model == 'aa':
            K,V,Q = H[:,:,0,:], H[:,:,1,:], H[:,:,2,:]
        elif self.att_model == 'sa':
            K = H[:,:,0,:]
            V = K
            Q = K
        else:
            raise Exception("Attention Model not availabel: options ['AA','CA','SA'], for All-Attentions, Cross-Attention, and Self-Attention.")
        
        V = self.drop_( V )
        
        Q = F.tanh( Q )
        Q = self.drop_( Q )
        
        K   = F.tanh( K )
        K   = self.drop_( K )
        
        return { 
            'K': K, 
            'Q': Q, 
            'V': V,
            'bx_packed': bx_packed,
            'doc_sizes': doc_sizes,
            'pad_mask': pad_mask
        }