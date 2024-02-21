import torch
import torch.nn as nn

class ETCEmbedding(nn.Module):
    def __init__(self, vocab_size, hiddens, drop, maxF=20, att_model='aa', use_tf=True, use_df=True):
        super(ETCEmbedding, self).__init__()
        self.hiddens        = hiddens
        self.maxF           = maxF
        self.drop_          = drop
        self.att_model      = att_model.lower()
        self.use_tf         = use_tf
        self.use_df         = use_tf
        self.key_emb        = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        if self.use_tf:
            self.TF_emb     = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        if self.use_df:
            self.DF_emb     = nn.Embedding(maxF, hiddens, scale_grad_by_freq=True, padding_idx=0)
        if self.att_model == 'aa':
            self.value_emb  = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
            self.query_emb  = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        elif self.att_model == 'ca':
            self.query_emb  = nn.Embedding(vocab_size, hiddens, scale_grad_by_freq=True, padding_idx=0)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.key_emb.weight.data)
        if self.att_model == 'aa':
            nn.init.xavier_normal_(self.query_emb.weight.data)
            nn.init.xavier_normal_(self.value_emb.weight.data)
        elif self.att_model == 'ca':
            nn.init.xavier_normal_(self.query_emb.weight.data)
        if self.use_tf:
            nn.init.xavier_normal_(self.TF_emb.weight.data)
        if self.use_df:
            nn.init.xavier_normal_(self.DF_emb.weight.data)
        
    def forward(self, doc_tids, TFs, DFs):
        batch_size = doc_tids.size(0)
        bx_packed  = doc_tids == 0
        pad_mask   = bx_packed.logical_not()
        doc_sizes  = pad_mask.sum(dim=1).view(batch_size, 1)
        pad_mask   = pad_mask.view(*bx_packed.shape, 1)
        pad_mask   = pad_mask.logical_and(pad_mask.transpose(1, 2))
        
        h_key = self.key_emb( doc_tids )
        
        if self.att_model == 'ca':
            h_query = self.query_emb( doc_tids )
            h_value = h_key
        elif self.att_model == 'aa':
            h_query = self.query_emb( doc_tids )
            h_value = self.value_emb( doc_tids )
        elif self.att_model == 'sa':
            h_query = h_key
            h_value = h_key
        else:
            raise Exception("Attention Model not availabel: options ['AA','CA','SA'], for All-Attentions, Cross-Attention, and Self-Attention.")
        
        if self.use_tf:
            TFs     = torch.clamp( TFs, max=self.maxF-1 )
            h_TFs   = self.TF_emb( TFs )
            h_TFs   = self.drop_(h_TFs)
            h_value = h_value + h_TFs
            h_query = h_query + h_TFs
            h_key   = h_key + h_TFs
        if self.use_df:
            DFs     = torch.clamp( DFs, max=self.maxF-1 )
            h_DFs   = self.DF_emb( DFs )
            h_DFs   = self.drop_(h_DFs)
            h_value = h_value + h_DFs
            h_query = h_query + h_DFs
            h_key   = h_key + h_DFs
        
        h_key   = torch.tanh( h_key )
        h_query = torch.tanh( h_query )
        h_value = self.drop_( h_value )
        h_query = self.drop_( h_query )
        h_key   = self.drop_( h_key )
        
        result = { 
            'K': h_key, 
            'Q': h_query, 
            'V': h_value,
            'bx_packed': bx_packed,
            'doc_sizes': doc_sizes,
            'pad_mask': pad_mask
        }
        
        return result