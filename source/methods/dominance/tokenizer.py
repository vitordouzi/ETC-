from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from tqdm.auto import tqdm
import torch
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
import numpy as np
from torch.nn.utils.rnn import  pad_sequence
from tqdm.auto import tqdm

class DominanceTokenizer():
    def __init__(self, norm=True, **kargs):
        self.vect = CountVectorizer(**kargs, binary=True)
        self.lbin = LabelBinarizer(sparse_output=True)
        self.norm = norm
    def fit(self, X, y):
        data = self.vect.fit_transform(tqdm(X, total=len(y)))
        self.N,self.V = data.shape
        
        Y      = self.lbin.fit_transform(y)
        self.dominance = (data.T @ Y)
        self.max_doc_len = int(np.percentile(np.array(data.sum(axis=1))[:,0], 90))
        self.df_ = np.array(data.sum(axis=0))[0,:]
        return self
    def transform(self, X):
        rows,cols = [],[]
        for i,doc in enumerate(self.vect.transform(X)):
            terms = doc.nonzero()[1]
            if len(terms) > self.max_doc_len:
                idf_ts = self.df_[terms]
                _,terms = zip(*sorted( zip(idf_ts, terms)))
                terms = list(terms)[:self.max_doc_len]
            cols.extend( terms )
            rows.extend( [ i for _ in terms ] )
        return csr_matrix((np.ones_like(rows),(rows, cols)), shape=(i+1, self.V))    
    def collate(self, X):
        data      = self.transform(X)
        
        tkns_btch = [ torch.LongTensor(doc.nonzero()[1])+1 for doc in data ]    
        tkns_btch = pad_sequence(list(tkns_btch), batch_first=True, padding_value=0)
        
        doms_btch = [ self.dominance[doc.nonzero()[1],:].A for doc in data ]  
        doms_btch = map(torch.tensor, doms_btch)
        doms_btch = pad_sequence(list(doms_btch), batch_first=True, padding_value=0.)*1.
        
        return { 'input_ids': tkns_btch, 'dom': doms_btch}
    def collate_val(self, params):
        X,y    = zip(*params)
        result = self.collate(X)
        result['labels'] =  torch.LongTensor(y)
        
        return result
    def collate_train(self, params):
        X,y       = zip(*params)
        data      = self.transform(X)
        Y         = self.lbin.transform(y)
        if self.norm:
            dom_norm  = self.dominance - (data.T @ Y)
        else:
            dom_norm  = self.dominance
        
        dom_batch = [ dom_norm[doc.nonzero()[1],:].A for doc in data ]    
        
        tkns_btch = [ torch.LongTensor(doc.nonzero()[1])+1 for doc in data ]    
        tkns_btch = pad_sequence(list(tkns_btch), batch_first=True, padding_value=0)
                
        dom_batch = map(torch.tensor, dom_batch)
        dom_batch = pad_sequence(list(dom_batch), batch_first=True, padding_value=0.)*1.
        
        return { 'input_ids': tkns_btch, 'dom': dom_batch, 'labels': torch.LongTensor(y) }
