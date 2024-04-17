from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn

def removeNaN(data):
    zeros   = torch.zeros_like(data)
    isnan   = torch.isnan(data)
    return torch.where(isnan, zeros, data)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.nn.utils.rnn import  pad_sequence
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy

class DominanceTokenizer():
    def __init__(self, norm=True, **kargs):
        self.vect = CountVectorizer(**kargs, binary=True)
        self.lbin = LabelBinarizer(sparse_output=True)
        self.norm = norm
    def fit(self, X, y):
        X_data = self.vect.fit_transform(tqdm(X, total=len(y)))
        Y      = self.lbin.fit_transform(y)
        self.dominance = (X_data.T @ Y)
        return self
    def transform(self, X):
        return self.vect.transform(X)
    
    def collate(self, X):
        data      = self.transform(X)
        
        tkns_btch = [ torch.LongTensor(doc.nonzero()[1])+1 for doc in data ]    
        tkns_btch = pad_sequence(list(tkns_btch), batch_first=True, padding_value=0)
        
        dom_batch = [ self.dominance[doc.nonzero()[1],:].A for doc in data ]  
        dom_batch = map(torch.tensor, dom_batch)
        dom_batch = pad_sequence(list(dom_batch), batch_first=True, padding_value=0.)*1.
        
        return { 'input_ids': tkns_btch, 'dom': dom_batch}
        
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
    