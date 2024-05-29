from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from unicodedata import normalize
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

try:
    from nltk.corpus import stopwords as stopwords_by_lang
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords as stopwords_by_lang

from gensim.models import FastText
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words  
from nltk.corpus import stopwords as stopwords_by_lang
nltk_stopwords = set(stopwords_by_lang.words('english'))
skl_stopwords  = set(stop_words)
def bigrams(seq):
    for i,ti in enumerate(seq):
        yield ti
        if i>0:
            yield ' '.join([seq[i-1],ti])
class FastTextEncoder():
    def __init__(self, model, vocab_size=2**20, min_df=2, max_doc_size=2**10):
        self.analyzer = CountVectorizer().build_analyzer()
        self.vocab_final_size = vocab_size
        self.min_df = min_df
        self.max_doc_size = max_doc_size
        self.stopwords = skl_stopwords.union(nltk_stopwords)
        if isinstance(model, str):
            self.model = FastText.load(model)
        else:
            self.model = model
    def fit(self, X,y=None):
        DF = Counter()
        docs = map(self.tokenizer, X)
        docs = map(DF.update, docs)
        list(docs)
        self.DF = { k: v for (k,v) in DF.most_common(self.vocab_final_size) if v >= self.min_df }
        self.mapper = { t: i+2 for (i,t) in enumerate(self.DF) }
        self.mapper['<PAD>'] = 0
        self.mapper['<UNK>'] = 1
        self.d = self.model.wv.vector_size
        return self
    def getValues(self, tkn):
        tid = self.mapper[tkn] if tkn in self.mapper else 1
        wv = self.model.wv.get_sentence_vector( tkn.split(' ') )
        return (tid, wv)
    def transform(self, X):
        docs = map(self.tokenizer, X)
        docs = map(lambda d: map(self.getValues, d), docs)
        docs = map(lambda d: list(sorted(d, reverse=True, key=lambda v: v[0]))[:self.max_doc_size], docs)
        docs = list(docs)
        
        tids = map(lambda d: [ tid for tid,wv in d ] if len(d) else [0], docs)
        wv_v = map(np.stack,  map(lambda d: [ wv for tid,wv in d ] if len(d) else [np.zeros(self.d)], docs))
        return list(wv_v), list(tids)
    def tokenizer(self, doc):
        doc = normalize('NFKD', doc)
        doc = self.analyzer(doc)
        doc = list(filter(lambda t: t not in self.stopwords, doc))
        doc = bigrams(doc)
        return set(doc)
    def collate(self, X):
        wv,tids = self.transform(X)
        
        wv = map(torch.FloatTensor, wv)
        wv = pad_sequence(wv, batch_first=True)
        
        tids = map(torch.LongTensor, tids)
        tids = pad_sequence(tids, batch_first=True)
        return { 'wv': wv, 'input_ids': tids }
    def collate_train(self, params):
        X,y = zip(*params)
        result = self.collate(X)
        result['labels'] = torch.tensor(y)
        return result