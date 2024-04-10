from os import path
from .base import create_path, read_lines, download
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import torch

class Fold(object):
    def __init__(self, dname, foldname, content, labels, nclass, train_idxs, test_idxs, fidx=0, val_idxs=[]):
        super(Fold, self).__init__()
        self.dname = dname
        self.fold_idx = fidx
        self.foldname = foldname
        self.content = content
        self.labels = labels
        self.N      = len(labels)
        self.nclass =  nclass
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.with_val = len(val_idxs) > 0
        self.n = (len(train_idxs), len(val_idxs), len(test_idxs)) if self.with_val else (len(train_idxs), len(test_idxs))

    @property
    def X_train(self):
        return [ self.content[idx] for idx in self.train_idxs ]
    @property
    def y_train(self):
        return [ self.labels[idx] for idx in self.train_idxs ]

    @property
    def X_test(self):
        return [ self.content[idx] for idx in self.test_idxs ]
    @property
    def y_test(self):
        return [ self.labels[idx] for idx in self.test_idxs ]

    @property
    def X_val(self):
        if not self.with_val:
            raise Exception(f"{self.foldname}-fold of the dataset {self.dname} without validation set.")
        return [ self.content[idx] for idx in self.val_idxs ]
    @property
    def y_val(self):
        if not self.with_val:
            raise Exception(f"{self.foldname}-fold of the dataset {self.dname} without validation set.")
        return [ self.labels[idx] for idx in self.val_idxs ]

    def to(self, format, **kargs):
        if format == 'etc':
            return self
        if format == 'hugging':
            from datasets import DatasetDict
            tokenizer = kargs['tokenizer'] if 'tokenizer' in kargs else None
            return DatasetDict(train=HugDataset(self.X_train, y=self.y_train, tokenizer=tokenizer),
                       val=HugDataset(self.X_val, y=self.y_val, tokenizer=tokenizer),
                       test=HugDataset(self.X_test, y=self.y_test, tokenizer=tokenizer))
        raise Exception(f'Format {format} not found. Options: [etc, hugging]')
class Dataset(object):
    def __init__(self, dname, nfold=10,
            dpath='~/.etc/', drepo='http://150.164.2.44/'):
        super(Dataset, self).__init__()
        self.dname = dname
        self.nfold = nfold
        self.drepo = drepo + f"datasets/{self.dname}"
        self.le = LabelEncoder()
        self.dpath = path.abspath(path.join(path.expanduser(dpath), "datasets", self.dname))
        self.init_dataset()
    
    def init_dataset(self):
        tpath = path.join(self.dpath, 'texts.txt') # [T]ext  Path
        lpath = path.join(self.dpath, 'score.txt') # [L]abel Path
        spath = path.join(self.dpath, 'splits') #    [S]plit Path

        if not path.exists(spath):
            create_path(spath)
        
        if not path.exists(tpath):
            download( tpath, self.drepo+'/texts.txt' )

        if not path.exists(lpath):
            download( lpath, self.drepo+'/score.txt' )

        self.texts  = read_lines(tpath)
        self.y      = read_lines(lpath)
        self.y      = list(map(int, self.y))
        self.y      = self.le.fit_transform(self.y)
        self.nclass = len(set(self.y))
        self.N      = len(self.y)
        self.splits = dict()
        self.n = len( self.get_split(self.nfold) )

            
    def load_split(self, foldname):
        spath = path.join(self.dpath, 'splits', f"split_{foldname}.csv") #    [S]plit Filepath
        if not path.exists(spath):
            download( spath, self.drepo+f"/splits/split_{foldname}.csv" )
        split = read_lines(spath)
        split = map( lambda x: x.split(';'), split )
        split = map( lambda part: map(str.split, part), split ) 
        split = map( lambda part: list(map(lambda idxs:  list(map(int, idxs)), part)), split ) 
        split = map( lambda part: part if len(part)==3 else self._create_val_(*part) , split)
        return list(split)
    
    def _create_val_(self, train_ids, test_ids):
        from sklearn.model_selection import train_test_split
        try:
            train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                            test_size=len(test_ids),
                                            stratify=[ self.y[i] for i in train_ids ],
                                            random_state=42)
        except ValueError:
            train_idx_atual, val_idx_atual = train_test_split(train_ids,
                                            test_size=len(test_ids),
                                            random_state=42)

        return (train_idx_atual, val_idx_atual, test_ids)

    def get_split(self, foldname):
        if foldname not in self.splits:
            self.splits[foldname] = self.load_split(foldname)
        
        return self.splits[foldname]
    
    def iter_folds(self, foldname):
        for (i,idxs) in enumerate(self.get_split(foldname)):
            params = {
                'fidx': i,
                'dname': self.dname,
                'foldname': foldname,
                'content': self.texts,
                'labels': self.y,
                'nclass': self.nclass,
                'train_idxs': idxs[0],
                'test_idxs': idxs[-1]
            }
            if len(idxs) == 3:
                params['val_idxs'] = idxs[1]
            yield Fold( **params )
    
    def __iter__(self):
        if self.nfold is None:
            raise Exception("Fold name can't be None.")
        return iter(self.iter_folds(self.nfold))
class HugDataset(data.Dataset):
    def __init__(self, X, y=None, tokenizer=None, max_length=256):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_pos = False

    def __getitem__(self, idx):
        text = str(self.X[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if self.random_pos:
            encoding['input_ids'] = self.shuffle_tokens(encoding['input_ids'])

        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.y is not None:
            result['labels'] = torch.tensor(self.y[idx], dtype=torch.long)

        return result
    def shuffle_tokens(self, tensor):
        # Create a mask for non-zero elements
        tensor = tensor.squeeze()
        non_zero_mask = (tensor != 0)

        # Get the indices of non-zero elements
        non_zero_indices = torch.arange(tensor.size(0))[non_zero_mask]

        # Shuffle the indices of non-zero elements
        shuffled_non_zero_indices = torch.randperm(non_zero_indices.size(0))

        # Reorder the non-zero elements using the shuffled indices
        shuffled_tensor = torch.zeros_like(tensor)
        shuffled_tensor[non_zero_indices] = tensor[non_zero_indices][shuffled_non_zero_indices]
        return shuffled_tensor.unsqueeze(0)
    
    def __len__(self):
        return len(self.X)