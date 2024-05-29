from ..trainers import Trainect
class ETCFTTrainer(Trainect):
    def __init__(self, tname, descriptor):
        super(ETCFTTrainer, self).__init__(tname, descriptor)
    
    def init_model(self, fold, output_dir: str = None):
        return ETCFTlassifier(**self.descriptor)
    
    def train_model(self, model, fold):
        return model.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)
    

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import copy
from .model import seed_everything, ETCModel
from .tokenizer import FastTextEncoder

class ETCFTlassifier():
    def __init__(self, tknz, model, nepochs:int=50,
                max_drop:float=.75, batch_size:int=16, min_f1=.97, seed=42, 
                weight_decay:float = 5e-3, lr:float = 5e-3, update_drop:bool=True, device='cuda'):
        super(ETCFTlassifier, self).__init__()
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
        seed_everything(42)
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.tknz      = FastTextEncoder(**self.tknz).fit(X_train, y_train)
        self.transfom_conf['nclass']     = len(set(y_val + y_train))
        self.transfom_conf['vocab_size'] = len(self.tknz.mapper)
        self.model     = ETCModel(**self.transfom_conf)
        self.optimizer = AdamW( self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=.95, patience=3, verbose=False)
        best = 99999.
        trained_f1 = (0.,0.)
        counter = 1
        dl_val = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size,
                                shuffle=False, num_workers=4, collate_fn=self.tknz.collate_train)
        N_train = len(y_train)

        logging_ = []
        with tqdm(total=self.nepochs, position=3, desc="First epoch") as e_pbar:
            with tqdm(total=N_train+len(y_val), position=4, smoothing=0., desc=f"First batch") as b_pbar:
                for e in range(self.nepochs):
                    self.model.to(self.device)
                    b_pbar.reset(total=N_train+len(y_val))
                    dl_train = DataLoader(list(zip(X_train, y_train)), num_workers=4,
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
                        print(f"{e+1}-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}")
                        b_pbar.desc = f"*-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                        e_pbar.desc = f"v-F1: ({(f1_mi*100.):.3}/{(f1_ma*100.):.3}) L={(loss_val/(i+1)):.6} M={metric:.5}"
                    elif counter > 10:
                        break
                    elif trained_f1[0] > self.min_f1 or trained_f1[1] > self.min_f1:
                        counter += 1
                    e_pbar.update(1)
                    b_pbar.update(-(N_train+len(y_val)))
        self.model = best_model.to(self.device)
        return logging_
        
    def predict(self, X):
        dl_test = DataLoader(X, batch_size=self.batch_size, shuffle=False, collate_fn=self.tknz.collate)
        model = self.model
        model.to(self.device)
        model.eval()
        y_preds = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(dl_test)):
                data = { k: v.to(self.device) for (k,v) in data.items() }
                result = model( **data )
                y_preds.extend(result['logits'].argmax(axis=-1).long().cpu().tolist())
        return y_preds