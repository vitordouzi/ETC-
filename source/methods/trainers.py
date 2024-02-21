from ..utils.dataset import Fold
from ..utils.base import create_path, save_json, load_json
from os import path
from collections import defaultdict
from ..utils.tick import Tick
import traceback
from sklearn.metrics import f1_score


class Trainect(object):
    def __init__(self, tname, descriptor):
        super(Trainect, self).__init__()
        self.tname      = tname
        self.descriptor = descriptor
        self.t_eval     = defaultdict(Tick)
        self.r_eval     = defaultdict(dict)
    
    def init_model(self, fold, output_dir: str = None):
        raise NotImplementedError('Abstraction')    
    
    def train_model(self, model, fold):
        raise NotImplementedError('Abstraction')      
    
    def predict(self, model, X):
        return model.predict(X)
    def save(self, output_path):
        result = {
            'tname': self.tname,
            'descriptor': self.descriptor,
            't_eval': { k: v.time for (k,v) in self.t_eval.items() },
            'r_eval': self.r_eval
        } 
        create_path(output_path)
        save_json(path.join(output_path, 'result.json'), result)
    def done(self, output_path):
        try:
            result_path = path.join(output_path, 'result.json')
            if not path.exists(result_path):
                return False
            result = load_json(result_path)
            return 'r_eval' in result and  \
                'status' in result['r_eval'] and \
                result['r_eval']['status'] == 'DONE' 
        except:
            return False
    def run(self, fold: Fold, output_dir: str = None, force:list=None):
        model = None
        if (self.done(output_dir) and force is None) or (force is not None and fold.fold_idx not in force):
            return None
        try:
            self.t_eval['init'].tick
            model = self.init_model(fold, output_dir)
            self.t_eval['init'].tick

            self.t_eval['train'].tick
            self.r_eval['train'] = self.train_model(model, fold)
            self.t_eval['train'].tick

            self.t_eval['train.pred'].tick
            self.r_eval['train.pred'] = self.predict(model, fold.X_train)
            self.t_eval['train.pred'].tick
            self.r_eval['train.true'] = fold.y_train

            if fold.with_val:
                self.t_eval['eval.pred'].tick
                self.r_eval['eval.pred'] = self.predict(model, fold.X_val)
                self.t_eval['eval.pred'].tick
                self.r_eval['eval.true'] = fold.y_val

            self.t_eval['test.pred'].tick
            self.r_eval['test.pred'] = self.predict(model, fold.X_test)
            self.t_eval['test.pred'].tick
            self.r_eval['test.true'] = fold.y_test

            f_mi = f1_score(self.r_eval['test.true'], self.r_eval['test.pred'], average='micro')
            f_ma = f1_score(self.r_eval['test.true'], self.r_eval['test.pred'], average='macro')

            print( fold.dname, self.tname, f"{fold.fold_idx}/{fold.foldname}", f_mi, f_ma )

            self.r_eval['status'] = 'DONE'

        except KeyError as error:
            raise KeyError(error)
        except Exception as error:
            self.r_eval['status'] = 'ERROR'
            traceb = traceback.format_exc()
            self.r_eval['status.error_msg'] = traceb
            print(traceb)
        self.save(output_dir)
        return model
    def __str__(self):
        return f'<Trainect ({self.tname}>'