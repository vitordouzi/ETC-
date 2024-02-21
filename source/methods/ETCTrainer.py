from .trainers import Trainect
from .ETC.classifier import ETCClassifier

class ETCTrainer(Trainect):
    def __init__(self, tname, descriptor):
        super(ETCTrainer, self).__init__(tname, descriptor)
    
    def init_model(self, fold, output_dir: str = None):
        return ETCClassifier(**self.descriptor)
    
    def train_model(self, model, fold):
        return model.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)