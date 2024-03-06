from os import path
from ..utils.base import create_path, seed_everything, from_descriptor
from tqdm.auto import tqdm
from ..utils.dataset import Dataset

class Experiment(object):
    def __init__(self, dst_descs, trnr_descs, output_path='~/.etc/', force:list=None, seed=42):
        super(Experiment, self).__init__()
        self.datasets = [ Dataset(**desc, dpath=output_path) for desc in dst_descs ]
        self.trnr = [ from_descriptor(desc) for desc in trnr_descs ]
        self.N = len(self.datasets)*len(self.trnr)
        self.force = force
        self.output_path = path.abspath(path.join(path.expanduser(output_path), "results"))
        self.seed = seed
        self.init_experiments()

    def init_experiments(self):
        for dst in self.datasets:
            create_path(path.join(self.output_path, dst.dname))

    def run(self):
        for dst in tqdm(self.datasets, desc=f"Dataset...", position=5):
            for trnr in tqdm(self.trnr, desc=f"Method...", position=4):
                for fold in tqdm(dst, desc=f"Fold...", position=3, total=dst.n):
                    seed_everything(self.seed)
                    foutput_path = path.join(self.output_path, dst.dname, trnr.tname, f"fold-{fold.foldname}", str(fold.fold_idx))
                    trnr.run( fold, foutput_path, force = self.force )
