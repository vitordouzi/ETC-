from os import path, mkdir
import numpy as np
import json
import io
import importlib

def from_descriptor(descriptor):
    module_name, class_name = descriptor['classpath'].rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    class_obj = module_obj.__getattribute__(class_name)
    return class_obj(**descriptor['init_params'])

def seed_everything(seed: int):
    import random, os
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_path(path_to_create):
    path_to_create = path.abspath(path_to_create)
    paths = path_to_create.split(path.sep)
    complete_path = '/'
    for p in paths[1:]:
        complete_path = path.join(complete_path, p)
        if not path.exists(complete_path):
            mkdir( complete_path )

def read_lines(filename):
    with io.open(filename, newline='\n') as filin:
        return filin.readlines()

def save_json(path_json_file, data):
    with open(path_json_file, 'w') as file_out:
        json.dump(data, file_out, cls=NumpyEncoder)

def load_json(path_json_file):
    with open(path_json_file, 'r') as file_in:
        return json.load(file_in)

def download(filename, url):
    import urllib3, shutil
    
    http = urllib3.PoolManager()
    with open(filename, 'wb') as out:
        print(url)
        r = http.request('GET', url, preload_content=False)
        shutil.copyfileobj(r, out)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
