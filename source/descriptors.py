from copy import deepcopy
import torch

########################################### Datasets ###########################################
DESC_DST = {
    'yelp_2015': { "nfold": 5, "dname": "yelp_2015" },
    'sogou': { "nfold": 5, "dname": "sogou" },
    'imdb_reviews': { "nfold": 5, "dname": "imdb_reviews" },
    'agnews': { "nfold": 5, "dname": "agnews" },

    'webkb': { "nfold": 10, "dname": "webkb" },
    '20ng': { "nfold": 10, "dname": "20ng" } ,
    'reut90': { "nfold": 10, "dname": "reut90" },
    'acm': {  "nfold": 10, "dname": "acm" },
    'patents': {  "nfold": 10, "dname": "patents" },
    'arxiv': {  "nfold": 10, "dname": "arxiv" },

    'ohsumed': {  "nfold": 10, "dname": "ohsumed" },
    'wos11967': { "nfold":10, "dname": "wos11967" },
    'yelp_reviews': { "nfold": 10, "dname": "yelp_review_2L" },
    'vadermovie': { "nfold": 10, "dname": "vader_movie_2L" },
    'dblp': { "nfold": 10, "dname": "dblp" },
    'aisopos_ntua': { "nfold": 10, "dname": "aisopos_ntua_2L" },
    'books': { "nfold": 10, "dname": "books" }
}



########################################### Methods ###########################################
DESC_CLS = {}

########################### HuggingFace ###########################
base_HF = {
    'classpath': 'source.methods.HuggingFaceTrainer.HuggingFaceTrainer',
    'init_params': { 'tname': 'HF-',
                     'descriptor': { 
                         'model_name': None,
                         'batch_size': 32,
                         'max_length': 256,
                         'epochs': 5,
                         'learning_rate': 5e-5
                     }
                    }

}

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'bert-mini'
copied['init_params']['descriptor']['model_name'] = 'prajjwal1/bert-mini'
DESC_CLS['bert-mini'] = copied

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'bert'
copied['init_params']['descriptor']['model_name'] = 'bert-base-uncased'
DESC_CLS['bert'] = copied

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'albert'
copied['init_params']['descriptor']['model_name'] = 'albert-base-v2'
DESC_CLS['albert'] = copied

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'roberta'
copied['init_params']['descriptor']['model_name'] = 'roberta-base'
DESC_CLS['roberta'] = copied

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'xlnet'
copied['init_params']['descriptor']['model_name'] = 'xlnet-base-cased'
DESC_CLS['xlnet'] = copied

copied = deepcopy(base_HF)
copied['init_params']['tname'] += 'distilbert'
copied['init_params']['descriptor']['model_name'] = 'distilbert-base-uncased'
DESC_CLS['distilbert'] = copied


########################### ETC ###########################
base_ETC = {
    'classpath': 'source.methods.ETCTrainer.ETCTrainer',
    'init_params': { 'tname': 'ETC-',
                     'descriptor': { 
                        'tknz':  { 
                            'min_df': 2,
                            'max_features': 500_000,
                            'stop_words': 'both',
                            'ngram_range': (1,2),
                            'with_CLS': False
                        },
                        'model': {
                            "gamma": 0.,
                            "hiddens": 300,
                            'nheads': 12,
                            'att_model': 'aa',
                            'sim_func': 'inner',
                            'norep': 0,
                            'use_tf': False,
                            'use_df': False,
                            'drop': .3
                        },
                        'nepochs': 50,
                        'max_drop': .75,
                        'batch_size': 16,
                        'min_f1': .95,
                        'seed': 42, 
                        'weight_decay': 5e-3,
                        'lr': 5e-3,
                        'update_drop': False,
                        'device': "cuda:0" if torch.cuda.is_available() else 'cpu'
                    }
                }
}

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'base'
copied['init_params']['descriptor']['model']['gamma'] = 5.
copied['init_params']['descriptor']['model']['sim_func'] = 'dist'
copied['init_params']['descriptor']['model']['norep'] = 2
copied['init_params']['descriptor']['model']['use_tf'] = True
copied['init_params']['descriptor']['model']['use_df'] = True
copied['init_params']['descriptor']['update_drop'] = True
DESC_CLS['etc-base'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'drop'
copied['init_params']['descriptor']['update_drop'] = True
DESC_CLS['etc-drop'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'DF'
copied['init_params']['descriptor']['model']['use_df'] = True
DESC_CLS['etc-df'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'TF'
copied['init_params']['descriptor']['model']['use_tf'] = True
DESC_CLS['etc-tf'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'scape'
copied['init_params']['descriptor']['model']['norep'] = 2
DESC_CLS['etc-scape'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'sim'
copied['init_params']['descriptor']['model']['sim_func'] = 'sim'
DESC_CLS['etc-sim'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += 'dist'
copied['init_params']['descriptor']['model']['sim_func'] = 'dist'
DESC_CLS['etc-dist'] = copied

copied = deepcopy(base_ETC)
copied['init_params']['tname'] += ''
copied['init_params']['descriptor']['model']['gamma'] = 5.
DESC_CLS['etc-fl'] = copied