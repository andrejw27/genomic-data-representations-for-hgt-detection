import numpy as np 
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, GridSearchCV
from .util import get_representations, scoring 

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

import os, sys, re
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

import logging 
logger = logging.getLogger('hpo')

########################## hyperparameter tuning functions ##########################

def hyperparameter_tuning(filename, model, representation, **kwargs):

    #reference: https://github.com/Superzchen/iLearnPlus/blob/main/iLearnPlusEstimator.py
    #parameters for features representation
    desc_default_para = {             # default parameter for descriptors
        'sliding_window': 5,
        'kspace': 3,
        'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
        'nlag': 2,
        'weight': 0.05,
        'lambdaValue': 3,
        'PseKRAAC_model': 'g-gap',
        'g-gap': 2,
        'k-tuple': 2,
        'RAAC_clust': 1,
        'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
        'kmer': 3,
        'mismatch': 1,
        'delta': 0,
        'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
        'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
        'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
        'distance': 0,
        'cp': 'cp(20)',
        'k_max': 7,
        'k_default': 3
    }  

    para_dict = {
        'EAAC': {'sliding_window': 5},
        'CKSAAP': {'kspace': 3},
        'EGAAC': {'sliding_window': 5},
        'CKSAAGP': {'kspace': 3},
        'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
        'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
        'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
        'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
        'KSCTriad': {'kspace': 3},
        'SOCNumber': {'nlag': 3},
        'QSOrder': {'nlag': 3, 'weight': 0.05},
        'PAAC': {'weight': 0.05, 'lambdaValue': 3},
        'APAAC': {'weight': 0.05, 'lambdaValue': 3},
        'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
        'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
        'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
        'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
        'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
        'Kmer': {'kmer': 3},
        'RCKmer': {'kmer': 3},
        'Mismatch': {'kmer': 3, 'mismatch': 1},
        'Subsequence': {'kmer': 3, 'delta': 0},
        'ENAC': {'sliding_window': 5},
        'CKSNAP': {'kspace': 3},
        'DPCP': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
        'DPCP type2': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
        'TPCP': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
        'TPCP type2': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
        'DAC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
        'DCC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
        'DACC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
        'TAC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
        'TCC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
        'TACC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
        'PseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
        'PseKNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
        'PCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
        'PCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
        'SCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
        'SCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
    }

    # copy parameters for each descriptor
    if representation in para_dict:
        for key in para_dict[representation]:
            desc_default_para[key] = para_dict[representation][key]

    if representation in ['Kmer', 'RCKmer']:
        if 'k' in kwargs.keys():
            k = kwargs['k']
            desc_default_para.update({'kmer':k})
        else:
            k = desc_default_para['kmer']

        key = '{}-{}'.format(representation,k)
    else:
        key = representation

    
    # Hyperparameter grid 
    param_grid_search = {
        'rf':{
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'criterion': ['entropy','gini'],
            'max_features': ['sqrt', 'log2'],
        },
        'svm':{
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf'],
            'gamma': ['scale', 0.0001, 0.01, 1.0]
        },
        'lr':{
            'penalty':['l1','l2','elasticnet','none'],
            'C' : np.logspace(-4,4,20),
            'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
            'max_iter'  : [100,1000,2500,5000]
        }
    }

    # for script testing only
    # param_grid_search = {
    #     'rf':{
    #         'n_estimators': [100],
    #         'max_depth': [None],
    #         'min_samples_split': [2],
    #         'min_samples_leaf': [2],
    #         'criterion': ['gini'],
    #         'max_features': ['sqrt'],
    #     },
    #     'svm':{
    #         'C': [10],
    #         'kernel': ['rbf'],
    #         'gamma': ['scale']
    #     },
    #     'lr':{
    #         'penalty':['l1'],
    #         'C' : np.logspace(-4,4,20),
    #         'solver': ['lbfgs'],
    #         'max_iter'  : [100]
    #     }
    # }

    # Define the classifier
    clf = {
        'rf': RandomForestClassifier(random_state=42),
        'svm': SVC(random_state=42),
        'lr': LogisticRegression(random_state=42),
    }

    file = os.path.join(pPath, filename)

    logger.info('representation Data with {}'.format(key))
    X_train,y_train,groups_train = get_representations(representation, file, desc_default_para)

    data_name = filename.split('/')[-1]
    data_name = data_name.split('.')[0]
    data_name = data_name.split('_')[0]

    logger.info('Grid Search {} {} {}'.format(data_name,model,key))

    # Define StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=clf[model], param_grid=param_grid_search[model], cv=cv, scoring=scoring, refit='F_1')

    # Fit the model
    grid_search.fit(X_train, y_train, groups=groups_train)

    return {'{}_{}_{}'.format(data_name,model,key):grid_search}

def task_hyperparameter_tuning(hpo_params):
    file = hpo_params['filename']
    model = hpo_params['model']
    representation = hpo_params['representation']

    if 'kwargs' in hpo_params.keys():
        kwargs = hpo_params['kwargs']
    else:
        kwargs = {}

    grid_search_res = hyperparameter_tuning(file, model, representation, **kwargs)

    return grid_search_res