import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score, matthews_corrcoef
from .util import get_representations, specificity_score, true_positive, true_negative, false_positive, false_negative, scoring

import os, sys
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path).parent
sys.path.append(pPath)

import logging 
logger = logging.getLogger('cross_val')

########################## training ML models functions ##########################
#  
def train_model(X,y,config):
    """
    Trains a machine learning model on the given dataset.

    Parameters:
    ----------
    X : array-like 
        The feature matrix used for training. Each row represents a sample and each column a feature.

    y : array-like 
        The target values (class labels) corresponding to the input samples.

    config : dict
        Dictionary for training configuration. Configuration includes:
            - model : str, default='RandomForest'
                The machine learning algorithm to use. Options can include 'RandomForest','SVM',
                'LogisticRegression','NaiveBayes','DecisionTree'
            - data_fold : str, default='StratifiedGroupKFold'
                The type of KFold for cross validation. 
            - n_fold : int, default=5
                Number of cross-validation folds.
            - scoring : dict
                Metrics to evaluate the model performance.
            - groups : array-like
                Groups (species) corresponding to the input samples
            - genus_level: bool, default=False

    Returns:
    -------
    scores : dict
        Evaluation metrics for the trained model, such as accuracy, precision, recall, MCC and F1-score.

    model : trained sklearn estimator
        The trained machine learning model.

    split_summary : list
        Information of training and validation split.
    """

    model = config['model']
    data_fold = config['data_fold']
    n_fold = config['n_fold']
    scoring = config['scoring']

    if 'groups' in config.keys():
        groups_ = config['groups'] 
    elif data_fold == "StratifiedGroupKFold":
        print("groups are needed in StratifiedGroupKFold")

    family_genus_map = pd.read_excel("dataset/prev_studies/Treasure_Island/Benbow/DatasetSupplementary.xlsx", sheet_name='Benbow')
    family_genus_df = family_genus_map[['Accession','Family','Genus']].drop_duplicates()

    genus_map = pd.Series(family_genus_df['Genus'].values, index=family_genus_df['Accession']).to_dict()

    #if training and validation data set are split at genus level
    if config['genus_level']:
        groups = np.array([genus_map[accession] for accession in groups_])
    else:
        groups = groups_

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(),
        'Bagging':BaggingClassifier(n_jobs=1)
    }

    clf = models[model]
    split_summary = []

    fold = StratifiedGroupKFold(n_splits = n_fold)

    for i, (train, valid) in enumerate(fold.split(X, y, groups)):
        train_X, train_y, train_group = X[train], y[train], groups[train]
        valid_X, valid_y, valid_group = X[valid], y[valid], groups[valid]

        train_perc = len(train)/(len(train)+len(valid))
        valid_perc = len(valid)/(len(train)+len(valid))

        n_group_train = len(set(train_group))
        n_group_valid = len(set(valid_group))

        pos_samples_train = train_y.tolist().count(1)
        neg_samples_train = train_y.tolist().count(0)

        pos_samples_valid = valid_y.tolist().count(1)
        neg_samples_valid = valid_y.tolist().count(0)

        split_summary.append([('{:.2f}/{:.2f}'.format(train_perc,valid_perc),
                                '{}/{}'.format(n_group_train,n_group_valid),
                                '{}/{}'.format(pos_samples_train,neg_samples_train),
                                '{}/{}'.format(pos_samples_valid,neg_samples_valid))])
        
    scores = cross_validate(clf, X, y, cv = fold.split(X, y, groups), scoring = scoring)
    
    return scores, clf, split_summary

def crossval(train_params):
    """
    Run a cross-validation of different machine learning models and data representations.

    Parameters:
    ----------
    train_params : dict 
        Configuration for training machine learning models. 
        - filename : str
            Filename of a training data set. Options can include 'benbow','islandpick','gicluster','rvm'
        - models : list
            List of machine learning models for cross-validation
        - representation : str
            Data representation to convert input sequences into numerical values
        - data_folds : str, default='StratifiedGroupKFold'
            Type of cross-validation fold
        - n_fold : int, default=5
            Number of cross-validation folds
        - representation_params : dict
            Dictionary of representation parameters. This includes parameters related to each corresponding 
            representation, for instance k values for kmer  
        - genus_level : bool, default=False
            whether split the training and validation data sets at Genus level


    Returns:
    -------
    results : dict
        Cross-validation results

    model : trained sklearn estimator
        The trained machine learning model.

    """

    logger = logging.getLogger('util')

    filename = train_params['filename']
    representation = train_params['representation']
    k_max = train_params['representation_params']['k_max']
    k_start = train_params['representation_params']['k_start']
    k_default = train_params['representation_params']['k_default']

    file = os.path.join(pPath, filename)

    if '/' in file:
        data_name = file.split('/')[-1].split('_')[0]
    else:
        data_name = file.split('_')[0]
        
    if 'Kmer' in representation:
        key = '{}-{}'.format(representation,train_params['representation_params']['kmer'])
    else:
        key = representation

    results = {}
    models = {}

    if 'Kmer' in representation:
        for k in range(k_start,k_max+1):
            train_params['representation_params'].update({'kmer':k})

            key = '{}-{}'.format(representation,k)
    
            try:
                logger.info('Representation with {}'.format(key))

                X,y,groups = get_representations(representation, file, train_params['representation_params'])

                for model in train_params['models']:
                    for data_fold in train_params['data_folds']:
                        config = {
                            'model': model,
                            'data_fold': data_fold,
                            'n_fold': train_params['n_fold'],
                            'scoring': scoring,
                            'groups': groups,
                            'genus_level':train_params['genus_level']
                        }

                        logger.info('{},{},{},{},{}'.format(data_name,model,data_fold,train_params['n_fold'],key))

                        logger.info('Training in progress')
                        scores, clf, split_summary = train_model(X,y,config)

                        logger.info('Training is done')

                        for score in scores.keys():
                            results.update({(data_name,
                                            model,
                                            data_fold,
                                            str(train_params['n_fold']),
                                            key,
                                            score):scores[score]})

                        results.update({(data_name,
                                        model,
                                        data_fold,
                                        str(train_params['n_fold']),
                                        key,
                                        'data_split'): split_summary})
                            
                        models.update({(data_name,
                                        model,
                                        data_fold,
                                        str(train_params['n_fold']),
                                        key): clf})
            except Exception as e:
                logger.error('error: {} in {}'.format(e,key), exc_info=True)

                for score in train_params['scoring'].keys():
                    results.update({(data_name,
                                    'model',
                                    'fold',
                                    str(train_params['n_fold']),
                                    key,
                                    'test_'+score):'n/a'})


    else:
        try:
            train_params['representation_params'].update({'kmer':k_default})

            key = representation 

            logger.info('Representation with {}'.format(key))
        
            X,y,groups = get_representations(representation, file, train_params['representation_params'])

            for model in train_params['models']:
                for data_fold in train_params['data_folds']:
                    
                    config = {
                        'model': model,
                        'data_fold': data_fold,
                        'n_fold': train_params['n_fold'],
                        'scoring': scoring,
                        'groups': groups,
                        'genus_level':train_params['genus_level']
                    }

                    logger.info('{},{},{},{},{}'.format(data_name,model,data_fold,train_params['n_fold'],key))

                    logger.info('Training in progress')
                    scores, clf, split_summary = train_model(X,y,config)
                    logger.info('Training is done')

                    for score in scores.keys():
                        results.update({(data_name,
                                        model,
                                        data_fold,
                                        str(train_params['n_fold']),
                                        key,
                                        score):scores[score]})
                    
                    results.update({(data_name,
                                    model,
                                    data_fold,
                                    str(train_params['n_fold']),
                                    key,
                                    'data_split'): split_summary})
                    
                    models.update({(data_name,
                                    model,
                                    data_fold,
                                    str(train_params['n_fold']),
                                    key): clf})
                
        except Exception as e:
            logger.error('error: {} in {}'.format(e,key), exc_info=True)

            for score in scoring.keys():
                results.update({(data_name,
                                'model',
                                'fold',
                                str(train_params['n_fold']),
                                key,
                                'test_'+score):'n/a'})
                
            results.update({(data_name,
                                    model,
                                    data_fold,
                                    str(train_params['n_fold']),
                                    key,
                                    'data_split'): split_summary})
    
    return results, models