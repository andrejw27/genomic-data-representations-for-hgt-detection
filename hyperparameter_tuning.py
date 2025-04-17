
import itertools
import time
from tqdm import tqdm
import pickle
import argparse
from multiprocessing import Pool
import multiprocessing

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from utils.hpo import task_hyperparameter_tuning

from logging_config import setup_logging
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-worker", type=int, default=1)
    parser.add_argument("--filename", type=str, default="benbow")
    args = parser.parse_args()
    return args

args = get_args()
filename = args.filename

logger = logging.getLogger(__name__)

log_path = os.path.join(pPath,'logs/hpo')

if not os.path.exists(log_path):
    logger.info('Creating Logging Folder')
    os.mkdir(log_path)

log_filename = os.path.join(log_path,'{}.log'.format(filename))
setup_logging(log_filename=log_filename)

def main():

    #parameters
    filename = args.filename
    files = ['dataset/train_data/'+filename+'_data.fasta']
  
    representations_models = {
        'benbow': [('RCKmer-7','svm'), ('Kmer-6','svm'), ('PCPseTNC','rf'), 
                    ('Z_curve_48bit','svm'), ('Z_curve_48bit','rf'), ('PseEIIP','rf')],
        'islandpick': [('RCKmer-7','svm'), ('Kmer-7','svm'), ('Z_curve_48bit','svm'), ('Z_curve_48bit','rf'), 
                        ('PseKNC','rf'), ('PCPseTNC','rf')],
        'gicluster': [('RCKmer-7','rf'), ('Kmer-6','rf'), ('Subsequence','lr'), ('Mismatch','lr'), ('CKSNAP','svm')],
        'rvm': [('Kmer-5','rf'), ('RCKmer-5','rf'), ('PCPseTNC','rf'), ('Z_curve_48bit','rf'), ('PseEIIP','rf')]
    }
    
    list_params = []

    for elements in itertools.product(files, representations_models[filename]) :
        (file, representation_model) = elements
        (representation, model) = representation_model

        if 'Kmer' in representation or 'kmer' in representation:
            kwargs = {'k':int(representation.split('-')[-1])}
            representation = representation.split('-')[0]
        else:
            kwargs = {}
        
        hpo_params = {
            'filename': file,
            'model': model,
            'representation': representation,
            'kwargs': kwargs
        }

        list_params.append(hpo_params)

    if len(list_params)>1 and args.n_worker > 1:
        n_worker = multiprocessing.cpu_count()
        n_worker = min(args.n_worker,n_worker)

        with Pool(n_worker) as p:
            results = p.map(task_hyperparameter_tuning, tqdm(list_params))
    else:
        results = [task_hyperparameter_tuning(list_params[0])]

    logger.info('Done Hyperparameter Tuning')

    hpo_results = {}

    for res in results:
        hpo_results.update(res)

    hpo_path = os.path.join(pPath,'hpo')

    if not os.path.exists(hpo_path):
        logger.info('Creating HPO Folder')
        os.mkdir(hpo_path)

    logger.info('Saving Output')

    for combination in hpo_results:
        temp_res = {}
        temp_res.update({'cv_results':hpo_results[combination].cv_results_})
        temp_res.update({'best_params':hpo_results[combination].best_params_})
        temp_res.update({'best_index':hpo_results[combination].best_index_})

        with open(os.path.join(hpo_path,"{}.pkl".format(combination)), 'wb') as f:
            pickle.dump(temp_res, file=f)

    logger.info('Done')

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start ---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))