
import itertools
import time
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import multiprocessing

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from utils.util import multiindex_dict_to_df
from utils.train import crossval
from utils.Parameters import Parameters

from logging_config import setup_logging
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-index", type=int, default=1)
    parser.add_argument("--n-worker", type=int, default=1)
    parser.add_argument("--filename", type=str, default="benbow")
    args = parser.parse_args()
    return args

args = get_args()
filename = args.filename
idx = args.representation_index

logger = logging.getLogger(__name__)

log_path = os.path.join(pPath,'logs/cross_val/{}'.format(filename))

if not os.path.exists(log_path):
    logger.info('Creating Logging Folder')
    os.mkdir(log_path)

log_filename = os.path.join(log_path,'{}_{}.log'.format(filename,idx))
setup_logging(log_filename=log_filename)

def main():
    
    #reference: https://github.com/Superzchen/iLearnPlus/blob/main/iLearnPlusEstimator.py
    #parameters for features representation

    parameters = Parameters()

    desc_default_para = parameters.DESC_DEFAULT_PARA

    para_dict = parameters.PARA_DICT

    #parameters
    filename = args.filename
    dataset_folder = 'dataset/train_data'

    files = [dataset_folder+'/'+filename+'_data.fasta']

    representations_dict = {
        1: ['NAC','CKSNAP','Subsequence','ASDC','Mismatch'],
        2: ['Z_curve_9bit', 'Z_curve_12bit','Z_curve_36bit','Z_curve_48bit','Z_curve_144bit'],
        3: ['PseEIIP','DAC','DCC','DACC','TAC','TCC','TACC', 'Moran','Geary','NMBroto','DPCP','TPCP'],
        4: ['MMI','PseDNC','PseKNC','PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC'],
        5: ['Kmer', 'RCKmer'],
        6: ['GC']
    }

    idx = args.representation_index

    logger.info('Working Path: {}'.format(pPath))

    representations = representations_dict[idx]

    logger.info('Representations: {}'.format(representations))
          
    models = ['RandomForest','SVM','LogisticRegression','NaiveBayes','DecisionTree']
    data_folds = ['StratifiedGroupKFold']
    n_folds = [5]

    input_params = []
    list_train_params = []

    for elements in itertools.product(files, representations, n_folds) :
        input_params.append(elements)

        (file, representation, n_fold) = elements

        # copy parameters for each descriptor
        if representation in para_dict:
            for key in para_dict[representation]:
                desc_default_para[key] = para_dict[representation][key]
        
        train_params = {
            'filename': file,
            'models': models,
            'representation': representation,
            'data_folds': data_folds,
            'n_fold': n_fold,
            'representation_params': desc_default_para,
            'genus_level': False
        }

        list_train_params.append(train_params)

    if len(list_train_params)>1 and args.n_worker > 1:
        n_worker = multiprocessing.cpu_count()
        n_worker = min(args.n_worker,n_worker)

        with Pool(n_worker) as p:
            results = p.map(crossval, tqdm(list_train_params))
    else:
        results = [crossval(list_train_params[0])]

    logger.info('Done Cross Validation')

    cv_results = {}
    list_models = {}

    for res in results:
        cv_result, model = res
        cv_results.update(cv_result)
        list_models.update(model)

    ## save model if necessary
    # model_path = os.path.join(pPath,'models', filename+'_'+str(idx))

    # if not os.path.exists(model_path):
    #     logger.info('Creating Models Folder')
    #     os.mkdir(model_path)
    
    # for model_config in tqdm(list_models, desc='Saving Models'):
    #     #print(model_config)
    #     model = list_models[model_config]
    #     model_filename = '_'.join(model_config)
    #     destination = os.path.join(model_path, model_filename+'.pkl')

    #     # save
    #     with open(destination,'wb') as f:
    #         pickle.dump(model,f)

    cv_results_df = multiindex_dict_to_df(cv_results)

    cv_path = os.path.join(pPath,'outputs/crossval/{}'.format(filename))

    if not os.path.exists(cv_path):
        logger.info('Creating Cross Validation Folder')
        os.makedirs(cv_path)

    logger.info('Saving Output')
    cv_file = os.path.join(cv_path,"{}_crossval_{}.xlsx".format(filename,idx))
    cv_results_df.to_excel(cv_file)
    logger.info('Cross Validation Path: {}'.format(cv_file))

    logger.info('Done')

if __name__=="__main__":
    start_time = time.time()
    logger.info('--- Start ---')
    main()
    finish_time = time.time()
    logger.info('--- Finish ---')
    logger.info(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))