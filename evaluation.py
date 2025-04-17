import time
import argparse
import os, sys
import numpy as np 
import pandas as pd 
import json 
import datetime

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from utils.Evaluations import Evaluations
from utils.util import get_organism_info

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-type", type=str, default='test')
    args = parser.parse_args()
    return args

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

def main():
    args = get_args()
    
    predictors_folder = "outputs/literature_predictions"
    predictors = ["alien_hunter", "islander", "islandpath_dimob", "islandviewer", "sigi_hmm", "islandpick"]
    model_dict = {}
    result_type = args.result_type #test or literature evaluation

    if result_type == 'test':
        benbow_file = "fine_tuned_model_test"
        treasure_island = "treasure_island_test"
    elif result_type == 'literature':
        benbow_file = "fine_tuned_model_literature"
        treasure_island = "treasure_island_literature"

    predictors += [treasure_island,benbow_file]

    for predictor in predictors:
        predictor_file = os.path.join(predictors_folder, predictor)
        predictor_file = predictor_file+'.xlsx'
        predictor_df = pd.read_excel(predictor_file)

        model_dict.update({predictor:get_organism_info(predictor_df)}) 

    #read ground truth data (GI_literature_set_table, GI_negative_set_table, positive_test_table_gc, negative_test_table_gc)
    if result_type == 'literature':
        pos_table = pd.read_excel("outputs/literature_reference/GI_literature_set_table.xlsx")
        neg_table = pd.read_excel("outputs/literature_reference/GI_negative_set_table.xlsx")
    elif result_type == 'test':
        pos_table = pd.read_excel("outputs/literature_reference/positive_test_table_gc.xlsx")
        neg_table = pd.read_excel("outputs/literature_reference/negative_test_table_gc.xlsx")

    organism_pos_test_dict = get_organism_info(pos_table)
    organism_neg_test_dict = get_organism_info(neg_table)

    total_orgs = organism_pos_test_dict.keys()

    eval = Evaluations()

    print("evaluation of {} data".format(result_type))
    eval_results = eval.evaluations_main_104(total_orgs, 
                                            model_dict, 
                                            organism_pos_test_dict, 
                                            organism_neg_test_dict, 
                                            result_type, 
                                            False)
                
    json_obj = json.dumps(eval_results, indent=1, default=myconverter)
    json_file = "outputs/evaluation/evaluation_result_{}_fine_tuned_model.json".format(result_type)

    with open(json_file, 'w') as file:
        json.dump(json_obj, file, indent=4)

if __name__=="__main__":
    start_time = time.time()
    print('--- Start evaluating baselines---')
    main()
    finish_time = time.time()
    print('--- Finish ---')
    print(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))