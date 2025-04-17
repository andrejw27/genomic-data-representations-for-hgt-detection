import time
from tqdm import tqdm
import argparse
import os
import pandas as pd 
from tqdm import tqdm
from utils.Predictor import Predictor

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes-path", type=str, default="dataset/genomes/benbow_test")
    parser.add_argument("--model", type=str, default="fine_tuned_model.pkl" )
    parser.add_argument("--output-dest", type=str)
    args = parser.parse_args()
    return args

def main():

    args = get_args()

    #folder containing genomes of interest 
    # ("dataset/genomes/benbow_test","dataset/genomes/literature" )
    genome_path = args.genomes_path
    files = [f for f in os.listdir(genome_path) if os.path.isfile(os.path.join(genome_path, f))]

    # path to load the saved model: utils/models/
    model = args.model 

    # path to save the predictions
    output_path = "outputs"
    output_path = os.path.join(output_path, genome_path.split('/')[-1], model.split('.')[0])

    print(output_path)

    # parameters for data representation, specify the representation according to the trained model
    data_rep_params = {
        'representation':'RCKmer',
        'representation_params':{'kmer':7}
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #run the following code only if the predictions do not exist for the specified genomes and model
    for file in tqdm(files):
        #accept fasta file only
        if file.endswith('.fasta'):
            filename = os.path.join(genome_path,file)

            #create a folder for each genome
            output_dest = os.path.join(output_path, file.split('.')[0])
            if not os.path.exists(output_dest):
                os.mkdir(output_dest)

                #initialize the predicor
                seq = Predictor(filename, output_file_path=output_dest, model_file=model)
                
                #update the representation according to the trained model
                params = data_rep_params

                seq.change_representation_parameters(params)

                #run the predictor
                pred = seq.predict()
                
                #save predictions to excel file
                seq.predictions_to_excel(pred)
            else:
                print("Predictions for {} already exist".format(file))
        else:
            print("The code only accepts fasta file!")

    #read predictions for each genome, then combine them into a file
    results = pd.DataFrame()

    if args.output_dest:
        results_dest = os.path.join(args.output_dest, model.split('.')[0]+'.xlsx')
    else:
        results_dest = '{}/{}.xlsx'.format('/'.join(output_path.split('/')[:-1]), model.split('.')[0])

    for dir in os.listdir(output_path):
        child_dirs = os.path.join(output_path,dir)
        if os.path.isdir(child_dirs):
            for file in os.listdir(child_dirs):
                #if 'out' not in file:
                    res = pd.read_excel(os.path.join(child_dirs,file))
                    res = res.drop(res.columns[0], axis=1)
                    res = res.assign(Genome=dir)
                    results = pd.concat([results,res])

    results = results.rename(columns={'accession':'Accession','start':'Start','end':'End'})
    results = results[results['probability']>0.5]
    results['Accession'] = results.apply(lambda x: x['Accession'].split('|')[0],axis=1)
    results.to_excel(results_dest, index=False)
    print(results_dest)

if __name__=="__main__":
    start_time = time.time()
    print('--- Start ---')
    main()
    finish_time = time.time()
    print('--- Finish ---')
    print(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))