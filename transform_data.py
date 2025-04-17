import argparse
import os
import time
from tqdm import tqdm 
import pickle
from utils.util import get_representations
from utils.Parameters import Parameters

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="benbow")
    args = parser.parse_args()
    return args

def main():
    parameters = Parameters()

    desc_default_para = parameters.DESC_DEFAULT_PARA

    args = get_args()

    filename = args.filename
    file = "dataset/train_data/{}_data.fasta".format(filename)

    representations = ['NAC','CKSNAP','Subsequence','ASDC', 
                'Z_curve_9bit', 'Z_curve_12bit','Z_curve_36bit','Z_curve_48bit','Z_curve_144bit',
                'PseEIIP','DAC','DCC','DACC','TAC','TCC','TACC','Moran','Geary','NMBroto','DPCP','TPCP',
                'MMI','PseDNC','PseKNC','PCPseDNC','PCPseTNC','SCPseDNC','SCPseTNC',
                'Mismatch', 'Kmer', 'RCKmer','GC']

    # for testing the script
    # representations =['NAC','CKSNAP']

    converted_data = {}

    for representation in tqdm(representations):
        if representation in ['Kmer', 'RCKmer']:
            for k in range(1,8):
                desc_default_para.update({'kmer':k})

                key = '{}-{}'.format(representation,k)
                X,y,group = get_representations(representation, file, desc_default_para)
                converted_data.update({key:X})
                converted_data.update({'y_'+key:y,
                                        'group_'+key:group})
        else:
            desc_default_para.update({'kmer':3})
            X,y,group = get_representations(representation, file, desc_default_para)
            converted_data.update({representation:X})
            converted_data.update({'y_'+representation:y,
                                    'group_'+representation:group})
    
    destination_path = 'dataset/data_representation'
    if not os.path.exists(destination_path):
        print('Creating Folder to store encoded data')
        os.makedirs(destination_path)

    # Dump dictionary to a file in binary format using pickle
    with open(os.path.join(destination_path,'transformed_{}.pkl'.format(filename)), 'wb') as file:
        pickle.dump(converted_data, file)

if __name__=="__main__":
    start_time = time.time()
    print('--- Start converting data into different representations ---')
    main()
    finish_time = time.time()
    print('--- Finish converting data ---')
    print(' --- Process took {:.3f} seconds ---'.format(finish_time-start_time))