import numpy as np 
import pandas as pd
import random
from tqdm import tqdm
import json
import pickle
from Bio import SeqIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, matthews_corrcoef,confusion_matrix, accuracy_score, precision_score, f1_score, fbeta_score, recall_score
from . import FileProcessing,CheckAccPseParameter

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

import os, sys, re
from pathlib import Path
file_path = os.path.split(os.path.realpath(__file__))[0]
pPath = Path(file_path)
sys.path.append(pPath)

import logging 
logger = logging.getLogger('util')

#reference: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
def flatten_dict(nested_dict):
    """
    transform a nested dictionary into a normal dictionary with {key:value} format

    Parameters: 
    ----------
    nested_dict : dict, a nested dictionary

    Returns: 
    -------
    output_dict : dict, flattened dictionary
    """

    output_dict = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                output_dict[tuple(key)] = val
    else:
        output_dict[()] = nested_dict
    return output_dict
    
def nested_dict_to_df(input_dict):
    """
    transform a nested dictionary into a dataframe

    Parameters: 
    ----------
    input_dict : dict, a nested dictionary

    Returns: 
    -------
    output_df : pandas.DataFrame
    """
    flat_dict = flatten_dict(input_dict)
    output_df = pd.DataFrame.from_dict(flat_dict, orient="index")
    output_df.index = pd.MultiIndex.from_tuples(output_df.index)
    output_df = output_df.unstack(level=-1)
    output_df.columns = output_df.columns.map("{0[1]}".format)
    return output_df

########################## function to turn multiindex dictionary to dataframe ##########################
def multiindex_dict_to_df(input_dict):
    """
    transform a dictionary with tuples as keys into a dataframe

    Parameters: 
    ----------
    input_dict : dict, a multiindex dictionary, example: {(tuple):value}
    
    Returns: 
    -------
    output_df : pandas.DataFrame
    """
    output_df = pd.DataFrame.from_dict(input_dict, orient="index")
    output_df.index = pd.MultiIndex.from_tuples(output_df.index)
    output_df = output_df.unstack(level=-1)
    output_df.columns = output_df.columns.map("{0[1]}".format)
    return output_df
    
########################## function to encode data set ##########################

def get_representations(desc, filename, desc_default_para):
    """
    transform a dictionary with tuples as keys into a dataframe

    Parameters: 
    ----------
    desc : str, type of data representation
    filename : str, path to the fasta file
    desc_default_para : dict, parameters for the corresponding data representation, example: k values for kmer

    Returns: 
    -------
    X : array-like, feature matrix used for training. Each row represents a sample and each column a feature.
    y : array-like, target values (class labels) corresponding to the input samples.
    groups : array-like, groups (species) corresponding to the input samples
    """
    
    descriptor = FileProcessing.Descriptor(filename, desc_default_para)
    
    if desc in ['DAC', 'TAC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DCC', 'TCC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['DACC', 'TACC']:
        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, descriptor.sequence_type, desc_default_para)
        status = descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
    elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
        my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, descriptor.sequence_type, desc_default_para)
        cmd = 'descriptor.' + desc + '(my_property_name, my_property_value)'
        status = eval(cmd)
    else:
        cmd = 'descriptor.' + desc + '()'
        status = eval(cmd)

    X = descriptor.encoding_array[1:][:,2:].astype(float)
    y = descriptor.encoding_array[1:][:,1].astype(int)
    groups = np.array(['_'.join(label.split('_')[:2]) for label in descriptor.encoding_array[1:][:,0]])

    return X,y,groups
        

########################## function to read cross validation results ##########################

def read_results(filename, header=['dataset','model','fold','n_fold','representation']):
    """
    a function to read cross validation results

    Parameters: 
    ----------
    filename : str, file of cross validation results

    Returns: 
    -------
    cross_val_df : pandas.DataFrame
    """

    cols = pd.read_excel(filename, header=None,nrows=1).values[0]
    col_dict = {}
    eval_metrics = []

    for col in cols[len(header):]:
        if col not in col_dict.keys():
            col_dict.update({col:1})
        else:
            col_dict.update({col:col_dict[col]+1})

        eval_metrics.append(col+'_'+str(col_dict[col]))

    header.extend(eval_metrics)

    cross_val_df = pd.read_excel(filename, header=None, skiprows=1) # skip 1 row
    cross_val_df.columns = header
    cross_val_df = cross_val_df.ffill() 

    return cross_val_df

########################## function to read boundary prediction evaluation results ##########################
def read_eval_result(json_file):
    """
    a function to read boundary prediction evaluation results

    Parameters: 
    ----------
    filename : str, file of boundary prediction evaluation results

    Returns: 
    -------
    eval_df : pandas.DataFrame
    """

    with open(json_file, 'r') as file:
        json_obj = json.load(file)
        eval_result = json.loads(json_obj)

    eval_df = pd.DataFrame()

    for predictor in eval_result.keys():
        eval = dict()

        for org in eval_result[predictor].keys():
            metrics_dict = dict()

            for metric in eval_result[predictor][org]:
                metrics_dict.update(metric)

            f_2_score = 0 if (metrics_dict['Precision'] == 0 or metrics_dict['Recall'] == 0) else (1 + 2**2) * (metrics_dict['Precision'] * metrics_dict['Recall']) / (2**2 * metrics_dict['Precision'] + metrics_dict['Recall'])
            metrics_dict.update({'F-2-Score':f_2_score})
            eval.update({org:metrics_dict})
        
        eval_df_temp = nested_dict_to_df(eval).reset_index()
        eval_df_temp = eval_df_temp.assign(Predictor=predictor)
        eval_df = pd.concat([eval_df,eval_df_temp])
    return eval_df 

########################## Define custom scorer ##########################
 
def specificity_score(y_true, y_pred):
    """
    a function to calculate specificity

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    specificity : float
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def true_negative(y_true, y_pred):
    """
    a function to calculate true negative

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of true negative
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn

def true_positive(y_true, y_pred):
    """
    a function to calculate true positive

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of true positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp

def false_positive(y_true, y_pred):
    """
    a function to calculate false positive

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    tn : int, number of false positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp

def false_negative(y_true, y_pred):
    """
    a function to calculate false negative

    Parameters: 
    ----------
    y_true : array-like, ground truth
    y_pred : array_like, predictions

    Returns: 
    -------
    fn : int, number of false negative
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn

#define scoring metrics
scoring = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'Specificity': make_scorer(specificity_score),
    'F_1': 'f1',  # F-beta with beta=1 is equivalent to F1-score
    'F_beta_0.5': make_scorer(fbeta_score, beta=0.5),
    'F_beta_2': make_scorer(fbeta_score, beta=2),
    'MCC': make_scorer(matthews_corrcoef),
    'TP': make_scorer(true_positive),
    'TN': make_scorer(true_negative),
    'FP': make_scorer(false_positive),
    'FN': make_scorer(false_negative),
}


########################## function to evaluate model ##########################

def evaluate_model(train_file, test_file, train_params, **kwargs):
    """
    a function to evaluate model given train  data set and test data set

    Parameters: 
    ----------
    train_file : str, file for train data set
    test_file : str, file for test data set
    train_params : dict, parameters for training include data representation, model

    Returns: 
    -------
    model : trained sklearn estimator, the trained machine learning model.
    eval_scores : dict, evaluation results
    """
    if 'k' in train_params.keys():
        train_params['representation_params'].update({'kmer':train_params['k']})

    if train_params['representation'] in ['Kmer', 'RCKmer']:
        key = "{}-{}".format(train_params['representation'],train_params['representation_params']['kmer'])
    else:
        key = train_params['representation']

    print('representation Train Data with {}'.format(key))
    X_train,y_train,groups_train = get_representations(train_params['representation'], train_file, train_params['representation_params'])
    print('representation Test Data with {}'.format(key))
    X_test,y_test,groups_test = get_representations(train_params['representation'], test_file, train_params['representation_params'])

    groups_train = np.array([group.split('.')[0] for group in groups_train])
    groups_test = np.array([group.split('.')[0] for group in groups_test])

    # ensure training data does not contain test data
    test_ids = set(groups_test)

    selected_ids = []
    for i, g in enumerate(groups_train):
        if g not in test_ids:
            selected_ids.append(i)

    reduced_X_train = X_train[selected_ids]
    reduced_y_train = y_train[selected_ids]
    reduced_group_train = groups_train[selected_ids]

    #ensure species in train and test data sets are mutually exclusive
    assert len(set(groups_test)-set(reduced_group_train)) == len(set(groups_test))

    #define the models to be trained
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, min_samples_leaf=1, min_samples_split=2,max_features='sqrt',random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
        'SVM': SVC(random_state=42, kernel='rbf', C=2, gamma='scale', probability=True),
        'NaiveBayes': GaussianNB(),
    }

    if 'model' in kwargs.keys():
        model = kwargs['model']
    elif 'model_path' in kwargs.keys():
        try:
            print('Loading model')
            with open(kwargs['model_path'], "rb") as input_file:
                model = pickle.load(input_file)
        except Exception as e:
            print(e)
    else:
        model = models[train_params['model']]

        print('Training in progress')
        model.fit(reduced_X_train,reduced_y_train)
        print('Training is done')

    print('Testing the model')
    y_pred = model.predict(X_test)
    #y_pred_prob = clf.predict_proba(X_test)[:,1]

    eval_scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F_1': f1_score(y_test, y_pred), 
        'F_beta_0.5': fbeta_score(y_test, y_pred, beta=0.5),
        'F_beta_2': fbeta_score(y_test, y_pred, beta=2),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TP': true_positive(y_test, y_pred),
        'TN': true_negative(y_test, y_pred),
        'FP': false_positive(y_test, y_pred),
        'FN': false_negative(y_test, y_pred),
    }

    return model, eval_scores

#####################################################################

def read_file(fasta_file, label):
    """
    a function to a read fasta file and transform it to a dataframe

    Parameters: 
    ----------
    fasta_file : str, fasta file to read
    label : str, label to assign to the data ('1' or '0')

    Returns: 
    -------
    output_df : pandas.DataFrame
    """

    # Read the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")
    list_id = []
    position = ""
    # Iterate over each sequence in the FASTA file
    for seq_record in sequences:
        #print(f"ID: {seq_record.id}")
        if '..' in seq_record.id:
            position = re.search("\d+\..\d+", seq_record.id)[0]
        elif '..' in seq_record.description:
            position = re.search("\d+\..\d+", seq_record.description)[0]
        else:
            position = ""
    
        start = 0
        end = 0
        
        if position != "":
            start = position.split('..')[0]
            end = position.split('..')[1]
    
        id_ = '_'.join(seq_record.id.split('_')[:2])
        list_id.append((id_, label, int(start), int(end)))
    
    output_df = pd.DataFrame(list_id, columns = ['Accession','Label','Start','End'])
    
    return output_df

#query sequence from reference database
def query_sequence(accession_id, start=0, end=0):
    """
    a function to a query sequence from genomic database

    Parameters: 
    ----------
    accession_id : str, accession id of the genome of interes
    start : int, start position of the genome
    end : int, end position of the genome

    Returns: 
    -------
    [sequence, sequence's description]
    """
    from Bio import Entrez, SeqIO
    Entrez.email = "A.N.Other@example.com"

    try:
        if start==0 & end==0:
            handle = Entrez.efetch(db='nucleotide',
                               id=accession_id, 
                               rettype="fasta")
        else:
            handle = Entrez.efetch(db='nucleotide',
                                   id=accession_id, 
                                   rettype="fasta",
                                   seq_start=start,
                                   seq_stop=end)
            
        seq = SeqIO.read(handle, "fasta")
        handle.close()
    
        return [str(seq.seq), seq.description]
    except Exception as e:
        print(f"An error occurred: {e}")
        return ["retrieval failed", "retrieval failed"]
    
#write dataframe to fasta file

def df_to_fasta(data, dna_only=True, query_db=False, **kwargs):
    """
    a lambda function to process each row of a dataframe and transform it to a fasta file

    Parameters: 
    ----------
    data : a row of a pandas.DataFrame with columns ['Accession','Start','End','Label']
    or ['Accession','Start','End','Label','Sequence','Description']
    dna_only : bool, whether to return dna sequence only or include IUPAC code
    query_db : bool, whether or not query

    Returns: 
    -------
    result : str, a record in a fasta file ">accession|label|description\nsequence"
    """

    accession = data['Accession']
    label = data['Label']
    result = ""

    if query_db:
        seq_start = data['Start']
        seq_end = data['End']
        query = query_sequence(accession, seq_start, seq_end)
        sequence, description = query[0], query[1]
    else:
        sequence = data['Sequence']
        description = data['Description']

    try:
        if label.isdigit():
            if int(label) == 1:
                identifier = "GI_{}".format(int(data['rank']))
                label = 1
            else:
                identifier = "Non_GI_{}".format(int(data['rank']))
                label = 0
    
        else:
            if label != "negative":
                identifier = "GI_{}".format(int(data['rank']))
                label = 1
            else:
                identifier = "Non_GI_{}".format(int(data['rank']))
                label = 0
    except:
        identifier = ""
        if label.isdigit():
            if int(label) == 1:
                label = 1
            else:
                label = 0
    
        else:
            if label != "negative":
                label = 1
            else:
                label = 0
    

    if dna_only:
        if len(set(sequence) - set({'A','T','G','C'})) == 0:
            if identifier != "":
                result = ">{}_{}|{}|{}\n{}".format(accession,identifier,label,description,sequence)
            else:
                result = ">{}|{}|{}\n{}".format(accession,label,description,sequence)
        else:
            pass
    else:
        if len(set(sequence) - set({'A','T','G','C'})) != 0:
            sequence = replace_iupac_with_nucleotide(sequence)
            
        if identifier != "":
                result = ">{}_{}|{}|{}\n{}".format(accession,identifier,label,description,sequence)
        else:
            result = ">{}|{}|{}\n{}".format(accession,label,description,sequence)
    
    try:
        if kwargs['write_file']:
            destination_file = kwargs['filename']
            # Writing sequences to a FASTA file
            with open(destination_file, 'a') as f:
                f.write(result + '\n')
    except Exception as e:
        return result

def check_sequence_type(fasta_list):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return: str, type of the given sequence
        """
        if type(fasta_list) == str:
            fasta_list = [fasta_list]
        
        tmp_fasta_list = []
        if len(fasta_list) < 100:
            tmp_fasta_list = fasta_list
        else:
            random_index = random.sample(range(0, len(fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in fasta_list:
                line = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line)
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in fasta_list:
                line = re.sub('U', 'T', line)
            return 'RNA'
        else:
            return 'Unknown'
            
def fasta_to_df(file, dna_only=True):
    """
    a function to read fasta file and convert it into a pandas.DataFrame

    Parameters: 
    ----------
    file : str, fasta file
    dna_only : bool, whether or not to return only records of dna sequences 

    Returns: 
    -------
    output_df : pandas.DataFrame, columns ['Accession','Sequence','Start','End','Description','Label']
    """

    sequences = SeqIO.parse(file, "fasta")
    
    data = []
    
    # Iterate over each sequence in the FASTA file
    for seq_record in sequences:
        desc = seq_record.description
        accession = '_'.join(desc.split('|')[0].split('_')[:2])
        #accession = desc.split('|')[0]
        label = desc.split('|')[1]
        position = re.search("\d+\-\d+", seq_record.id)[0]
        start = int(position.split('-')[0])
        end = int(position.split('-')[1])
        sequence = str(seq_record.seq)
    
        if dna_only:
            if len(set(sequence) - set({'A','T','G','C'})) == 0:
                data.append((accession, str(seq_record.seq), start, end, desc.split('|')[-1], label))
        else:
            data.append((accession, str(seq_record.seq), start, end, desc.split('|')[-1], label))
            
    output_df = pd.DataFrame(data, columns = ['Accession','Sequence','Start','End','Description','Label'])

    return output_df

def replace_iupac_with_nucleotide(sequence):
    """
    a function to replace IUPAC codes in a dna sequence with DNA letters

    Parameters: 
    ----------
    sequence : str, dna sequence 

    Returns: 
    -------
    str, original sequence with replaced IUPAC codes
    """

    original_sequence = []

    # Define the mapping from IUPAC codes to possible nucleotides
    iupac_map = {
        'A': ['A'],       # Adenine
        'C': ['C'],       # Cytosine
        'G': ['G'],       # Guanine
        'T': ['T'],       # Thymine
        'R': ['A', 'G'],  # Purine
        'Y': ['C', 'T'],  # Pyrimidine
        'S': ['G', 'C'],  # Strong
        'W': ['A', 'T'],  # Weak
        'K': ['G', 'T'],  # Keto
        'M': ['A', 'C'],  # Amino
        'B': ['C', 'G', 'T'],  # Not A
        'D': ['A', 'G', 'T'],  # Not C
        'H': ['A', 'C', 'T'],  # Not G
        'V': ['A', 'C', 'G'],  # Not T
        'N': ['A', 'C', 'G', 'T']  # Any nucleotide
    }
    
    for char in sequence:
        if char in iupac_map:
            # Choose one of the possible nucleotides at random
            chosen_nucleotide = random.choice(iupac_map[char])
            original_sequence.append(chosen_nucleotide)
        else:
            original_sequence.append(char)  # For standard nucleotides A, C, G, T
    return ''.join(original_sequence)

def get_organism_info(data_set):
    """
    a function to get organism info from a pandas.DataFrame

    Parameters: 
    ----------
    data_set : pandas.DataFrame, column = ['Accesion','Start','End']

    Returns: 
    -------
    organism : dict, dictionary of organisms with format {Accession:[Accession, Start, End]}
    """
    organism = {}
    for i, data in data_set.iterrows():
        acc = data.Accession
        if acc in organism.keys():
            organism[acc].append([acc, data.Start, data.End])
        else:
            organism[acc] = [[acc, data.Start, data.End]]
    return organism