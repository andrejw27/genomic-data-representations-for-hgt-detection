# genomic-data-representations-for-hgt-detection

**Abstract**:

Horizontal gene transfer (HGT) accelerates the spread of antimicrobial resistance (AMR) via mobile genetic elements such as plasmids, phages, and transposons, allowing pathogens to acquire resistance genes across species. This process drives the evolution of multidrug-resistant `superbugs' in clinical settings.
Detection of HGT is critical to mitigating AMR, but traditional methods based on sequence assembly or comparative genomics lack resolution for complex transfer events. Machine learning (ML) offers improved detection, but several studies in other domains have demonstrated the influence of data representations on ML tasks. There is, however, no clear recommendation on the best data representation for HGT detection.
The study evaluated 44 genomic data representations using five ML models across four data sets.
We demonstrated that ML performance is highly dependent on the chosen genomic data representation.
The *RCKmer* representation paired with a support vector machine (SVM) was found to be optimal (F1: 0.959; MCC: 0.908), outperforming other approaches. Moreover, models trained on multi-species data sets showed better generalizability.
We found that genomic surveillance benefits from task-specific genome data representations. This work provides state-of-the-art, fine-tuned models for identifying and annotating resistance-associated genomes, laying the groundwork for computational approaches to combat AMR.

---

**Initial steps:**

1. Clone this repo:

```
git clone git@github.com:andrejw27/genomic-data-representations-for-hgt-detection.git
```

2. `cd` into the root directory (`genomic-data-representations-for-hgt-detection/`)
3. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html): `conda/install.sh`
4. Create conda environment: `sh conda/create_env.sh`
5. Activate conda environment: `conda activate genomic-data-rep`
6. **Optional**: Remove conda environment (if necessary): `sh conda/remove_env.sh`

---


**- Cross-validation**:

run `./cross_val.py` which requires 3 arguments:

* **representation_index**: to select which data representations to be used in cross-validation
* **n_worker**: number of workers to execute the script for parallel processing
* **filename**: filename of the data set (benbow, islandpick, gicluster, rvm)

**Steps**:

* unzip train data files in `dataset/train_data` before running the code below
* change the value for `--filename` to either `benbow, islandpick, gicluster, or rvm` to run the cross-validation on each data set
* change the value for `--representation-index` to a number from `1 to 5` as the data representations are split into 5 lists.

```
python cross_val.py --representation-index 1 --n-worker 5 --filename "benbow"
```

**- Hyperparameter tuning**:

run  `./hyperparameter_tuning.py` which requires 2 arguments, namely filename and n_worker

the list of data representations and machine learning models for each data set is provided in the script

the search space for the hyperparameters is given in file `utils/hpo.py` in `task_hyperparameter_tuning` function

**Steps**:

* change the value for `--filename` to either `benbow, islandpick, gicluster, or rvm` to run the hyperparameter tuning on each data set

```
python hyperparameter_tuning.py --n-worker 5 --filename "benbow"
```

**- Predict genomic islands (GIs):**

run `./predictGI.py` that takes 2 arguments:

* **genomes_path**: path to the folder of genomes in fasta files
* **output_dest**: path to store the predictions which will be stores in outputs

**Steps:**

* unzip the model in `utils/models` before running the code below
* change value for `--genomes-path` to any folder of your genomes with fasta format

```
python predictGI.py --genomes-path "dataset/genomes/benbow_test" --output-dest "outputs/predictions"
```

**- Evaluate the baselines**:

run `./evaluation.py` to measure performance of baselines on either benbow test or literature evaluation data set

**Steps**:

* change the value for  `--result-type` to either `test or literature` to calculate the evaluation results on benbow test or literature data set, respectively

```
python evaluation.py --result-type "literature"
```

**- Transform data sets into different data representations:**

run `./transform_data.py` to convert data sets into different representations. This is necessary to calculate the correlations between data representations. Use `adjusted_rv.R` script to calculate the correlations.

**Steps**:

* change the value for `--filename` to either `benbow, islandpick, gicluster, or rvm` to transform each data set

```
python transform_data.py --filename "benbow"
```

---

To reproduce tables and figures in the manuscript, we provide jupyter notebooks:

* **prepare_dataset.ipynb**: prepare train and test data sets
* **read_results.ipynb**: read experimental results, convert them into tables, and visualize them
* **data_representation_correlation.ipynb**: visualize correlations between different data representations
* **boundary_prediction.ipynb**: contains steps to predict boundary of GIs
