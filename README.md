# Adaptive $\beta$-Hill Climbing Aided Whale Optimization Algorithm (AdBet-WOA)
- This repository contains the official implementation of our paper titled " Deep Feature Selection using Adaptive $\beta$-Hill Climbing aided Whale Optimization Algorithm for Lung and Colon Cancer Detection" submitted to Biomedical Signal Processing and Control, Elsevier.

## Overview
- AdBet-WOA is a hybrid meta-heuristic optimization algorithm using the Adaptive $\beta$-Hill Climbing local search property as the exploitation agent in the popular Whale Optimization Algorithm.
- The overall flowchart of **AdBet-WOA** and its Pseudo-algorithm are given as follows : <p align="center">  
![FlowAlgo](https://user-images.githubusercontent.com/84792746/191527947-daf67768-2ddb-4364-97c6-19f7008a348d.jpg)
</p>

- The end-to-end implementation of our proposed algorithm on the LC25000 dataset is shown as follows : <p align="center">  
![Framework](https://user-images.githubusercontent.com/84792746/191545040-2079726b-576e-4478-8a57-8c21e57ba4ed.jpg)
</p>

## Dependencies 
    directory> pip install -r requirements.txt
## Arguments
    directory\AltWOA> python main.py -h
    usage: main.py [-h] [--number_classes NUMBER_CLASSES] [--train_path TRAIN_PATH] [--validation_path VALIDATION_PATH] [--test_path TEST_PATH] [--epoch EPOCH]
                   [--lrate LRATE] [--batch_size_training BATCH_SIZE_TRAINING] [--batch_size_validation BATCH_SIZE_VALIDATION] [--load_model_wts LOAD_MODEL_WTS] [-- population_size POPULATION_SIZE] [--number_mutations NUMBER_MUTATIONS] [--epoch_for_algo EPOCH_FOR_ALGO]

    optional arguments:
      -h, --help            show this help message and exit
      --number_classes NUMBER_CLASSES   Class distribution of data
      --train_path TRAIN_PATH
                            Path to train data
      --validation_path VALIDATION_PATH
                            Path to validation data
      --test_path TEST_PATH
                            Path to test data
      --epoch EPOCH   Number of iterations
      --lrate LRATE   Learning rate
      --batch_size_training BATCH_SIZE_TRAINING
                            Send batch size for training
      --batch_size_validation BATCH_SIZE_VALIDATION
                            Send batch size for validation and testing
      --load_model_wts LOAD_MODEL_WTS
                            True: if model weights are required to be loaded
      --population_size POPULATION_SIZE
                            Set population size
      --population_for_mating POPULATION_FOR_MATING
                            Set population for mating
      --number_mutations NUMBER_MUTATIONS
                            Number of mutations
      --epoch_for_algo EPOCH_FOR_ALGO
                            Set the epoch count for algorithm
                            
## Code Execution
    directory/main.py -num_class {} -train {} -validation {} -test {} -epoch {} -lr {} -batch_tr {} -batch_val {} -load {} -pop_size {} -pop_mating {} -num_mut {} -epoch_algo {}
## Description
    directory
         |
         +--> utils
         |      |
         |      +--> data.py              # The python file which performs the dataloader initialization for train, val and test
         |      +--> feature_space.py     # The python file used to store the generated features in csv files
         |      +--> trainer.py           # The python file for training the dataset with our complete framework
         |
         +--> algorithm
         |      |
         |      +--> adbetwoa.py          # The python file which contains our entire proposed algorithm
         |      +--> algo_utils.py        # The python file containing other necessary peripheral functions related to algorithm
         |      +--> csv_read.py          # The python file for reading the csv files storing generated features
         |      +--> transf_functions.py  # The python file with other peripheral methods
         |
         +--> dos_class
         |      |
         |      +--> resnet_encoder.py    # The python file which generated the features for two-class classification
         |
         +--> tres_class
         |      |
         |      +--> effnet_encoder.py    # The python file which generated the features for three-class classification
         |
         +--> cinco_class
         |      |
         |      +--> effnet_encoder.py    # The python file which generated the features for five-class classification
         |
         +--> model_weights               # Folder to save the wiights of the best performing model 
         |
         |
         |                                  
         +--> main.py                    # The main function which reads the arguments dependiong on user input and performs the whole 
                                           end-to-end operation in a chronological order
