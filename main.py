'''

Created on Sep,2022

'''

from xmlrpc.client import boolean
from utils.trainer import Trainer
from utils.feature_space import Feature_Extract
from algorithm import adbetwoa, csv_read

import argparse
import os
import numpy as np

def params():

      parser = argparse.ArgumentParser(description = 'Training model with Traditional Features')
      
      # required input args
      parser.add_argument('-num_class','--number_classes',type=int, 
                        default = 2, 
                        help = 'Class distribution of data')
      parser.add_argument('-train','--train_path',type=str, 
                        default = 'data/train/', 
                        help = 'Path to train data')
      parser.add_argument('-validation','--validation_path',type=str, 
                        default = 'data/val/', 
                        help = 'Path to validation data')                    
      parser.add_argument('-test','--test_path',type=str, 
                        default = 'data/test/', 
                        help = 'Path to test data')
      parser.add_argument('-epoch','--epoch',type=int, 
                        default = 50, 
                        help = 'epoch number')
      parser.add_argument('-lr','--lrate',type=float, 
                        default = 1e-2, 
                        help = 'learning rate')
      parser.add_argument('-batch_tr','--batch_size_training',type=int, 
                        default = 30, 
                        help = 'Set batch size for training')
      parser.add_argument('-batch_val','--batch_size_validation',type=int, 
                        default = 30, 
                        help = 'Set batch size for validation')
      parser.add_argument('-load','--load_model_wts',type=boolean, 
                        default = False, 
                        help = 'True: if model wts are required to load')
      
      # algorithm hyperparameters
      parser.add_argument('-pop_size','--population_size',type=int, 
                        default = 50, 
                        help = 'set population size')
      parser.add_argument('-pop_mating','--population_for_mating',type=int, 
                        default = 25, 
                        help = 'Set population for mating')
      parser.add_argument('-num_mut','--number_mutations',type=int, 
                        default = 50, 
                        help = 'Set number of mutations')
      parser.add_argument('-epoch_algo','--wpoch_for_algo',type=int, 
                        default = False, 
                        help = 'Set the epoch count for algorithm')

      args = parser.parse_args()

      return args



if __name__ == '__main__':
   
      args = params()  
    
      # training
      ob = Trainer(args)
      train_load, val_load, test_load = ob.train()

      # feature extraction
      f_ext = Feature_Extract(args.num_class, train_load, val_load, test_load)
      root = os.getcwd()
      f_ext.extract(root)

      # run algorithm
      features_tuple = csv_read.file_read(args, root)
      features_train = np.concatenate((features_tuple[0], features_tuple[1]), axis=0)
      features_test = features_tuple[2]

      w_obj = adbetwoa.AdBetWOA(args, features_train, features_test)
      solution = w_obj.WOA()

      # test data final metrics
      w_obj.test(solution)
