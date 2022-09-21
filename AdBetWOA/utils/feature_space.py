'''
Created Sep,2022
'''



from dos_class import resnet_encoder as res
from tres_class import effnet_encoder as eff_3
from cinco_class import effnet_encoder as eff_5

import numpy as np
import torch
import csv
import os


class Feature_Extract():
    def __init__(self, num_class, train_load, val_load, test_load):

        assert num_class<6 and num_class>1 and num_class!=4 

        self.num_class = num_class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_load = train_load
        self.val_load = val_load
        self.test_load = test_load

        self.columns = 1793 if self.num_class > 2 else 513


    def generator(model, loaders_list, device, columns):

        features_tuple = ()
        for loader in loaders_list:
            
            features = np.zeros((1,columns))
            labels = np.array([])
        
            for images,labels in loader:
                images = images.to(device)
                featr,_ = model(images)
                features = np.append(featr, featr.cpu().detach().numpy(), axis=0)
                labels = np.append(labels,labels.detach().numpy(),axis=0)

            features_tuple = features_tuple + (features, labels)

        return features_tuple


    def extract(self, root):

        if self.num_class<3:
            model = res.Model(self.num_class)
            
            path = "model_weights/resnet_class.pth"
            assert os.path.exists(path) 

            state=torch.load(path)
            model.load_state_dict(state['model_state'])
        else:
            if self.num_class>4:
                model = eff_5.Model(self.num_class)

                path = "model_weights/effnet_5_class.pth"
                assert os.path.exists(path) 

                state=torch.load(path)
                model.load_state_dict(state['model_state'])
            else:
                model = eff_3.Model(self.num_class)

                path = "model_weights/effnet_3_class.pth"
                assert os.path.exists(path) 

                state=torch.load(path)
                model.load_state_dict(state['model_state'])

        # load weights
        

        loaders_list = [self.train_load, self.val_load, self.test_load]
        features_tuple = generator(model, loaders_list, self.device, self.columns) 

        # features to be stored in csv format
        
        p_fix = iter(['train', 'validation', 'test'])
        for features in features_tuple:
            token = next(p_fix)

            with open(root+'/{}_features.csv'.format(token), 'a+',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(range(self.columns))
                for iter in range(features[1].shape[0]-1):
                    writer.writerow(np.append(features[0][iter+1],features[1][iter+1]))