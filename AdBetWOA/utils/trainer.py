'''

Created on Sep,2022

'''

from dos_class import resnet_encoder as res
from tres_class import effnet_encoder as eff_3
from cinco_class import effnet_encoder as eff_5
from data import load


import torch
import copy
import torch.nn as nn
import os




class Trainer():

    def __init__(self, args):
        self.args = args

    
    def eval(model, criterion, val_load, device, batch_val):
        # validation
        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        for images,labels in val_load:

            images = images.to(device)
            labels = labels.to(device)
            _,output = model(images)
            _,pred = torch.max(output,1)
            loss = criterion(output,labels)
            running_val_loss += loss.item()*batch_val
            running_val_acc += torch.sum(pred==labels)
        
        return running_val_loss, running_val_acc
    
    
    def train(self):
        # data loaders
        train_ds, val_ds, _, train_load, val_load, test_load, device = load(self)

        # model initialization
        if self.args.num_class == 2:
            model = res.Model(self.args.num_class)
            if self.args.load:
                path = "model_weights/resnet.pth"
                assert os.path.exists(path) 

                state=torch.load(path)
                model.load_state_dict(state['model_state'])

        if self.args.num_class == 3:
            model = eff_3.Model(self.args.num_class)
            if self.args.load:
                path = "model_weights/effnet_3_class.pth"
                assert os.path.exists(path) 

                state=torch.load(path)
                model.load_state_dict(state['model_state'])

        if self.args.num_class == 5:
            model = eff_5.Model(self.args.num_class)
            if self.args.load:
                path = "model_weights/effnet_5_class.pth"
                assert os.path.exists(path) 

                state=torch.load(path)
                model.load_state_dict(state['model_state'])

        else:
            exit()

        # loss function
        criterion = nn.CrossEntropyLoss()
        criterion=criterion.to(device)
       
        # optimizer
        optim = torch.optim.Adam(model.parameters(), self.args.lr)
        optim = optim.to(device)
        if self.args.load:
            optim.load_state_dict(state['optimizer_state'])



        best_acc=0.85 
        train_loss_list = []
        val_loss_list = []
        best_model_wts = copy.deepcopy(model.state_dict())
        for epoch in range(self.args.epoch):
            
            model.train()
            running_train_loss = 0.0
            running_train_acc = 0.0
            for images,labels in train_load:

                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    _,output = model(images)
                    _,pred = torch.max(output,1)
                    loss = criterion(output,labels)
                    loss.backward()
                    optim.step()
            optim.zero_grad()

            running_train_loss += loss.item()*self.args.batch_tr
            running_train_acc += torch.sum(pred==labels)


            epoch_train_loss = running_train_loss/len(train_ds)
            epoch_train_acc = running_train_acc.double()/len(train_ds)

            print("Epoch: {}".format(epoch+1))
            print('-'*10)
            print('Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch_train_loss,epoch_train_acc))
            
            
            running_val_loss, running_val_acc = eval(model, criterion, val_load, device, self.args.batch_val)
            
            epoch_val_loss = running_val_loss/len(val_ds)
            epoch_val_acc = running_val_acc.double()/len(val_ds)

            print('Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch_val_loss,epoch_val_acc))
            print('\n')
            
            
            if best_acc < epoch_val_acc:
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state={
                        "model_state":model.state_dict(),
                        "optimizer_state":optim.state_dict()
                      }
                torch.save(state,path)
            
            train_loss_list = train_loss_list + [epoch_train_loss]
            val_loss_list = val_loss_list + [epoch_val_loss]

        #model = model.load_state_dict(best_model_wts)

        print("The model with the best performance has an accuracy of :{:.4f}".format(best_acc))

        return train_load, val_load, test_load