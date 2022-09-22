'''
Created on Sep,2022
'''

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness



def load(self):
    
    img_size = 224 if self.args.num_class == 2 else 380
        
    #Augmentation
    train_transform = transforms.Compose([  transforms.Resize((img_size,img_size)),
                                            transforms.RandomHorizontalFlip(p=0.7),
                                            transforms.RandomRotation(degrees=(45,90)),
                                            transforms.RandomVerticalFlip(p=0.7),
                                            transforms.ColorJitter(brightness=0.5),
                                            RandomAdjustSharpness(sharpness_factor=5),
                                            transforms.RandomAutocontrast(),
                                            transforms.ToTensor()
                                        ])

    val_transform = transforms.Compose([    transforms.Resize((img_size,img_size)),
                                            transforms.ToTensor()
                                      ])
                        
    test_transform = transforms.Compose([   transforms.Resize((img_size,img_size)),
                                            transforms.ToTensor()
                                       ])


    train_ds = ImageFolder(self.args.train, transform=train_transform)
    val_ds = ImageFolder(self.args.validation, transform=val_transform)
    test_ds = ImageFolder(self.args.test, transform=test_transform)
    
    
    train_load = DataLoader(dataset=train_ds, batch_size=self.args.batch_tr, shuffle=False, num_workers=2, drop_last=False)
    valid_load = DataLoader(dataset=val_ds, batch_size=self.args.batch_val, shuffle=False, num_workers=2, drop_last=False)
    test_load = DataLoader(dataset=test_ds, batch_size=self.args.batch_val, shuffle=False, num_workers=2, drop_last=False)
    
    
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    return  train_ds, val_ds, test_ds, train_load, valid_load, test_load, device
