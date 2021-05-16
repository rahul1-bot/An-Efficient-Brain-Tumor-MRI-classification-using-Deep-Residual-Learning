# -*- coding: utf-8 -*-
# Created on: 14/5/21 

# Requied Python version >= Python 3.9 
# if Python 3.5 <= Python version < Python 3.9:
#       from __future__ import annotations
#       from typing import *

from __future__ import annotations
__authors__ = [
    'Ramhul_Sawhney', 'Nimkhil_Kumar_Pradhan', 'Amitwa_Kumar_Khan'
]


# Spacing: b/t each methods: 2 lines
#       : b/t each classes: 4 lines
#      : inside methods: 0 lines 
"""
Project Abstract:
A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous 
System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is 
approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. 
Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to 
detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. 
A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.
Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual 
classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using ConvolutionNeural Network (CNN), 
Artificial Neural Network (ANN), and TransferLearning (TL) would be helpful to doctors all around the world.


Context:
Brain Tumors are complex. There are a lot of abnormalities in the sizes and location of the brain tumor(s). This makes it really difficult for complete 
understanding of the nature of the tumor. Also, a professional Neurosurgeon is required for MRI analysis. Often times in developing countries the lack of
skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’. So an automated system on Cloud 
can solve this problem.


Project Control-Flow: 1) class Dataset
                            :__init__
                            :__len__
                            :__getitem__
                            :__repr__
                            :@ normalize
                            :@ get_data
                            
                        2) Class DataAnlysis
                            :__repr__
                            :@ One Bacts pics Visualization
                            :@ bar plot between all the tumors
                            :@ Images after applying 7-8 different convolutional filters
                            :@ ...
                        
                        3) Class DataPreprocess
                            :__repr__
                            :@ Check.. if needed, Resize all the images
                            :@ RandomCrop of Images
                            :@ RandomHorizontal and Vertical Flip of the image
                            :@ Normalize and denormalize images to Tensors
                            :@ Arguments all the images to deal with the issue of Overfitting

                        4) Class GPU_Acceleartion
                            :__init__
                            :__len__
                            :__iter__
                            :__repr__
                            :@ hardware_accelearation
                            :@ set_GPU
                            :@ load_GPU
                            :@ is_working_GPU

                        5) Class IMAGE_CLASSIFIER_BASE
                            :@ accuracy
                            :@ training_step
                            :@ Validation_step
                            :@ validation_epoch_end
                            :@ epoch_end

                        6) Class HyperParams
                            : >Layers
                            : >Neurons Engineering
                            : >Activation Functions

                        7) Class CNNModel
                            :__init__
                            :@ forward  # Neural Nets Forward Pass
                            *:@ Backward # Neural Nets BackPropogation 
                        

                        8) Class Train_Test_Fit
                            :__repr__
                            :@ evaluate
                            :@ get_learning_rate
                            :@ fit_one_cycle
                            :@ fit_model -> _text

                        9) Class EvaluateCNN
                            :__repr__
                            :@ plot_1: Accuracy vs No. of Epochs
                            :@ plot_2: Loss vs No. of Epochs
                            :@ plot_3: Learning Rate Vs Batch no
                            :@ plot_4: Model Report
                            :@ plot_5: Model Confusion Matrix

                      10) Class SaveCNN := Brain_MRI_model.pt
                            :__repr__
                            :@ save_CNN
                            :@ is_save_CNN
                            :@ load_CNN
                            :@ is_loaded_CNN

""" 

# Python Imports 
import typing
__name__: typing.__name__ = '__main__'
from typing import Any, NewType, Generator, Optional, Union
import os, warnings
warnings.filterwarnings(action= 'ignore')


# Data Analysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Scripting ML 
from sklearn.metrics import confusion_matrix


# DL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



# torch typing scripts
_path =  NewType("_path", Any)
_transform = NewType("_transform", Any)
_img = NewType("_img", Any)
_criterion = NewType("_criterion", Any)
_optimizer = NewType("_optimizer", Any)
_loss = NewType("_loss", Any)
_layer = NewType("_layer", Any)
_activation = NewType("_activation", Any)
_text = NewType("_text", Any)
_plot = NewType("_plot", Any)
_loader = NewType("_loader", Any)
_recurse = NewType("_recurse", Any)


#############################################################################################################
#...DIR ...
# Testing:    
#        glioma_tumor 
#        meningioma_tumor 
#        no_tumor
#        pituitary_tumor
#
# Training: 
#        glioma_tumor 
#        meningioma_tumor 
#        no_tumor
#        pituitary_tumor
#...
class BrainMRIDataset(Dataset):
    """
        A Custom `DataLoader Iterator` Class to load the set of Images in a `batch` 
        and applies some kind of `transformation` to every set of image in a batch of `Iterators`.

        INIT Args:
            path: _path     = Path of the Image Dataset folder
            sub_path: str   = str <Training | Testing | Validation>
            
        OPTIONAL Args:
            batch_size: int = size of the batches in the Training and Testing Dataset
            img_resolution: Optional[int]   = resolution of the image to load. :Options = [8, 16, 32, 64, 128] pixels
            transform: Optional[_transform] = `Sequencial Container` of various feature engineering teachiques.
        
        EXAMPLES:
            >>> df_train: pd.DataFrame = BRAINDataset(path= .../BRAIN_TUMOR_MRI/, 
                                                      sub_path= "Training",
                                                      batch_size= 8,
                                                      img_resolution= 32,
                                                      transform= None)
            
            >>> df_test: pd.DataFrame = BRAINDataset(path= .../BRAIN_TUMOR_MRI/,
                                                     sub_path= "Testing",
                                                     batch_size= 1,
                                                     img_resolution= 32,
                                                     transform= None)


        UNIT TEST: 
            >>> path: _path = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\BRIAN_unit_testing\\"
            >>> sub_path: list[str] = ["Testing", "Training"]

            >>> df_train = BrainMRIDataset(path= path, 
                                           sub_path= sub_path[1],
                                           batch_size= 8)
    
            >>> df_test = BrainMRIDataset(path= path, 
                                          sub_path= sub_path[0],
                                          batch_size= 1)
    
            >>> print(df_test)

    """
    def __init__(self, path: _path, 
                       sub_path: str, 
                       batch_size: Optional[int] = 4, 
                       img_resolution: Optional[int] = 64, 
                       transform: Optional[_transform] = None) -> None:
        self.path = path
        self.sub_path = sub_path
        self.batch_size = batch_size
        self.img_resolution = img_resolution
        if transform:
            self.transform = transform
        self.categories: list[str] = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        if sub_path == 'Training':
            self.dataset: pd.DataFrame = self.get_data(path, 'Training', self.categories)
        if sub_path == 'Testing':
            self.dataset: pd.DataFrame = self.get_data(path, 'Testing', self.categories)
        indexes: list[int] = [x for x in range(len(self.dataset))]
        self.index_batch: list[list[int]] = [
            indexes[i : i + batch_size]
            for i in range(0, len(indexes), batch_size)
        ]
        
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    def __len__(self) -> tuple[int, ...]:
        return len(self.index_batch)
    
    
    def __getitem__(self, index: int) -> dict[_img, str]:
        batch: list[int] = self.index_batch[index]
        size: tuple[int, ...] = (self.img_resolution, self.img_resolution)
        images: list[int] = []
        labels: list[str] = []
        for i in batch:
            img: _img = Image.open(self.dataset.iloc[i].path).convert('LA').resize(size)
            img: np.ndarray = np.array(img)
            lbl: list[str] = [self.dataset.iloc[i].label]
            images.append(img)
            labels.append(np.array(lbl))
        images: torch.Tensor = torch.tensor(images).type(torch.float32)
        images: torch.Tensor = images.permute(0, 3, 1, 2)
        labels: torch.Tensor = torch.tensor(labels).type(torch.float32)
        images: torch.Tensor = self.normalize(images)
        return images, labels


    @classmethod
    def normalize(cls, x: torch.Tensor) -> torch.Tensor:
        return x / 255


    @classmethod
    def get_data(cls, path: _path, sub_path: str, categories: list[str]) -> pd.DataFrame:
        glioma_tumor: _path     = path + sub_path + "\\" + categories[0] + "\\"
        meningioma_tumor: _path = path + sub_path + "\\" + categories[1] + "\\"
        no_tumor: _path         = path + sub_path + "\\" + categories[2] + "\\"
        pituitary_tumor: _path  = path + sub_path + "\\" + categories[3] + "\\"
        glioma_tumor_list: list[str]  = [
            os.path.abspath(os.path.join(glioma_tumor, p))
            for p in os.listdir(glioma_tumor)                            
        ]
        meningioma_tumor_list: list[str] = [
            os.path.abspath(os.path.join(meningioma_tumor, p))
            for p in os.listdir(meningioma_tumor)
        ]
        no_tumor_list: list[str] = [
            os.path.abspath(os.path.join(no_tumor, p))
            for p in os.listdir(no_tumor)
        ]  
        pituitary_tumor_list: list[str] = [
            os.path.abspath(os.path.join(pituitary_tumor, p))
            for p in os.listdir(pituitary_tumor)
        ]
        glioma_tumor_labels: list[int]     = [0 for _ in range(len(glioma_tumor_list))]
        meningioma_tumor_labels: list[int] = [1 for _ in range(len(meningioma_tumor_list))]
        no_tumor_labels: list[int]         = [2 for _ in range(len(no_tumor_list))]
        pituitary_tumor_labels: list[int]  = [3 for _ in range(len(pituitary_tumor_list))]
        paths: _path      = glioma_tumor_list + meningioma_tumor_list + no_tumor_list + pituitary_tumor_list
        labels: list[int] = glioma_tumor_labels + meningioma_tumor_labels + no_tumor_labels + pituitary_tumor_labels
        dataframe: pd.DataFrame = pd.DataFrame.from_dict({'path': paths, 'label': labels})
        dataframe: pd.DataFrame = dataframe.sample(frac= 1)
        return dataframe 


############################################################################################################

# Class Data Analysis 
class BRAINAnalysis:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    @classmethod
    def plot(cls) -> _plot:
        ...

############################################################################################################


# Class Data Preprocess
class BRAINPreprocess:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    @classmethod
    def something(cls) -> None: ...
    

###########################################################################################################


# class GPU_acceleration
class GPU_Acceleration:
    def __init__(self, dataloader: _loader, device: Any) -> None:
        self.dataloader = dataloader
        self.device = device

    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    @classmethod
    def get_default_device(cls) -> str:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    

    @classmethod
    def to_device(cls, data: torch.Tensor, device: Any) -> _recurse:
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking= True)
    

    def __len__(self) -> int:
        return len(self.dataloader)
    

    def __iter__(self) -> Generator[_loader, None, None]:
        for data in self.dataloader:
            yield self.to_device(data, self.device)
      
      
    @classmethod
    def is_working_GPU(cls) -> bool:
        return True if cls.get_default_device() == 'cuda' else False


##########################################################################################################


# Image Classifier Base : -> Inherited : CNN_model
class Image_Classifier_Base(nn.Module):
    def accuracy(self, outputs: Any, labels: str) -> float:
        _, preds = torch.max(outputs, dim= 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    

    def training_step(self, batch: int) -> float:
        images, labels = batch
        out: Any = self(images)
        loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        return loss

    
    def validation_step(self, batch: int) -> dict[str, float]:
        images, labels = batch
        out: Any = self(images)
        loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        acc: float = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    

    def validation_epoch_end(self, outputs: Any) -> dict[str, tuple[int|float, ...]]:
        batch_losses: list = [
            x['val_loss'] for x in outputs
        ]
        epoch_loss: Container = torch.stack(batch_losses).mean()
        batch_accs: list = [
            x['val_acc'] for x in outputs
        ]
        epoch_acc: float = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    
    def epoch_end(self, epoch: int, result: float) -> _text:
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


##############################################################################################################


# Class Model params
class CNNHyperParams:
    '''
        Params:
            epochs: int           = No. of times the `Entire Dataset` is passed `Forward` abd `Backward` through the Neural Net
            criterion: _criterion = Criterion which `Optimizes` a Multi-class hinde loss 
            optimizor: _optimizor = ...
            learning_rate: float  = ...
            weight_decay: float   = ...
            momentum: float       = ...    

        DL Layers:
            convolution: _layer = Linear operation that involves the multiplication of a set of weights with the inpu
            pooling: _layer     = ...
            linear: _layer      = ...

        Neurons Engineering:
            flatten = ...
            dropout = ...

        Activation Functions:
            relu: _activation    = ...
            softmax: _activation = ...
            tanh: _activation    = ...

    '''
    #@ params
    epochs: int = 5
    optimizer: _optimizor = torch.optim.Adam
    learning_rate: float = 0.01
    criterion: _criterion = nn.CrossEntropyLoss()
    weight_decay: float = 0.01
    momentum: float = 0.9

    #@ Layers
    convolutional_1: _layer = nn.Conv2d(in_channels= 2, out_channels= 32, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_2: _layer = nn.Conv2d(in_channels= 32, out_channels= 64, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_3: _layer = nn.Conv2d(in_channels= 64, out_channels= 128, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_4: _layer = nn.Conv2d(in_channels= 128, out_channels= 32, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')

    
    pooling_1: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, 
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)
    
    pooling_2: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2,
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)
    
    pooling_3: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2,
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)

    pooling_4: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, 
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)

                            
    linear_1: _layer = nn.Linear(in_features= 128, out_features= 64, bias= True)
    linear_2: _layer = nn.Linear(in_features= 64, out_features= 32, bias= True)
    linear_3: _layer = nn.Linear(in_features= 32, out_features= 16, bias= True)
    linear_4: _layer = nn.Linear(in_features= 16, out_features= 8, bias= True)
    linear_5: _layer = nn.Linear(in_features= 8, out_features= 4, bias= True)


    #@ Activation functions 
    relu: _activation = nn.ReLU()
    tanh: _activation = nn.Tanh()
    softmax: _activation = nn.Softmax()

    #@ neuron characteristics
    flatten = nn.Flatten()
    dropout = nn.Dropout(p= 0.3, inplace= False)


##################################################################################################


# CNN Model 
class CNNModel(Image_Classifier_Base):
    '''
        CNN_Model(
            (convolution_layers): Sequential(
                (0): Conv2d(2, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): ReLU()
                (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
                (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
                (4): ReLU()
                (5): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
                (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
                (7): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
                (8): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1))
                (9): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
            )
            (linear_layers): Sequential(
                (0): Dropout(p=0.3, inplace=False)
                (1): Linear(in_features=128, out_features=64, bias=True)
                (2): ReLU()
                (3): Linear(in_features=64, out_features=32, bias=True)
                (4): ReLU()
                (5): Linear(in_features=32, out_features=16, bias=True)
                (6): ReLU()
                (7): Linear(in_features=16, out_features=8, bias=True)
                (8): ReLU()
                (9): Linear(in_features=8, out_features=4, bias=True)
            )
        )

    '''
    def __init__(self, num_classes: int) -> None:
        super(CNNModel, self).__init__()
        
        self.convolution_layers = nn.Sequential(
            CNNHyperParams.convolutional_1,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_1,

            CNNHyperParams.convolutional_2,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_2,

            CNNHyperParams.convolutional_3,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_3,

            CNNHyperParams.convolutional_4,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_4
        )

        self.linear_layers = nn.Sequential(
            CNNHyperParams.linear_1,
            CNNHyperParams.relu,

            CNNHyperParams.linear_2,
            CNNHyperParams.relu,

            CNNHyperParams.linear_3,
            CNNHyperParams.relu,

            CNNHyperParams.linear_4,
            CNNHyperParams.relu,

            CNNHyperParams.linear_5
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.convolution_layers(x)
        x: torch.Tensor = CNNHyperParams.dropout(x)
        x: torch.Tensor = CNNHyperParams.flatten(x)
        x: torch.Tensor = self.linear_layers(x)
        return CNNHyperParams.softmax(x)

###################################################################################################


# class Train_test_fit
class Train_Test_fit:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    @torch.no_grad()
    def evaluate(self, model: _model, val_loader: _loader) -> dict[str, float]: 
        model.eval()
        outputs: list[float] = [
            model.validation_step(batch) for batch in val_loader
        ]
        return model.validation_epoch_end(outputs)
        
    
    def get_learning_rate(self, optimizer: _optimizer) -> Any:
        for param_group in optimizer.param_groups:
            return param_group['lr']

    
    def fit_one_cycle(self, epochs: int, 
                            max_learning_rate: float, 
                            model: _model, 
                            train_loader: _loader, 
                            val_loader: _loader,  # test_loader
                            weight_decay: Optional[float|int] = 0,
                            grad_clip: Optional[float] = None, 
                            opt_function: Optional[_optimizer] = torch.optim.SGD) -> _text: 
        
        torch.cuda.empty_cache()
        history: list = []
        optimizer: _optimizer = opt_function(model.parameters(), max_learning_rate, weight_decay= weight_decay)
        #@: one-cycle LR scherudlar
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                 max_learning_rate, 
                                                 epochs= epochs, 
                                                 steps_per_epoch= len(train_loader))
        for epoch in range(epochs):
            model.train()
            train_losses: list = []
            lrs: list = []
            for batch in train_loader:
                loss: _loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
                optimizer.step()
                optimizer.zero_grad()

                lrs.append(self.get_learning_rate(optimizer))
                sched.step()

            #@: validation
            result: float = self.evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
    
        return history

##############################################################################################################  

# class Evaluate CNN
class EvaluateCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    @classmethod
    def accuracy_vs_no_of_epochs(cls, history: dict[str, float]) -> _plot:
        accuracies: list[float] = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs No. of epochs')
        plt.show()
    

    @classmethod
    def loss_vs_no_of_epochs(cls, history: dict[str, float]) -> _plot:
        train_losses: list[float] = [
            x.get('train_loss') for x in history
        ]
        val_losses: list[float] = [
            x['val_loss'] for x in history
        ]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs No. of Epochs')
        plt.show()
    

    @classmethod
    def learning_rate_vs_batch_number(cls, history: dict[str, float]) -> _plot:
        lrs: np.ndarray = np.concatenate([
            x.get('lrs', []) for x in history
        ])
        plt.plot(lrs)
        plt.xlabel('Batch No.')
        plt.ylabel('Learnjing rate')
        plt.title('Learning Rate vs Batch No.')
        plt.show()

    
    @classmethod
    def model_report(cls) -> pd.DataFrame | _plot:
        ...


    @classmethod
    def confusion_matrix(cls) -> _plot:
        ...




#############################################################################################################

# Class Save Model
class SaveCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    @classmethod
    def save(cls, model: _model, model_name: Optional[str] = 'BrainMRI_Model.pt') -> None:
        torch.save(model.state_dict(), model_name)
    

    @classmethod
    def is_save_model(cls, model: _model) -> bool:
        ...
    

    @classmethod
    def load_model(cls, new_model: _model, old_model: str) -> _model:
        new_model.load_state_dict(torch.load(old_model))
        return new_model
    

    @classmethod
    def is_load_model(cls, new_model: _model, old_model: _model) -> bool:
        ...



##############################################################################################################

# Driver code
if __name__.__contains__('__main__'):
    '''
    Module Usage : 
        from main_module import *

    '''
    pass

    

