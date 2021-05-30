# -*- coding: utf-8 -*-
from __future__ import annotations

__authors__: list[str] = ['Rahul_Sawhney', 'nikhil_kumar_pradhan']
#$ exec(False) if not __pipeline_1__.__dict__() or any('Rahul_Sawhney', 'nikhil_kumar_pradhan',) not in __authors__ 

__doc__ =  r'''
    Project Topic: Brain Tumor MRI Classification with State of the art Computer Vision Models

    project Abstract: ...

    Project Control Flow: pipe1ine_1: Data Engineering
                                1) Kaggle API call
                                2) Class Dataset
                                3) Class DataAnalysis
                                4) Class DataPreprocess

                          pipeline_2: Machine Learning
                                4) Class GPU_acceleration
                                5) class Image_Classification_Base 
                                6) class Hyperparams
                                7) Class CNNModel : @: general_CNN_model
                                8) class Pretrained_models
                                9) Class Train_test_fit
                          
                          pipeline_3: Evaluation Metrics
                                10) class Evaluate_Model
                                11) Class SaveModel
''' 
#@ pipeline_1: Data Engineering
    #@: class Kaggle_API_call
    #       : __init__                      -> None
    #       : __repr__                      -> str(dict[str, str])
    #       : __getitem__                   -> Any
    #       : connect_api()                 -> None
    #       : is_connected_api()            -> bool
    #       : retrieve_data()               -> _dir
    #
    #@: class dataset
    #       : __init__                      -> None
    #       : get_data()                    -> pd.DataFrame[_img_path, int]
    #       : __repr__                      -> str(dict[str, str])
    #       : __len__                       -> int
    #       : __getitem__                   -> tuple[_img, int]
    #
    #@: class Data Analysis
    #       : __repr__                      -> str(dict[str, str]) 
    #       : batch_img_display             -> _plot
    #       : batch_img_rgb_display         -> _plot
    #       : histplot                      -> _plot
    #       : ...                           -> _plot
    #
    #@: class DataPreprocess 
    #       : __repr__                      -> str(dict[str, str]) 
    #       : train_transform_container()   -> Container[Module[_transform]]
    #       : test_transfrom_container()    -> Container[Module[_transform]]


# Python Imports
import typing
from typing import Any, NewType, Generator, Optional, Union, ClassVar, Container
import os, warnings, sys
warnings.filterwarnings(action= 'ignore')


# Data Analysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as tFF


# DL Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# torch typing scripts
_path =  NewType('_path', Any)
_transform = NewType('_transform', Any)
_img_path = NewType('_img', Any)
_criterion = NewType('_criterion', Any)
_optimizer = NewType('_optimizer', Any)
_loss = NewType('_loss', Any)
_layer = NewType('_layer', Any)
_activation = NewType('_activation', Any)
_text = NewType('_text', Any)
_plot = NewType('_plot', Any)
_loader = NewType('_loader', Any)
_recurse = NewType('_recurse', Any)


#@: API Call
class Kaggle_api_call:
    def __init__(self, data_name: str) -> None:
        ...

    def __repr__(self) -> str(dict[str, str]):
        ...
    
    __str__ = __repr__

    def __getitem__(self) -> Any:
        ...

    @classmethod
    def connect_api(cls) -> None:
        ...
    
    @classmethod
    def is_connected_api(cls) -> bool:
        ...
    
    @classmethod
    def retrieve_data(cls, download: Optional[bool] = False) -> _dir:
        ...




#@: Class DataSet
class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, path: _path, sub_path: str, 
                                    categories: list[str], 
                                    img_resolution: Optional[int] = 64,
                                    transform: Optional[Container[Module[_transform]]] = None) -> None:
        self.path = path
        self.sub_path = sub_path
        self.categories = categories
        self.img_resolution = img_resolution
        if transform:
            self.transform = transform
        self.dataset: pd.DataFrame[_img_path, int] = self.get_data(path, sub_path, self.categories)
    


    @classmethod
    def get_data(cls, path: _path, sub_path: str, categories: list[str]) -> pd.DataFrame[_img_path, int]:
        glioma_tumor: _path = path + sub_path + '\\' + categories[0] + '\\'
        meningioma_tumor: _path = path + sub_path + '\\' + categories[1] + '\\'
        no_tumor: _path = path + sub_path + '\\' + categories[2] + '\\'
        pituitary_tumor: _path = path + sub_path + '\\' + categories[3] + '\\'

        
        glioma_tumor_img_path: list[_img_path] = [
            os.path.abspath(os.path.join(glioma_tumor, p)) for p in os.listdir(glioma_tumor)
        ] 
        meningioma_tumor_img_path: list[_img_path] = [
            os.path.abspath(os.path.join(meningioma_tumor, p)) for p in os.listdir(meningioma_tumor)
        ]
        no_tumor_img_path: list[_img_path] = [
            os.path.abspath(os.path.join(no_tumor, p)) for p in os.listdir(no_tumor)
        ]
        pituitary_tumor_img_path: list[_img_path] = [
            os.path.abspath(os.path.join(pituitary_tumor, p)) for p in os.listdir(pituitary_tumor)
        ]

        glioma_tumor_label: list[int] = [0 for _ in range(len(glioma_tumor_img_path))]
        meningioma_tumor_label: list[int] = [1 for _ in range(len(meningioma_tumor_img_path))]
        no_tumor_label: list[int] = [2 for _ in range(len(no_tumor_img_path))]
        pituitary_tumor_label: list[int] = [3 for _ in range(len(pituitary_tumor_img_path))]

        # pd.Dataframe[_img_path, int]  
        all_img_path: list[_img_path] = glioma_tumor_img_path + meningioma_tumor_img_path + no_tumor_img_path + pituitary_tumor_img_path
        all_label: list[int] = glioma_tumor_label + meningioma_tumor_label + no_tumor_label + pituitary_tumor_label

        dataframe: pd.DataFrame[_img_path, int] = pd.DataFrame.from_dict({'path' : all_img_path, 'label': all_label})
        dataframe: pd.DataFrame[_img_path, int] = dataframe.sample(frac= 1)
        return dataframe



    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    
    __str__ = __repr__


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, index: int) -> tuple[_img, int]:
        img_size: tuple[int, ...] = (self.img_resolution, self.img_resolution)
        image: _img = Image.open(self.dataset.iloc[index].path).convert('LA').resize(img_size)
        label: int = self.dataset.iloc[index].label 
        
        if self.transform:
            image: torch.Tensor = self.transform(image)
        
        return image, label       




#@: Class Data Analysis
class BrainAnalysis:
    labels_map: ClassVar[dict[int, str]] = {
        0: 'glioma_tumor',
        1: 'meningioma_tumor',
        2: 'no_tumor',
        3: 'pituitary_tumor'
    }

    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__


    
    @classmethod
    def batch_img_display(cls, training_data: object) -> _plot:
        figure: plt.figure = plt.figure(figsize= (8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_index: int = torch.randint(len(training_data), size= (1,)).item()
            img, label = training_data[sample_index]
            figure.add_subplot(rows, cols, i)
            plt.title(cls.labels_map[label])
            plt.axis('off')
            plt.imshow(np.asarray(tFF.to_pil_image(img).convert('RGB')), cmap= 'gray')
        plt.show() 



    @classmethod
    def data_loader_img_display(cls, training_loader: _loader) -> _plot:
        train_feature, train_label = training_data_loader.__iter__().__next__()
        image: _img = np.asarray(tFF.to_pil_image(train_feature[0]).convert('RGB'))
        label: str = train_label[0]
        plt.title(cls.labels_map[int(label.numpy())])
        plt.imshow(image, cmap= 'gray')
        plt.show()


   
    @classmethod
    def train_histplot(cls, train_loader: _loader) -> _plot:
        ...

    
    @classmethod
    def test_histplot(cls, test_loader: _loader) -> _plot:
        ...




#@: Class DataPreprocess
class BrainPreprocess:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__


    @classmethod
    def train_transform_container(cls) -> Container[Module[_transform]]:
        container: Container[Module[_transform]] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(256, padding= 4, padding_mode= 'reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return container


    @classmethod
    def test_transform_container(cls) -> Container[Module[_transform]]:
        container: Container[Module[_transform]] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return container


    @classmethod
    def Normalize_img(cls, x: torch.Tensor) -> torch.Tensor:
        return x / 256



    @classmethod
    def de_normalize_img(cls, x: torch.Tensor) -> torch.Tensor:
        return x * 256
    




#@: Driver Code
if __name__.__contains__('__main__'):
    path: _path = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\__Desktop\\__temp\\BRIAN_MRI\\'
    sub_path: list[str] = ['Testing', 'Training']
    categories: list[str] = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


    training_data: object = BrainDataset(path= path,
                                         sub_path= sub_path[1],
                                         categories= categories,
                                         img_resolution= 64,
                                         transform= BrainPreprocess.train_transform_container())
    
    testing_data: object = BrainDataset(path= path,
                                        sub_path= sub_path[0],
                                        categories= categories, 
                                        img_resolution= 64,
                                        transform= BrainPreprocess.test_transform_container())
    
    
    training_data_loader: _loader = DataLoader(dataset= training_data, 
                                               batch_size= 20, 
                                               shuffle= True, 
                                               num_workers= 4,
                                               pin_memory= True)


    testing_data_loader: _loader = DataLoader(dataset= testing_data,
                                              batch_size= 20, 
                                              shuffle= False,
                                              num_workers= 4,
                                              pin_memory= True)

    # print(training_data_loader)
    # BrainAnalysis.batch_img_display(training_data= training_data)
    # BrainAnalysis.data_loader_img_display(training_loader= training_data_loader)
