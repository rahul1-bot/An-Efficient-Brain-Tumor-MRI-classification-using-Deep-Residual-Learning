"""
Authors: Rahul Sawhney, Nikhil Kumar Pradhan, Amit Kumar
Project Control-Flow: 1) class Dataset
                      2) class DataAnalysis
                      3) class Preprocess
                      4) class GPU_Acceleration
                      5) class Hyperparams
                      6) class CNNModel
                      7) class TrainCNN
                      8) class TestCNN
                      9) class EvaluateCNN
                      10) class SaveCNN

Dataset Link: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri


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
                      
"""

# py imports
from __future__ import annotations
import typing
typing.__name__ = "__main_module__"
from typing import Any, NewType, Generator, Optional
import os, warnings
warnings.filterwarnings(action= 'ignore')


# Data Analysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Scripting ML 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# DL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms



# torch typing scripts
_path =  NewType("_path", Any)
_transform = NewType("_transform", Any)
_img = NewType("_img", Any)
_criterion = NewType("_criterion", Any)
_optimizor = NewType("_optimizor", Any)
_loss = NewType("_loss", Any)
_layer = NewType("_layer", Any)
_activation = NewType("_activation", Any)
_text = NewType("_text", Any)
_plot = NewType("_plot", Any)



# Data set class
class BrainMRIDataset(Dataset):
    def __init__(self, root: _path, batch_size: int, img_resolution: Optional[int] = 64, transform: Optional[_transform] = None) -> None:
        self.root: _path = root
        self.batch_size: int = batch_size
        self.img_resolution: int = img_resolution
        if transform:
            self.transform: _transform = transform
        self.categories: list[str] = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

    
    def __len__(self) -> tuple[int, ...]:
        ...


    def __getitem__(self, index: int) -> _img:
        return super(BrainMRIDataset, self).__getitem__(index)
    

    @classmethod
    def normalize(cls, x: torch.Tensor) -> Any:         # RGB: 225 pixels
        return x / 255


    @classmethod
    def get_data(cls, path: _path, sub_path: _path, categories: str) -> pd.DataFrame:
        ...



#@ nimkhil @Amit
# Class Data Analysis
class BRAINAnalysis:
    @classmethod
    def plot(cls) -> _img:
        ...

    ...


#@ ...
# Class Data Preprocess
class BRAINPreprocess:
    @classmethod
    def something(cls) -> Any:
        ...




# class GPU_acceleration
class GPU_Acceleration:
    ...


# class Hyperparams
class CNNHyperParams:
    epochs: int = ...
    criterion: _criterion = ...
    optimizor: _optimizor = ...
    # DL layers
    convolution_0: _layer = nn.Conv2d(in_channels= ..., out_channels= ..., 
                                      kernel_size= ..., stride= ...)
    
    convolution_1: _layer = nn.Conv2d(in_channels= ..., out_channels= ..., 
                                      kernel_size= ..., stride= ...)

    convolution_2: _layer = nn.Conv2d(in_channels= ..., out_channels= ..., 
                                      kernel_size= ..., stride= ...)

    convolution_3: _layer = nn.Conv2d(in_channels= ..., out_channels= ..., 
                                      kernel_size= ..., stride= ...)


    pooling_0: _layer = nn.MaxPool2d(kernel_size= ..., stride= ...)
    pooling_1: _layer = nn.MaxPool2d(kernel_size= ..., stride= ...)
    pooling_2: _layer = nn.MaxPool2d(kernel_size= ..., stride= ...)
    pooling_3: _layer = nn.MaxPool2d(kernel_size= ..., stride= ...)

    
    linear_0: _layer = nn.Linear(in_features= 64, out_features= ...)        # took img_resolution: 64 * 64 pixels
    linear_1: _layer = nn.Linear(in_features= 32, out_features= ...)
    linear_2: _layer = nn.Linear(in_features= 16, out_features= ...)
    linear_3: _layer = nn.Linear(in_features= 8,  out_features= ...)

    # neurons engineering
    flatten = nn.Flatten(start_dim= ..., end_dim= ...)
    dropout = nn.Dropout()

    # DL activation functions
    relu: _activation = F.relu(input= ..., inplace= ...)
    softmax: _activation = F.softmax(input= ..., inplace= ...)
    tanh: _activation = F.tanh(input= ..., inplace= ...)

    


# class CNN model
class CNNModel(nn.Module):
    def __init__(self) -> None:
        super(CNNModel, self).__init__()
        self.cnn_layers: _layers = nn.Sequential(
            # CNN architecture
            CNNHyperParams.convolution_0,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_0,

            CNNHyperParams.convolution_1,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_1,

            CNNHyperParams.convolution_2,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_2,

            CNNHyperParams.convolution_3,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_3
        )

        #data --> input:CNN:Output --> input:ANN:Output
        self.linear_layers: _layers = nn.Sequential(
             # flattening output neurons
            CNNHyperParams.flatten,
            CNNHyperParams.dropout,

            # ANN Architecture
            CNNHyperParams.linear_0,
            CNNHyperParams.tanh,

            CNNHyperParams.linear_1,
            CNNHyperParams.tanh,

            CNNHyperParams.linear_2,
            CNNHyperParams.tanh,

            CNNHyperParams.linear_3,
            CNNHyperParams.tanh,

            # output 
            CNNHyperParams.softmax

        )


    def forward(self, x: torch.Tensor) -> _model:
        x: torch.Tensor = self.cnn_layers(x)
        x: torch.Tensor = self.linear_layers(x)
        return x



# class train CNN
class TrainCNN(object):
    losses: list[float] = []

    @classmethod
    def train(cls, model: _model, epochs: int,  x_train: Tensor, y_train: Tensor, 
              optimizor: _optimizor, criterion: _criterion ) -> _text:
        ...


    @classmethod
    def test(cls) -> _text:
        ...


# CNN on test SET
class TestCNN:
    ...



# class Eval CNN
class EvaluateCNN:
    @classmethod
    def plot_1(cls) -> _plot:
        ...


    @classmethod
    def plot_2(cls) -> _plot:
        ...

   
    @classmethod
    def plot_3(cls) -> _plot:
        ...


    @classmethod
    def plot_4(cls) -> _plot:
        ...



# class Save CNN
class SaveCNN:
    @classmethod
    def save(cls, model: _model) -> None:
        torch.save(model.state_dict(), 'BrainMRI_MODEL.pt')
    
    
    @classmethod
    def load_model(cls, new_model: _model, old_model: str) -> _model:
        new_model.load_state_dict(torch.load(old_model))
        return new_model




# Driver code
if __name__ == "__main_module__":
    ...
