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
from PIL import Image


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

# Data set class
class BrainMRIDataset(Dataset):
    def __init__(self, path: _path, sub_path: str, batch_size: int, img_resolution: Optional[int] = 64, transform: Optional[_transform] = None) -> None:
        self.path = path
        self.sub_path = sub_path
        self.batch_size = batch_size  
        self.img_resolution = img_resolution
        self.categories: list[str] = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        if transform:
            self.transform = transform
        
        if sub_path == "Training":
            self.dataset: pd.DataFrame = self.get_data(path, "Training", self.categories)
        if sub_path == "Testing":
            self.dataset: pd.DataFrame = self.get_data(path, "Testing", self.categories)
        
        indexes: list[int] = [x for x in range(len(self.dataset))]
        self.index_batch: list[int] = [indexes[i:i + batch_size] for i in range(0, len(indexes), batch_size)]



    def __len__(self) -> tuple[int, ...]:
        return self.dataset.shape



    def __getitem__(self, index: int) -> dict[_img, str]:
        batch: list = self.index_batch[index]
        size: tuple[int, ...] = (self.img_resolution, self.img_resolution)      # 64 * 64 pixels
        images: list = []
        labels: list = []
        for i in batch:
            img: _image = Image.open(self.dataset.iloc[i].path).convert("LA").resize(size)
            img: np.ndarray = np.array(img)
            lbl: list[str] = [self.dataset.iloc[i].label]
            images.append(img)
            labels.append(lbl)
        
        images: torch.Tensor = torch.Tensor(images).type(torch.float32)
        images: torch.Tensor = images.permute(0, 3, 1, 2)
        lables: torch.Tensor = torch.tensor(lables).type(torch.float32)
        images: torch.Tensor = self.normalize(images)
        mapper: dict[_img, str] = {"image": images, "label": labels}
        return mapper 



    @classmethod
    def normalize(cls, x: torch.Tensor) -> Any:         # RGB: 255 pixels
        return x / 255



    @classmethod
    def get_data(cls, path: _path, sub_path: str, categories: list[str]) -> pd.DataFrame:
        glioma_tumor: _path     = path + sub_path + "/" + categories[0] + "/"
        meningioma_tumor: _path = path + sub_path + "/" + categories[1] + "/"
        no_tumor: _path         = path + sub_path + "/" + categories[3] + "/"
        pituitary_tumor: _path  = path + sub_path + "/" + categories[4] + "/"

        glioma_tumor_list: list[str] = [os.path.join(glioma_tumor, p) 
                                        for p in os.listdir(glioma_tumor)]

        meningioma_tumor_list: list[str] = [os.path.join(meningioma_tumor, p)
                                            for p in os.listdir(meningioma_tumor)]
        
        no_tumor_list: list[str] = [os.path.join(no_tumor, p) 
                                    for p in os.listdir(no_tumor)]

        pituitary_tumor_list: list[str] = [os.path.join(pituitary_tumor, p) 
                                           for p in os.listdir(pituitary_tumor)]

        
        glioma_tumor_labels: list[int] = [0 for _ in range(len(glioma_tumor_list))]
        meningioma_tumor_labels: list[int] = [1 for _ in range(len(meningioma_tumor_list))]
        no_tumor_labels: list[int] = [2 for _ in range(len(no_tumor_list))]
        pituitary_tumor_labels: list[int] = [3 for _ in range(len(pituitary_tumor_list))]


        paths: _path = glioma_tumor_list + meningioma_tumor_list + no_tumor_list + pituitary_tumor_list
        labels: list[int] = glioma_tumor_labels + meningioma_tumor_labels + no_tumor_labels + pituitary_tumor_labels
        
        dataframe: pd.DataFrame = pd.DataFrame.from_dict({"path": paths, "label": labels})
        dataframe: pd.DataFrame = dataframe.sample(frac= 1)
        return dataframe






#@ nimkhil @Amitwa
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
    @classmethod
    def Hardware_Accelerator(cls) -> _text:
        device: Any = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using : {device}")


    @classmethod
    def set_GPU(cls) -> None:
        ...
    

    @classmethod
    def load_GPU(cls) -> None:
        ...


    @classmethod
    def is_working_GPU(cls) -> bool:
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
    def train(cls, model: _model, epochs: int,  x_train: torch.Tensor, 
                   y_train: torch.Tensor, optimizor: _optimizor, criterion: _criterion ) -> _text:
        ...


    @classmethod
    def test(cls, model: _model, x_test: torch.Tensor, y_test: torch.Tensor, 
                  optimizor: _optimizor, criterion: _criterion) -> _text:
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


    @classmethod
    def plot_5(cls) -> _plot:
        ...


    @classmethod
    def plot_6(cls) -> _plot:
        ...



# class Save CNN
class SaveCNN:
    @classmethod
    def save(cls, model: _model) -> None:
        torch.save(model.state_dict(), 'BrainMRI_MODEL.pt')
    
    
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





# Driver code
if __name__ == "__main_module__":
    path: _path = ...
    sub_path: list[str] = ["Testing", "Training"]
    

    df_train: pd.DataFrame = BrainMRIDataset(path= path, 
                                            sub_path= sub_path[1],
                                            batch_size= 6, 
                                            img_resolution= 64, 
                                            transforms= None)
    
    df_test: pd.DataFrame = BrainMRIDataset(path= path, 
                                            sub_path= sub_path[0],
                                            batch_size= 6, 
                                            img_resolution= 64, 
                                            transforms= None)
    


    