# -*- coding: utf-8 -*-
from __future__ import annotations

__authors__: list[str] = ['Rahul_Sawhney', 'Nikhil_kumar_pradhan']
#$ exec(False) if not __pipeline_2__.__dict__() or any('Rahul_Sawhney', 'Nikhil_kumar_pradhan',) not in __authors__ 

from pipeline_1 import *
import torchvision.models as models
from typing import NewType, Any


#@: Pipeline2: Machine Learning
    #@: class GPU_Acceleration
    #       : __init__                          -> None
    #       : __repr__                          -> str(dict[str, str])
    #       : get_default_device()              -> str
    #       : to_device()                       -> _recurse| base::str
    #       : __len__                           -> int
    #       : __iter__                          -> Generator[str, None, None]
    #       : is_working_gpu()                  -> bool
    #
    #@: Class Image_Classifier_Base (nn.Module)
    #       : accuracy()                        -> torch.Tensor
    #       : training_step()                   -> float    
    #       : validation_step()                 -> dict[str, float|int]
    #       : validation_epoch_end()            -> dict[str, float|int]
    #       : epoch_end()                       -> _text
    #
    #@: Class Hyperparams :@ ClassVar Class
    #       : __repr__                          -> str(dict[str, str])
    #       : > model_hyper_params              :
    #       : > model_layers                    :
    #       : > model_activation_functions      :
    #       : > neurons_characteristics         :
    #
    #@: Class CNN_Model (Image_Classifier_Base):
    #       : __init__
    #       : forward()                         -> torch.Tensor
    #       : *backpropogation()                -> None | Nimkhil 
    #
    #@: Class PreTrained_Models:
    #       : __repr__                          -> str(dict[str, str])
    #       : ResNet()                          -> _model
    #       : UNet()                            -> _model
    #       : Inception()                       -> _model
    #
    #@: Class Train_Test_fit
    #       : __repr__                          -> str(dict[str, str])
    #       : @torch.no_grad( evaluate() )      -> dict[str, float]
    #       : get_learning_rate()               -> float
    #       : fit_one_cycle()                   -> _text | list[dict[str, float|int]]


# torch typing scripts
_path =  NewType("_path", Any)
_transform = NewType("_transform", Any)
_img_path = NewType("_img", Any)
_criterion = NewType("_criterion", Any)
_optimizer = NewType("_optimizer", Any)
_loss = NewType("_loss", Any)
_layer = NewType("_layer", Any)
_activation = NewType("_activation", Any)
_text = NewType("_text", Any)
_plot = NewType("_plot", Any)
_loader = NewType("_loader", Any)
_recurse = NewType("_recurse", Any)




#@: GPU acceleration 
class GPU_Accerlaration:
    def __init__(self, train_loader: _loader, device: str) -> None:
        self.train_loader = train_loader
        self.device = device

    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    
    __str__ = __repr__


    @classmethod
    def get_default_device(cls) -> str:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


    @classmethod
    def to_device(cls, data: torch.Tensor, device: str) -> _recurse | [base, str]:
        if isinstance(data, (list, tuple)):
            return [cls.to_device(x, device) for x in data]
        return data.to(device, non_blocking= True)


    def __len__(self) -> int:
        return len(self.train_loader)

    
    def __iter__(self) -> Generator[_loader, None, None]:
        for batch in self.train_loader:
            yield self.to_device(batch, self.device)
    

    @classmethod
    def is_working_gpu(cls) -> bool:
        return True if cls.get_default_device() == 'cuda' else False





#@: Image Classifier Base
class Image_Classifier_Base(nn.Module):                                                 
    def accuracy(self, output: torch.Tensor, labels: list[int]) -> torch.Tensor:        
        _, preds = torch.max(output, dim= 1)                                            # [[[...]], [[...]], [[...]], ...] -> max() -> [[[INDEX]], [[.M.]], [[.M.]], [[.M.]], ...] -> dim -> [[INDEX], [.M., .M., ...]]
                                                                                        # preds = [...]
                                                                                        # dim: dimention to reduce
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))             #  


    def training_step(self, batch: list[(_img, int)]) -> float:
        images, labels = batch
        out: torch.Tensor = self(images)
        #loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        loss: _loss = F.cross_entropy(out, labels)
        return loss


    def validation_step(self, batch: list[(_img, int)]) -> dict[str, float]:
        images, labels = batch
        out: torch.Tensor = self(images)
        loss: _loss = F.cross_entropy(out, labels)
        acc: float = self.accuracy(out, labels)
        return {
            'val_loss': loss.detach(), 'val_acc': acc                               # .detech() -> tensor ka gradient memory se hta deggi
        }

    
    def validation_epoch_end(self, outputs: list[dict[str, float|int]]) -> dict[str, float|int]:
        batch_losses: list[float] = [x['val_loss'] for x in outputs]            # [ {'val_loss': ..., 'val_acc': ... }, {'val_loss':..., 'val_acc': ...}, {'':..., ''...}, ...] for each dictionary hold the value of 'val_loss'
        epoch_loss: Container = torch.stack(batch_losses).mean()                # loss ko stack mei dal ke normal mean() -> torch.Tensor
        batch_accs: list[float] = [x['val_acc'] for x in outputs]               # same as above but 'val_acc'
        epoch_acc: float = torch.stack(batch_accs).mean()                       # ...           -> torch.Tensor
        return {
            'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()          #               -> torch.Tensor([[3.43]]) -> .item() -> 3.43
        }


    def epoch_end(self, epoch: int, result: dict) -> _text: #| Generator[str, None, None]
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,                                      # result: { 'lrs' : [] , 'train_loss' : ..., 'val_loss': ..., 'val_acc': ...}
                result['lrs'][-1], 
                result['train_loss'], 
                result['val_loss'], 
                result['val_acc'])
        )

    


#@: class CNNHyperParamas
class CNNHyperParams:
    epochs: int = 5
    optimizer: _optimizor = torch.optim.Adam                # isme different optimizers lga ke dekh
    #learning_rate: float = 0.01
    criterion: _criterion = nn.CrossEntropyLoss()
    weight_decay: float = 0.01
    momentum: float = 0.9

    # hmara image Input: # {20 X 2 X 256 X 256}  [ batch: input.batch_size, in_channels: image_channels, length: image.length,  breadth: image.breadth ]

    convolutional_1: _layer = nn.Conv2d(in_channels= 2, out_channels= 32,  # {20 X 32 X 122 X 122}   [ {32 <different>} images {<32> different kernels lga ke of size(3, 3)} of size(122 * 122) aye] * batch.size 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_2: _layer = nn.Conv2d(in_channels= 32, out_channels= 64, # {20 X 64 X 122 X 122}   [ {64 <different>} images {<64> different kernels lga ke} of size(122 * 122) ] * BATCH.size 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')                                          
    

    convolutional_3: _layer = nn.Conv2d(in_channels= 64, out_channels= 128, # {20 X 128 X 122 X 122}  [ {128 <different>} images {<128> different kernels lga ke} of size(122 * 122) ] * BATCH.size 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_4: _layer = nn.Conv2d(in_channels= 128, out_channels= 256, # {20 X 256 X 122 X 122}  [ {256 <different>} images {<256> different kernels lga ke} of size(122 * 122) ] * BATCH.size 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')

    
    pooling_1: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 1,        
                                     padding= 0, dilation= 1,
                                     return_indices= False)
                                    
    
    pooling_2: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 1,
                                     padding= 0, dilation= 1,
                                     return_indices= False)
                                     
    

    pooling_3: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 1,
                                     padding= 0, dilation= 1,
                                     return_indices= False)
                                     

    pooling_4: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, 
                                     padding= 0, dilation= 1,
                                     return_indices= False)
                                     

    # last convolution_4 layer ke baad image input size : [20 X 256 X 122 X 122]
    # linear layer ko ussi hisab se km krr diyo taki last layer ka in_features > 4 se jada ho                                                
    linear_1: _layer = nn.Linear(in_features= 256 * 122 * 122, out_features= 64, bias= True)    # out liya  ->    [256, 128, 64, 32, 16, 8]
    linear_2: _layer = nn.Linear(in_features= 64, out_features= 32, bias= True)                 # out liya  -> 
    linear_3: _layer = nn.Linear(in_features= 32, out_features= 16, bias= True)                 # out liya  -> 
    linear_4: _layer = nn.Linear(in_features= 16, out_features= 8, bias= True)                  # out liya  -> 
    linear_5: _layer = nn.Linear(in_features= 8, out_features= 4, bias= True)                   # out HOGGA -->>> 4

 
    relu: _activation = nn.ReLU()
    softmax: _activation = nn.Softmax()

    flatten = nn.Flatten()                                                          
    dropout = nn.Dropout(p= 0.3, inplace= False)                                                # isse [ 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 ] le krr try kariyo
    # LINEAR LAYER MEI : dropout ke binna -> input image size : [20 , 256 * 122 * 122]
    #                  : jb droput 0.2 liya -> input image size : [18, 256 * 122 * 122]


#@: Basic CNN Model
class CNNModel(Image_Classifier_Base):
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
        #print(x.shape)
        x: torch.Tensor = self.convolution_layers(x)
        #print(x.shape)
        #x: torch.Tensor = CNNHyperParams.dropout(x)
        #print(x.shape)
        x: torch.Tensor = CNNHyperParams.flatten(x)
        #print(x.shape)
        x: torch.Tensor = self.linear_layers(x)
        #print(x.shape)
        return x



#@: Pretrained Model
class Pretrained_models:
    def __repr__(self) -> str(dict[str, str]):
        return {item: value for item, value in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])}


    __str__ = __repr__


    @classmethod
    def Resnet(cls, train_loader: _loader, pretrained: Optional[bool] = False) -> _model:
        ...

    
    @classmethod
    def UNet(cls, train_loader: _loader, pretrained: Optional[bool] = False) -> _model:
        ...


    @classmethod
    def Inception(cls, train_loader: _loader, pretrained: Optional[bool] = False) -> _model:
        ...





#@: Train_test_fit
class Train_test_fit:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__


    @torch.no_grad()
    def evaluate(self, model: _model, test_loader: _loader) -> dict[str, float]:
        model.eval()
        outputs: list[dict[str, float|int]] = [model.validation_step(batch) for batch in test_loader]
        return model.validation_epoch_end(outputs)


    def get_learning_rate(self, optimizer: _optimizer) -> float:
        for param_group in optimizer.param_groups:
            return param_group['lr']
    


    def fit_one_cycle(self, epochs: int, max_learning_rate: float, 
                                         model: _model, 
                                         train_loader: _loader, 
                                         test_loader: _loader, 
                                         weight_decay: Optional[float|int] = 0,
                                         grad_clip: Optional[float] = None,
                                         opt_function: Optional[_optimizer] = torch.optim.Adam) -> _text | list[dict[str, float|int]]:
        torch.cuda.empty_cache()
        history: list = []
        optimizer: _optimizer = opt_function(model.parameters(), lr= max_learning_rate, weight_decay= weight_decay)

        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer= optimizer, 
                                                    max_lr= max_learning_rate, 
                                                    epochs= epochs, 
                                                    steps_per_epoch= len(train_loader))
        for epoch in range(epochs):
            model.train()
            train_losses: list = []
            lrs: list = []
            for batch in train_loader:
                loss: _loss = model.training_step(batch= batch)
                train_losses.append(loss)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_value_(parameters= model.parameters(), grad_clip= grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()

                lrs.append(self.get_learning_rate(optimizer= optimizer))
                sched.step()
            
            # validation 
            result: dict[str, float] = self.evaluate(model= model, test_loader= test_loader)
            result['train_loss'] = torch.stack(tensors= train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch= epoch, result= result)
            history.append(result)

        return history



#@: Driver code
if __name__.__contains__('__main__'):
    #print('hemllo')
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
    

    model: _model = CNNModel(3)
    #print(model)

    history: list[dict[str, float]] = [Train_test_fit().evaluate(model= model, test_loader= testing_data_loader)]
    print(history)
