# -*- coding: utf-8 -*-
from __future__ import annotations

__authors__: list[str] = ['Rahul_Sawhney', 'nikhil_kumar_pradhan']
#$ exec(False) if not __pipeline_1__.__dict__() or any('Rahul_Sawhney', 'nikhil_kumar_pradhan',) not in __authors__ 


from pipeline_2 import *
from typing import NewType, Any


#@: Pipeline 3: Model Evaluation / Model Monitoring
    #@: Class EvaluateCNN
    #       : __repr__                              -> str(dict[str, str])
    #       : accuracy_vs_no_of_epochs()            -> _plot
    #       : loss_vs_no_of_epochs()                -> _plot    
    #       : learning_rate_vs_batch_number()       -> _plot
    #       : model_report()                        -> pd.DataFrame
    #       : confusion_matrix()                    -> _plot
    #  
    #@: Class SaveCNN
    #       : __repr__                              -> str(dict[str, str])
    #       : save()                                -> None
    #       : is_save_model()                       -> bool
    #       : load_model()                          -> _model
    #       : is_loaded_model()                     -> bool



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


#@: Class EvaluateCNN
class EvaluateCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__


    @classmethod
    def accuracy_vs_no_of_epochs(cls, history: dict[str, tuple[int|float, ...]]) -> _plot:
        accuracies: list[float] = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs No. of epochs')
        plt.show()
    

    @classmethod
    def loss_vs_no_of_epochs(cls, history: dict[str, tuple[int|float, ...]]) -> _plot:
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
    def learning_rate_vs_batch_number(cls, history: dict[str, tuple[int|float, ...]]) -> _plot:
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



#@: Save Models
class SaveCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__

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




#@: Driver Code
if __name__.__contains__('__main__'):
    print('hemllo')