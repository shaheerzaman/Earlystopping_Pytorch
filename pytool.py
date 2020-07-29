import numpy as np
import torch

class EarlyStopping:
    '''Early stop the training if validation loss doesn't improve after 
    a given patience'''
    def __init__(self, patience=7, verbose=False, 
            delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score -= val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''save model when validation loss decreas'''
        if self.verbose:
            print(f'validation loss decrease ({self.val_loss_min:.6f})')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
   
        