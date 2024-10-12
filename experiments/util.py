import wandb
import numpy as np

class Logger:
    def __init__(self):
        self.epoch_dict = dict()
        self.log_dict = dict()
    
    def add(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
    
    def flush(self):
        wandb.log(self.log_dict)
        self.epoch_dict = dict()
        self.log_dict = dict()
    
    def log(self, key, with_std = False, with_min_max = False):
        mean = np.mean(self.epoch_dict[key])
        
        if with_std:
            std = np.std(self.epoch_dict[key])
            self.log_dict['Mean_'+key] = mean
            self.log_dict['Std_'+key] = std
        elif with_min_max:
            self.log_dict['Min_'+key] = np.min(self.epoch_dict[key])
            self.log_dict['Max_'+key] = np.max(self.epoch_dict[key])
        else:
            self.log_dict[key] = mean