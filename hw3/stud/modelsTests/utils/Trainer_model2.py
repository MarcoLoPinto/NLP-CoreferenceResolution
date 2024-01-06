import torch
from torch.utils.data import DataLoader

try:
    from .Trainer import Trainer
except: # notebooks
    from stud.modelsTests.utils.Trainer import Trainer

class Trainer_model2(Trainer):

    def __init__(self):
        super().__init__()

    def init_history(self, saved_history):
        history = {}
        history['train_history'] = [] if saved_history == {} else saved_history['train_history']
        history['valid_loss_history'] = [] if saved_history == {} else saved_history['valid_loss_history']
        history['valid_f1_history'] = [] if saved_history == {} else saved_history['valid_f1_history']
        return history

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        raise NotImplementedError

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        raise NotImplementedError

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        raise NotImplementedError

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        f1 = evaluations_results['f1']

        history['valid_loss_history'].append(valid_loss)
        history['valid_f1_history'].append(f1)

        return history

    def print_evaluations_results(self, valid_loss, evaluations_results):
        f1 = evaluations_results['f1']
        print(f'#                   Validation loss => {valid_loss:0.6f} | f1: {f1:0.6f} #')

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        return (
            history['valid_f1_history'][-1] > max([0.0] + history['valid_f1_history'][:-1]) and 
            history['valid_f1_history'][-1] > min_score
        )

        