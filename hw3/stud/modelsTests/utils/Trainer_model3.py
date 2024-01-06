import torch
from torch.utils.data import DataLoader

try:
    from .Trainer import Trainer
except: # notebooks
    from stud.modelsTests.utils.Trainer import Trainer

class Trainer_model3(Trainer):

    def __init__(self):
        super().__init__()

    def init_history(self, saved_history):
        history = {}
        history['train_history'] = [] if saved_history == {} else saved_history['train_history']
        history['valid_loss_history'] = [] if saved_history == {} else saved_history['valid_loss_history']
        history['valid_accuracy_history'] = [] if saved_history == {} else saved_history['valid_accuracy_history']
        return history

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        raise NotImplementedError

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        raise NotImplementedError

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        evaluations_results = Trainer_model3.evaluate(predictions, labels)
        return evaluations_results

    def update_history(self, history, valid_loss, evaluations_results):
        ''' must return the updated history dictionary '''
        acc = evaluations_results['accuracy']

        history['valid_loss_history'].append(valid_loss)
        history['valid_accuracy_history'].append(acc)

        return history

    def print_evaluations_results(self, valid_loss, evaluations_results):
        acc = evaluations_results['accuracy']
        print(f'#               Validation loss => {valid_loss:0.6f} | accuracy: {acc:0.6f} #')

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        return (
            history['valid_accuracy_history'][-1] > max([0.0] + history['valid_accuracy_history'][:-1]) and 
            history['valid_accuracy_history'][-1] > min_score
        )

        