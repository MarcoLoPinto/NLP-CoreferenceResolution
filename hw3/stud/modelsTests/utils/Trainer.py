import torch
from torch.utils.data import DataLoader

class Trainer():

    def __init__(self):
        pass

    def train(
        self,
        final_model,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader = None,
        epochs: int = 5,
        verbose: bool = True,
        save_best = True,
        save_path_name = None,
        min_score = 0.5,
        saved_history = {},
        device = 'cpu'
    ):  
        """Training and evaluation function in order to save the best model

        Args:
            final_model (any): the ML wrapped model
            optimizer (torch.optim.Optimizer): the torch.optim.Optimizer used 
            train_dataloader (DataLoader): the train data created with torch.utils.data.Dataloader
            valid_dataloader (DataLoader, optional): the dev data created with torch.utils.data.Dataloader. Defaults to None.
            epochs (int, optional): number of maximum epochs. Defaults to 5.
            verbose (bool, optional): if True, then each epoch will print the training loss, the validation loss and the f1-score. Defaults to True.
            save_best (bool, optional): if True, then the best model that surpasses min_score will be saved. Defaults to True.
            save_path_name (str, optional): path and name for the best model to be saved. Defaults to None.
            min_score (float, optional): minimum score acceptable in order to be saved. Defaults to 0.5.
            saved_history (dict, optional): saved history dictionary from another session. Defaults to {}.
            device (str, optional): if we are using cpu or gpu. Defaults to 'cpu'.

        Returns:
            a dictionary of histories
        """

        history = self.init_history(saved_history) # override

        final_model.model.to(device)

        for epoch in range(epochs):
            losses = []
            
            final_model.model.train()

            # batches of the training set
            for step, sample in enumerate(train_dataloader):
                dict_out = self.compute_forward(final_model.model, sample, device, optimizer = optimizer) # override
                losses.append(dict_out['loss'].item())

            mean_loss = sum(losses) / len(losses)
            history['train_history'].append(mean_loss)
            
            if verbose or epoch == epochs - 1:
                print(f'Epoch {epoch:3d} => avg_loss: {mean_loss:0.6f}')
            
            if valid_dataloader is not None:

                valid_out = self.compute_validation(final_model, valid_dataloader, device) # override

                evaluations_results = self.compute_evaluations(valid_out['labels'], valid_out['predictions']) # override

                history = self.update_history(history, valid_out['loss'], evaluations_results) # override

                if verbose:
                    self.print_evaluations_results(valid_out['loss'], evaluations_results) # override

                # saving...

                if save_best and save_path_name is not None:
                    if self.conditions_for_saving_model(history, min_score): # override
                        print(f'----- Best value obtained, saving model -----')
                        final_model.model.save_weights(save_path_name)
                    
        return history


    def init_history(self, saved_history):
        ''' must return the initialized history dictionary '''
        raise NotImplementedError

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
        raise NotImplementedError

    def print_evaluations_results(self, valid_loss, evaluations_results):
        print('Not implemented.')
        raise NotImplementedError

    def conditions_for_saving_model(self, history, min_score):
        ''' must return True or False '''
        raise NotImplementedError

    ########### evaluation method copied! ###########

    @staticmethod
    def evaluate(predictions_s, samples):
        total = 0
        correct = 0
        tp, tn, fp, fn = 0,0,0,0
        for pred, label in zip(predictions_s, samples):
            gold_pron_offset = label["p_offset"]
            pred_pron_offset = pred[0][1] if len(pred[0]) > 0 else None
            gold_pron = label["pron"]
            pred_pron = pred[0][0] if len(pred[0]) > 0 else None
            gold_both_wrong = label["is_coref_A"] == "FALSE" and label["is_coref_B"] == "FALSE"
            pred_entity_offset = pred[1][1] if len(pred[1]) > 0 else None
            pred_entity = pred[1][0] if len(pred[1]) > 0 else None
            if gold_both_wrong:
                if pred_entity is None and gold_pron_offset == pred_pron_offset and gold_pron == pred_pron:
                    correct += 1
                    tn += 1
                else:
                    fn += 1
                total += 1
            else:
                gold_entity_offset = (
                    label["offset_A"] if label["is_coref_A"] == "TRUE" else label["offset_B"]
                )
                gold_entity = (
                    label["entity_A"] if label["is_coref_A"] == "TRUE" else label["entity_B"]
                )
                if (
                    gold_pron_offset == pred_pron_offset
                    and gold_pron == pred_pron
                    and gold_entity_offset == pred_entity_offset
                    and gold_entity == pred_entity
                ):
                    correct += 1
                    tp += 1
                else:
                    fp += 1
                total += 1
        acc = float(correct) / total

        return {'accuracy':acc, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn}