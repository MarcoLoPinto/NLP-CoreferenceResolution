import torch
from torch.utils.data import DataLoader

try:
    from .Trainer_model3 import Trainer_model3
except: # notebooks
    from stud.modelsTests.utils.Trainer_model3 import Trainer_model3

class Trainer_model1_transformer_simple_multilogits(Trainer_model3):

    def __init__(self):
        super().__init__()

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        # inputs:
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        token_type_ids = sample['token_type_ids'].to(device) if 'token_type_ids' in sample else None # some transformer models don't have it

        # other:
        predicted_pronouns = sample['possible_pronoun_ids'].to(device)

        # infos useful for output:
        words_ids = sample['words_ids']

        # outputs:
        labels = sample['gold_pronoun'].to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        
        predictions = model.forward(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,

            predicted_pronouns = predicted_pronouns
        )

        # predictions_flattened = predictions.reshape(-1, predictions.shape[-1]) 
        # labels_flattened = labels.view(-1)

        # predictions_flattened = predictions_flattened.to(device)
        # labels_flattened = labels_flattened.to(device)

        if model.loss_fn is not None:
            sample_loss = model.compute_loss(predictions, labels)
        else:
            sample_loss = None

        if optimizer is not None:
            sample_loss.backward()
            optimizer.step()

        return {
            'labels':labels, 
            'predictions':predictions, 
            'loss':sample_loss,
            'words_ids':words_ids,
        }

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        val_labels, val_predictions, valid_loss = self.compute_predictions(final_model.model, valid_dataloader, device)
        val_labels, val_predictions = self.compute_final_predictions(final_model, valid_dataloader.dataset.data_raw, device)
        return {'labels': val_labels, 'predictions': val_predictions, 'loss': valid_loss}

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        total = 0
        correct = 0
        offset_errors = 0
        for pred, label in zip(predictions, labels):
            gold_pron_offset = label["p_offset"]
            pred_pron_offset = pred["p_offset"]
            gold_pron = label["pron"]
            pred_pron = pred["pron"]

            if gold_pron_offset == pred_pron_offset and gold_pron == pred_pron:
                correct += 1
            if gold_pron_offset != pred_pron_offset and gold_pron == pred_pron:
                offset_errors += 1
            total += 1
            
        acc = float(correct) / total

        if offset_errors > 0:
            print(f"WARN: there are {offset_errors} offset errors, out of {correct} correct in {total} total!")

        return {'accuracy':acc}

    #################### Extras ####################

    def compute_predictions(self, model, valid_dataloader, device):
        valid_loss = 0.0
        labels = {}
        predictions = {}

        model.eval()
        model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataloader):
                dict_out = self.compute_forward(model, sample, device, optimizer = None)

                valid_loss += dict_out['loss'].tolist() if dict_out['loss'] is not None else 0

        return labels, predictions, (valid_loss / len(valid_dataloader))

    def compute_final_predictions(self, final_model, valid_dataset_raw, device):
        labels = []
        predictions = []

        final_model.model.eval()
        final_model.model.to(device)
        with torch.no_grad():
            for step, sample in enumerate(valid_dataset_raw):

                labels.append(sample)

                predictions.append( final_model.predict(sample) )

        return labels, predictions