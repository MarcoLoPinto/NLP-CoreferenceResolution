import torch
from torch.utils.data import DataLoader
from seqeval.metrics import accuracy_score, f1_score

try:
    from .Trainer_model2 import Trainer_model2
    from ..dataset.NERDataset_transformer_simple import NERDataset_transformer_simple
except: # notebooks
    from stud.modelsTests.utils.Trainer_model2 import Trainer_model2
    from stud.modelsTests.dataset.NERDataset_transformer_simple import NERDataset_transformer_simple

class Trainer_model2_transformer_simple_multilogits(Trainer_model2):

    def __init__(self):
        super().__init__()

    def compute_forward(self, model, sample, device, optimizer = None):
        ''' must return a dictionary with "loss" key in it '''
        # inputs:
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        token_type_ids = sample['token_type_ids'].to(device) if 'token_type_ids' in sample else None # some transformer models don't have it

        # infos useful for output:
        output_mask = sample['output_mask']

        # outputs:
        labels = sample['ner_tags_formatted'].to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        
        predictions = model.forward(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        predictions_flattened = predictions.reshape(-1, predictions.shape[-1]) 
        labels_flattened = labels.view(-1)

        predictions_flattened = predictions_flattened.to(device)
        labels_flattened = labels_flattened.to(device)

        if (model.loss_fn is not None) and (len(labels) > 0):
            sample_loss = model.compute_loss(predictions_flattened, labels_flattened)
        else:
            sample_loss = None

        if optimizer is not None:
            sample_loss.backward()
            optimizer.step()

        return {
            'labels':labels, 
            'predictions':predictions, 
            'loss':sample_loss,
            'output_mask':output_mask,
        }

    def compute_validation(self, final_model, valid_dataloader, device):
        ''' must return a dictionary with "labels", "predictions" and "loss" keys '''
        val_labels, val_predictions, valid_loss = self.compute_predictions(final_model.model, valid_dataloader, device)
        val_labels, val_predictions = self.compute_final_predictions(final_model, valid_dataloader.dataset.data_raw, device)
        return {'labels': val_labels, 'predictions': val_predictions, 'loss': valid_loss}

    def compute_evaluations(self, labels, predictions):
        ''' must return a dictionary of results '''
        evaluations_results = {}
        f1_s = f1_score(labels, predictions, average="macro")
        evaluations_results['f1'] = f1_s

        return evaluations_results

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

                label = sample['ner_tags']
                prediction = final_model.predict(sample)['ner']

                try:
                    assert len(label) == len(prediction)
                except:
                    raise Exception(step, len(label), len(prediction))

                labels.append(label)
                predictions.append(prediction)

        return labels, predictions