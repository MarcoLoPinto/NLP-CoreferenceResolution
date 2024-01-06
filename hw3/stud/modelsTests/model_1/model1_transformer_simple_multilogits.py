'''
    The Net part
'''

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np

import re

from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification

class Model1_net(nn.Module):
    """ entity resolution part """
    def __init__(self, fine_tune_transformer = False, hparams = {}, loss_fn = None, load_transformer_config = False):
        """entity resolution part

        Args:
            fine_tune_transformer (bool, optional): if the transformer weights needs to be fine-tuned or freezed. Defaults to False.
            hparams (any, optional): Parameters necessary to initialize the model. It can be either a dictionary or the path string for the file. Defaults to {}.
            loss_fn (any, optional): Loss function. Defaults to None.
            load_transformer_config (bool, optional): if the model needs to be entirely loaded or just the configuration (used to speed-up the loading process in the evaluation part). Defaults to False.
        """
        super().__init__()

        hparams = hparams if type(hparams) != str else self._load_hparams(hparams)

        self.n_labels = 1 # binary problem
        
        # layers:

        if load_transformer_config:
            config = AutoConfig.from_pretrained(hparams['transformer_name'])
            self.transformer_model = AutoModel.from_config(config)
            self.transformer_model.config.output_hidden_states = True
            self.transformer_model.config.output_attentions = False
        else:
            self.transformer_model = AutoModel.from_pretrained(
                hparams['transformer_name'], output_hidden_states=True, output_attentions=False
            )
        if not fine_tune_transformer:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        # RESIZING BECAUSE OF TOKENIZER!
        if hparams['resize_token_embeddings'] == True:
            self.transformer_model.resize_token_embeddings(hparams['token_embeddings_len'])

        transformer_out_dim = self.transformer_model.config.hidden_size

        # m2o_lstm_hidden_size = transformer_out_dim // 2
        # m2o_lstm_bidirectional = True
        # m2o_lstm_num_layers = 3
        # m2o_lstm_dropout = 0.3

        # self.m2o_lstm = nn.LSTM(
        #     input_size = transformer_out_dim,
        #     hidden_size = m2o_lstm_hidden_size,

        #     bidirectional = m2o_lstm_bidirectional,
        #     num_layers = m2o_lstm_num_layers,
        #     dropout = m2o_lstm_dropout if m2o_lstm_num_layers > 1 else 0,
        #     batch_first = True,
        # )

        # m2o_lstm_out = (1 + 1*m2o_lstm_bidirectional)*m2o_lstm_hidden_size

        self.dropout = nn.Dropout(0.2)

        # self.fc1 = nn.Linear(transformer_out_dim * 2, transformer_out_dim)
        # self.ln1 = nn.LayerNorm(transformer_out_dim)
        # self.relu = nn.ReLU()

        self.classifier = nn.Linear(transformer_out_dim, self.n_labels)

        self.sigmoid = nn.Sigmoid()

        # Loss function:
        self.loss_fn = loss_fn
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        predicted_pronouns = None,
        token_type_ids = None,
    ):
        """forward function

        Args:
            input_ids (torch.Tensor): the tensors inputs generated via the Tokenizer
            attention_mask (torch.Tensor): attention mask generated via the Tokenizer
            token_type_ids (any, optional): generated via the Tokenizer. Defaults to None.

        Returns:
            torch.Tensor: the logits
        """

        transformer_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if token_type_ids is None: # some transformer models don't have it
            transformer_kwargs['token_type_ids'] = token_type_ids

        transformer_outs = self.transformer_model(**transformer_kwargs)

        # summing all the considered dimensions
        transformer_out = torch.stack(
            transformer_outs.hidden_states[-1:],
            dim=0).sum(dim=0)
        # transformer_out = self.dropout(transformer_out)
        
        # transformer_out = transformer_outs.pooler_output

        # transformer_out = self.fc1(transformer_out)
        # transformer_out = self.ln1(transformer_out)
        # transformer_out = self.relu(transformer_out)

        if predicted_pronouns is not None:
            transformer_out = transformer_out * predicted_pronouns.unsqueeze(-1)

        logits = self.sigmoid( self.classifier(transformer_out) ).squeeze(-1)

        return logits # (batch, sentence_len)

    def compute_loss(self, x, y_true):
        """computes the loss for the net

        Args:
            x (torch.Tensor): The predictions
            y_true (torch.Tensor): The true labels

        Returns:
            any: the loss
        """
        if self.loss_fn is None:
            return None
        return self.loss_fn(x, y_true)

    def get_indices(self, torch_outputs):
        """

        Args:
            torch_outputs (torch.Tensor): a Tensor with shape (batch_size, sentence_len, label_vocab_size) containing the output of the net
        
        Returns:
            The method returns the position for each element
        """
        indices = torch_outputs.squeeze(-1).argmax(-1) # resulting shape = (batch_size)
        return indices

    def get_scores_and_indices(self, torch_outputs):
        """

        Args:
            torch_outputs (torch.Tensor): a Tensor with shape (batch_size, sentence_len, label_vocab_size) containing the output of the net
        
        Returns:
            The method returns the scores and indices for each element
        """
        scores_and_indices = torch_outputs.squeeze(-1).max(-1) # resulting shape = (batch_size)
        return scores_and_indices.values, scores_and_indices.indices
    
    def load_weights(self, path, strict = True):
        """load the weights of the model

        Args:
            path (str): path to the saved weights
            strict (bool, optional): Strict parameter for the torch.load() function. Defaults to True.
        """
        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device), strict=strict)
        self.eval()
    
    def save_weights(self, path):
        """save the weights of the model

        Args:
            path (str): path to save the weights
        """
        torch.save(self.state_dict(), path)

    def _load_hparams(self, hparams):
        """loads the hparams from the file

        Args:
            hparams (str): the hparams path

        Returns:
            dict: the loaded hparams
        """
        return np.load(hparams, allow_pickle=True).tolist()

    def get_device(self):
        """get the device where the model is

        Returns:
            str: the device ('cpu' or 'cuda')
        """
        return next(self.parameters()).device


'''
    The model
'''

try:
    from .modelsTests.dataset.PronDataset_transformer_simple import PronDataset_transformer_simple
    from .modelsTests.utils.Trainer_model1_transformer_simple_multilogits import Trainer_model1_transformer_simple_multilogits
except: # notebooks
    from stud.modelsTests.dataset.PronDataset_transformer_simple import PronDataset_transformer_simple
    from stud.modelsTests.utils.Trainer_model1_transformer_simple_multilogits import Trainer_model1_transformer_simple_multilogits


import os

class Model1():
    def __init__(
        self, 
        device = None,
        root_path = '../../../../',
        model_save_file_path = None,
        model_load_weights = True,
        loss_fn = None,
        fine_tune_transformer = False,
        saves_path_folder = 'test3',
        tokenizer = None,
    ):
        """the wrapper model for model1

        Args:
            device (str, optional): the device in which the model needs to be loaded. Defaults to None.
            root_path (str, optional): root of the current environment. Defaults to '../../../../'.
            model_save_file_path (str, optional): the path to the weights of the model. Defaults to None.
            model_load_weights (bool, optional): if the model needs to load the weights. Defaults to True.
            loss_fn (any, optional): the loss function. Defaults to None.
            fine_tune_transformer (bool, optional): If the transformer needs to be fine-tuned. Defaults to False.
            saves_path_folder (str, optional): the path to the saves folder (for the parameters and other possible weights). Defaults to 'test3'.
            tokenizer (any, optional): the tokenizer to use for the model. Defaults to None.
        """

        self.trainer = Trainer_model1_transformer_simple_multilogits()

        # root:

        saves_path = os.path.join(root_path, f'model/{saves_path_folder}/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        # define principal paths:

        model_name = 'model1_weights_transformer_simple_multilogits.pth'

        model_save_file_path = os.path.join(saves_path,model_name) if model_save_file_path is None else model_save_file_path

        self.model = Model1_net( 
            hparams = self.hparams,
            loss_fn = loss_fn,
            fine_tune_transformer = fine_tune_transformer,
            load_transformer_config = model_load_weights,
        )

        if model_load_weights:
            self.model.load_weights(model_save_file_path)

        self.device = self.model.get_device() if device is None else device
        self.model.to(device)

        self.model.eval()
        
        # load vocabs:

        self.dataset = PronDataset_transformer_simple(
            data_path = None,
            tokenizer = self.hparams['transformer_name'] if tokenizer is None else tokenizer,
        )
        self.dataset.collate_fn = self.dataset.create_collate_fn()

    def predict(self, sentence):
        """predict a sentence output

        Args:
            sentence (dict): sentence input

        Returns:
            dict: the formatted output from the pipeline
        """
        result = deepcopy(sentence)

        # convert to ids
        sentences_each_coref = self.dataset.collate_fn([sentence])
        with torch.no_grad():
            dict_out = self.trainer.compute_forward(self.model, sentences_each_coref, self.device, optimizer = None)
        
        predicted_p_index = dict_out['predictions'].argmax(-1).detach().cpu().tolist()[0]
        input_ids = sentences_each_coref['input_ids'].detach().cpu().tolist()[0]

        ambiguous_pronoun_index = predicted_p_index
        ambiguous_pronoun = self.dataset.tokenizer.decode(input_ids[ambiguous_pronoun_index])
        ambiguous_pronoun_offset = len( self.dataset.tokenizer.decode( input_ids[1:ambiguous_pronoun_index] ) )
        if 'text' in sentence:
            text = sentence['text']
            try:
                margin_error = 12
                subtext = text[ambiguous_pronoun_offset-margin_error:]

                regexed = r"" + "\W(" + ambiguous_pronoun + ")\W"
                resex = re.search(regexed, subtext).start()+1

                final_offset = len(text[:ambiguous_pronoun_offset-margin_error]) + resex
                ambiguous_pronoun_offset = final_offset
            except:
                pass

        result['pron'] = ambiguous_pronoun
        result['p_offset'] = ambiguous_pronoun_offset

        return result
