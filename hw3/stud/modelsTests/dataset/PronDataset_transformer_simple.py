from torch.utils.data import Dataset

import csv
import torch
import numpy as np
import re
from copy import deepcopy

try:
    from .SuperDataset import SuperDataset
except: # notebooks
    from stud.modelsTests.dataset.SuperDataset import SuperDataset

from transformers import AutoTokenizer

class PronDataset_transformer_simple(SuperDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, tokenizer = None):
        """The dataset class used for the homework

        Args:
            data_path (str): the path to the dataset file
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None
        """
        super().__init__(data_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer
            
        self.split_by_entities = False
        if self.data is not None:
            self.data_raw = deepcopy(self.data)
            self.data = PronDataset_transformer_simple.format_by_entities(self.data, False)

    @staticmethod
    def format_by_entities(data, split_by_entities):
        """Format the dataset to be used in the pipeline

        Args:
            data (list): the raw dataset
            split_by_entities (bool): If each sample needs to be splitted by each entity.

        Returns:
            list: a list of samples
        """
        data_formatted = []

        if len(data) > 0:
            for sample in data:

                if 'entity_A' in sample:
                    sample_entities = [
                        {'entity': sample['entity_A'], 'offset': int(sample['offset_A'])}, # candidate 1
                        {'entity': sample['entity_B'], 'offset': int(sample['offset_B'])}, # candidate 2
                    ]
                    if 'is_coref_A' in sample:
                        sample_entities[0]['is_coref'] = sample['is_coref_A']; sample_entities[1]['is_coref'] = sample['is_coref_B']
                elif 'entities' in sample:
                    sample_entities = sample['entities']
                else:
                    sample_entities = []

                sample_copy = deepcopy(sample)
                if 'entity_A' in sample_copy:
                    del sample_copy['entity_A']; del sample_copy['offset_A']
                    del sample_copy['entity_B']; del sample_copy['offset_B']
                if 'is_coref_A' in sample_copy:
                    del sample_copy['is_coref_A']; del sample_copy['is_coref_B']

                if split_by_entities and len(sample_entities) > 1:

                    for i, entity in enumerate(sample_entities):
                        sample_copy = deepcopy(sample_copy)
                        sample_copy['id'] = sample['id'] + '_' + str(i)
                        sample_copy['entities'] = [entity]
                        data_formatted.append(sample_copy)

                else:

                    sample_copy['entities'] = sample_entities
                    data_formatted.append(sample_copy)

        return data_formatted

    def get_tok_embedding_ids(self, word):
        """Tokenize a word

        Args:
            word (str): A word

        Returns:
            list: a Tensor list of integers
        """
        return self.tokenizer(word)['input_ids'][1:-1]

    def create_collate_fn(self):
        """ The collate_fn parameter for torch's DataLoader """
        def collate_fn(batch):
            """ each sentence in the pre-processed batch has: 
                'id', 'text',
                no 'pron' or 'p_offset' !
            """
            batch = PronDataset_transformer_simple.format_by_entities(batch, False)

            batch_formatted = {}

            sentences = []
            for sentence in batch:
                sentences.append(sentence['text'])

            batch_formatted = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                is_split_into_words=False,
            )

            batch_formatted['words_ids'] = [
                [v 
                    if (v != None)
                    else -1 
                    for j,v in enumerate(batch_formatted.word_ids(batch_index=i))
                ]
                for i in range(len(batch))
            ]

            error_margin = 5

            batch_formatted['pronoun_hypotesis_ids'] = self.get_tok_embedding_ids('His his Her her He he She she Him him')
            batch_formatted['possible_pronoun_ids'] = []
            for i, input_ids_sentence in enumerate(batch_formatted['input_ids']):
                possible_pronoun_ids_row = []
                for j,v in enumerate(input_ids_sentence):
                    res = 0.
                    if (v in list(batch_formatted['pronoun_hypotesis_ids'])):
                        res = 1.
                    possible_pronoun_ids_row.append(res)
                batch_formatted['possible_pronoun_ids'].append(possible_pronoun_ids_row)
            batch_formatted['possible_pronoun_ids'] = torch.as_tensor(batch_formatted['possible_pronoun_ids'])

            batch_formatted['gold_pronoun_id'] = []
            batch_formatted['gold_pronoun'] = []
            if len(batch[0]) > 0 and 'pron' in batch[0]:
                for sentence in batch:
                    gold_pronoun_id_row = {}
                    gold_pronoun_id_row[ self.get_tok_embedding_ids(sentence['pron'])[0] ] = {'name':sentence['pron'], 'offset':sentence['p_offset']}
                    batch_formatted['gold_pronoun_id'].append( gold_pronoun_id_row )

                for i, input_ids_sentence in enumerate(batch_formatted['input_ids']):
                    gold_pronoun_row = []
                    for j,v in enumerate(input_ids_sentence):
                        res = 0.
                        if (v in list(batch_formatted['gold_pronoun_id'][i].keys())):
                            v_offset = batch_formatted['gold_pronoun_id'][i][v.tolist()]['offset']
                            v_idx = len(self.tokenizer.encode(batch[i]['text'][:v_offset-1]))
                            if (j >= v_idx - error_margin) and (j <= v_idx + error_margin):
                                res = 1.
                        gold_pronoun_row.append(res)
                    batch_formatted['gold_pronoun'].append(gold_pronoun_row)
                batch_formatted['gold_pronoun'] = torch.as_tensor(batch_formatted['gold_pronoun'])

            return batch_formatted
    
        return collate_fn