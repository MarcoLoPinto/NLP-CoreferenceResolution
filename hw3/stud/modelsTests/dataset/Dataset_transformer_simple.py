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

class Dataset_transformer_simple(SuperDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, tokenizer = None, split_by_entities = False):
        """The dataset class used for the homework

        Args:
            data_path (str, optional): the path to the optional dataset file. Defaults to None.
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None.
            split_by_entities (bool, optional): If each sample needs to be splitted by each entity. Defaults to False.
        """
        super().__init__(data_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer
            
        self.split_by_entities = split_by_entities
        if self.data is not None:
            self.data_raw = deepcopy(self.data)
            self.data = Dataset_transformer_simple.format_by_entities(self.data, split_by_entities)

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
                else:
                    sample_entities = sample['entities']

                sample_copy = deepcopy(sample)
                if 'entity_A' in sample_copy:
                    del sample_copy['entity_A']; del sample_copy['offset_A']
                    del sample_copy['entity_B']; del sample_copy['offset_B']
                if 'is_coref_A' in sample_copy:
                    del sample_copy['is_coref_A']; del sample_copy['is_coref_B']

                if 'p_offset' in sample_copy:
                    sample_copy['p_position'] = len(sample_copy['text'][:sample_copy['p_offset']-1].split(' '))

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
                'pron', 'p_offset',
                'entity_A', 'offset_A', 'is_coref_A',
                'entity_B', 'offset_B', 'is_coref_B'
                (or entites)
            """
            batch = Dataset_transformer_simple.format_by_entities(batch, True)

            batch_formatted = {}

            sentences = []
            mentions = []
            for sentence in batch:
                try:
                    assert len(sentence['entities']) == 1
                except:
                    raise Exception(sentence)
                for e in sentence['entities']:
                    mention = f"{sentence['pron']} {e['entity']}"
                    # mention = f"{sentence['pron']}"
                    # mention = f"{e['entity']}"
                sentences.append(sentence['text'])
                mentions.append(mention)

            batch_formatted = self.tokenizer(
                sentences, mentions,
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

            batch_formatted['pron_ids'] = [ self.get_tok_embedding_ids(sentence['pron'])[0] for sentence in batch ]
            batch_formatted['pron_mask'] = []
            for i, input_ids_sentence in enumerate(batch_formatted['input_ids']):
                pron_mask_row = [0] * len(input_ids_sentence)
                pron_mask_row[ batch_formatted['words_ids'][i].index( batch[i]['p_position'] ) ] = 1
                batch_formatted['pron_mask'].append(pron_mask_row)
            batch_formatted['pron_mask'] = torch.torch.as_tensor(batch_formatted['pron_mask'])

            batch_formatted['entity_ids'] = [ self.get_tok_embedding_ids(sentence['entities'][0]['entity'])[0] for sentence in batch ]
            batch_formatted['entity_mask'] = torch.as_tensor([ 
                [1
                    if (v == batch_formatted['entity_ids'][i]) and (batch_formatted['attention_mask'][i][j] == 1)
                    else 0
                    for j,v in enumerate(input_ids_sentence)
                ]
                for i, input_ids_sentence in enumerate(batch_formatted['input_ids'])
            ])

            # batch_formatted['pron_entity_matrix'] = torch.as_tensor([
            #     p[None,:] * e[:,None] for p,e in zip(batch_formatted['pron_mask'], batch_formatted['entity_mask'])
            # ])

            if self.split_by_entities and len(batch) > 0 and 'is_coref' in batch[0]['entities'][0]:
                batch_formatted['binary_is_coref'] = torch.as_tensor([
                    [ 1. if sentence['entities'][i]['is_coref'] == 'TRUE' else 0. for i in range(len(sentence['entities'])) ]
                    for sentence in batch
                ])
            else:
                batch_formatted['binary_is_coref'] = torch.as_tensor([]*len(batch))

            # batch_formatted['text_with_mentions'] = ...

            # batch_pron = [ sentence['pron'] for sentence in batch ]
            # batch_p_offset = [ sentence['p_offset'] for sentence in batch ]

            # b, idx = 0, 0

            # while b < len(batch_formatted['words_ids']):
            #     sample_words_ids = batch_formatted['words_ids'][b]

            #     while idx < len(sample_words_ids):
            #         word_id = sample_words_ids[idx]

            #         if word_id == -1 or batch_formatted['input_ids'][b][idx] in self.mention_tags.values():
            #             pass

            #         idx+=1

            #     b+=1


            return batch_formatted
    
        return collate_fn