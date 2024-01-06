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

class Dataset_transformer_simple_mnli(SuperDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, split_by_entities = False):
        """The dataset class used for the homework

        Args:
            data_path (str, optional): the path to the optional dataset file. Defaults to None.
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None.
            split_by_entities (bool, optional): If each sample needs to be splitted by each entity. Defaults to False.
        """
        super().__init__(data_path)
            
        self.split_by_entities = split_by_entities
        if self.data is not None:
            self.data_raw = deepcopy(self.data)
            self.data = Dataset_transformer_simple_mnli.format_by_entities(self.data, split_by_entities)

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
            batch = Dataset_transformer_simple_mnli.format_by_entities(batch, True)

            batch_formatted = {}
            batch_formatted['sentences'] = [ sample['text'] for sample in batch ]
            batch_formatted['pronouns'] = [ sample['pron'] for sample in batch ]
            batch_formatted['entities'] = [ sample['entities'][0]['entity'] for sample in batch ]

            return batch_formatted
    
        return collate_fn