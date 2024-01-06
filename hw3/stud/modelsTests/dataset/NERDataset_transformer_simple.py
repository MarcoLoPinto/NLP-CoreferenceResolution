from torch.utils.data import Dataset

import os
import torch
import numpy as np
import re
from copy import deepcopy

from transformers import AutoTokenizer

class NERDataset_transformer_simple(Dataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, tokenizer = None):
        """The dataset class used for the homework

        Args:
            data_path (str): the path to the dataset file
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None
        """

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer

        self.id_to_ner = ['O','B','I']
        self.ner_to_id = {'O':0,'B':1,'I':2}
            
        self.data = NERDataset_transformer_simple.read_dataset(data_path) if data_path is not None else None

        if self.data is not None:
            self.data_raw = deepcopy(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    @staticmethod
    def read_dataset(file_path, max_sentences = 40_000, min_length = 16, max_length = 50):
        """generates the data from a file path

        Args:
            file_path (str): the path to the dataset
            max_sentences (int, optional): Max number of samples. Defaults to 40_000.
            min_length (int, optional): Minimum words for each sample. Defaults to 16.
            max_length (int, optional): Maximum words for each sample. Defaults to 50.

        Returns:
            list: a list of samples
        """
        data = []
        sentences = 0
        allowed_ners = {'B-PER':'B','I-PER':'I'}

        if os.path.basename(file_path) == 'conllpp_dev.txt':
            max_sentences = max_sentences // 3

        with open(file_path, 'r') as file:

            sample = {'words':[], 'ner_tags':[]}

            for row in file:
                line = row.rstrip().split(' ')
                if len(line) <= 1 or line[0] == '-DOCSTART-':
                    if (len(sample['words']) > min_length and len(sample['words']) <= max_length) and not all([e == 'O' for e in sample['ner_tags']]):
                        sample_copy = deepcopy(sample)
                        sample_copy['text'] = ' '.join(sample['words'])
                        data.append(sample_copy)
                    sample = {'words':[], 'ner_tags':[]}
                    continue
                    
                [word, pos, _, ner_tag] = line

                if ner_tag in allowed_ners:
                    ner_tag = allowed_ners[ner_tag]
                else:
                    ner_tag = 'O'

                sample['words'].append(word); sample['ner_tags'].append(ner_tag)

                sentences += 1
                if sentences >= max_sentences:
                    break
        return data

    def get_tok_embedding_ids(self, word):
        return self.tokenizer(word)['input_ids'][1:-1]

    def create_collate_fn(self):
        """ The collate_fn parameter for torch's DataLoader """
        def collate_fn(batch):
            """ each sentence in the pre-processed batch has: 
                'id', 'text',
                'pron', 'p_offset',
            """

            has_ner = len(batch) > 0 and 'ner_tags' in batch[0]

            batch_formatted = {}

            sentences = []
            ner_tags = []
            for sentence in batch:
                if has_ner:
                    sentences.append(sentence['words'])
                    ner_tags.append([self.ner_to_id[e] for e in sentence['ner_tags']])
                else:
                    sentences.append(sentence['text'])
                
            batch_formatted = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                is_split_into_words=has_ner,
            )

            batch_formatted['ner_tags'] = ner_tags

            ner_tags_formatted = []
            output_mask = []
            
            for i, sample in enumerate(batch):
                words_ids = batch_formatted.word_ids(batch_index=i)
                
                previous_word_id = None
                nones_seen = 0
                ner_formatted_row = []
                output_mask_row = []

                for j, word_id in enumerate(words_ids):

                    if word_id is None: # Special tokens have the id = None. Setting the label to -1 so they are ignored by the loss
                        if has_ner:
                            ner_formatted_row.append(-1)
                        output_mask_row.append(0)
                        nones_seen += 1

                    elif word_id != previous_word_id: # if differs, it's not a subword of the previous word
                        if has_ner:
                            if word_id < len(ner_tags[i]) and nones_seen <= 1:
                                ner_formatted_row.append( ner_tags[i][word_id] )
                            else:
                                ner_formatted_row.append(-1)

                        output_mask_row.append(1 if nones_seen <= 1 else 0)

                    else: # if it's a subword!
                        if has_ner:
                            ner_formatted_row.append(-1)
                        output_mask_row.append(0)

                    previous_word_id = word_id

                ner_tags_formatted.append(ner_formatted_row)
                output_mask.append(output_mask_row)

            batch_formatted['ner_tags_formatted'] = torch.as_tensor(ner_tags_formatted)
            batch_formatted['output_mask'] = output_mask

            return batch_formatted
    
        return collate_fn