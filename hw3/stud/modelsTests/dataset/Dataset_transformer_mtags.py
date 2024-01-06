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

class Dataset_transformer_mtags(SuperDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, tokenizer = None, mention_tags = None, split_by_entities = False):
        """The dataset class used for the homework

        Args:
            data_path (str): the path to the dataset file
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None
            mention_tags (dict, optional): mention tags used for the dataset. Defaults to None
            split_by_entities (bool, optional): If each sample needs to be splitted by each entity. Defaults to False
        """
        super().__init__(data_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer

        self.mention_tags = deepcopy(mention_tags)

        if not any('_id' in vvv for vvv in self.mention_tags):
            mtl = []
            for e in self.mention_tags.values():
                mtl.append(e) if e not in mtl else None

            self.tokenizer.add_special_tokens( { 'additional_special_tokens': mtl } ) # do this for the model: model.resize_token_embeddings(len(tokenizer))
            # self.tokenizer.add_tokens( mtl , special_tokens=True )
            for k,v in deepcopy(self.mention_tags).items():
                self.mention_tags[k + '_id'] = self.tokenizer.convert_tokens_to_ids(v)
            
        self.split_by_entities = split_by_entities
        if self.data is not None:
            self.data_raw = deepcopy(self.data)
            self.data = Dataset_transformer_mtags.format_by_entities(self.data, split_by_entities)

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

                if split_by_entities:

                    for i, entity in enumerate(sample_entities):
                        sample_copy = deepcopy(sample_copy)
                        sample_copy['id'] = sample['id'] + '_' + str(i)
                        sample_copy['entities'] = [entity]
                        data_formatted.append(sample_copy)

                else:
                    sample_copy['entities'] = sample_entities
                    data_formatted.append(sample_copy)

        return data_formatted

    def add_mention_tags_to_text(self, text, pron, p_offset, entities):
        """Add mention tags to the text

        Args:
            text (str): The input text
            pron (str): The pronoun
            p_offset (int): The pronoun offset
            entities (list): list of entities

        Returns:
            str: The formatted text
        """
        text_new = self.add_tag_to_word_in_phrase(text, self.mention_tags['p_open'], self.mention_tags['p_close'], p_offset, pron)
        for e in entities:
            text_new = self.add_tag_to_word_in_phrase(text_new, self.mention_tags['e_open'], self.mention_tags['e_close'], e['offset'], e['entity'])
        return text_new

    def add_tag_to_word_in_phrase(self, text, tag_open, tag_close, position, word):
        """Add a custom tag in the text

        Args:
            text (str): The input text
            tag_open (str): The open tag (e.g. <t>)
            tag_close (str): The close tag (e.g. </t>)
            position (int): The starting position of the word
            word (str): The word to be encapsulated

        Returns:
            str: The new formatted text
        """
        regex_subs = r"" + re.escape(tag_open) + word + re.escape(tag_close)
        text_new = text[:position] + re.sub(re.escape(word), regex_subs, text[position:],1)
        return text_new

    def create_collate_fn(self, split_by_entities = False, add_entity_tags_to_text = True, add_pron_tags_to_text = True):
        """ The collate_fn parameter for torch's DataLoader """
        def collate_fn(batch):
            """ each sentence in the pre-processed batch has: 
                'id', 'text',
                'pron', 'p_offset',
                'entity_A', 'offset_A', 'is_coref_A',
                'entity_B', 'offset_B', 'is_coref_B'
                (or entites)
            """
            batch = Dataset_transformer_mtags.format_by_entities(batch, split_by_entities)

            batch_formatted = {}

            text_with_mentions = []
            for sentence in batch:
                text = sentence['text']
                if add_pron_tags_to_text:
                    text = self.add_tag_to_word_in_phrase(  text, self.mention_tags['p_open'], self.mention_tags['p_close'], 
                                                            sentence['p_offset'], sentence['pron'])
                if add_entity_tags_to_text:
                    for e in sentence['entities']:
                        text = self.add_tag_to_word_in_phrase(  text, self.mention_tags['e_open'], self.mention_tags['e_close'], 
                                                                e['offset'], e['entity'])

                text_with_mentions.append(text)

            batch_formatted = self.tokenizer(
                text_with_mentions,
                return_tensors="pt",
                padding=True,
                is_split_into_words=False,
            )

            batch_formatted['words_ids'] = [
                [v 
                    if (v != None) and (batch_formatted['input_ids'][i][j] not in self.mention_tags.values())
                    else -1 
                    for j,v in enumerate(batch_formatted.word_ids(batch_index=i))
                ]
                for i in range(len(batch))
            ]

            
            if self.split_by_entities and len(batch) > 0 and 'is_coref' in batch[0]['entities'][0]:
                batch_formatted['binary_is_coref'] = torch.as_tensor([
                    [ 1. if sentence['entities'][i]['is_coref'] == 'TRUE' else 0. for i in range(len(sentence['entities'])) ]
                    for sentence in batch
                ])
            else:
                batch_formatted['binary_is_coref'] = torch.as_tensor([]*len(batch))

            batch_formatted['text_with_mentions'] = text_with_mentions

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