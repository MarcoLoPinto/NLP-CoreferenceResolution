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

class Dataset_transformer_simple_multilogits(SuperDataset):
    """
    This class is needed in order to read and work with data for this homework
    """
    def __init__(self, data_path:str = None, tokenizer = None, mention_tags = None):
        """The dataset class used for the homework

        Args:
            data_path (str): the path to the dataset file
            tokenizer (any, optional): the tokenizer used for the model, either the name (string) or the tokenizer instance. Defaults to None
            mention_tags (any, optional): the custom tokenizer tags for the model. Defaults to None
        """
        super().__init__(data_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if type(tokenizer) == str else tokenizer

        self.mention_tags = deepcopy(mention_tags)

        if not any('_id' in vvv for vvv in self.mention_tags):
            mtl = []
            for e in self.mention_tags.values():
                mtl.append(e) if e not in mtl else None

            self.tokenizer.add_special_tokens( { 'additional_special_tokens': mtl } ) # do this for the model: model.resize_token_embeddings(len(tokenizer))
            for k,v in deepcopy(self.mention_tags).items():
                self.mention_tags[k + '_id'] = self.tokenizer.convert_tokens_to_ids(v)
            
        self.split_by_entities = False
        if self.data is not None:
            self.data_raw = deepcopy(self.data)
            self.data = Dataset_transformer_simple_multilogits.format_by_entities(self.data, False)

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
            batch = Dataset_transformer_simple_multilogits.format_by_entities(batch, False)

            batch_formatted = {}

            sentences = []
            mentions = []
            for sentence in batch:
                text = sentence['text']
                sentences.append(text)
                for e in sentence['entities']:
                    text = self.add_tag_to_word_in_phrase(  text, self.mention_tags['e_open'], 
                                                            self.mention_tags['e_close'], 
                                                            e['offset'], e['entity'])
                mention = f"{sentence['pron']}"
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

            batch_formatted['predicted_entity_ids'] = []
            for sentence in batch:
                pred_entity_ids_row = {}
                for e in sentence['entities']:
                    pred_entity_ids_row[ self.get_tok_embedding_ids(e['entity'])[0] ] = {'name':e['entity'], 'offset':e['offset']}
                batch_formatted['predicted_entity_ids'].append( pred_entity_ids_row )

            error_margin = 5

            batch_formatted['predicted_entities'] = []
            for i, input_ids_sentence in enumerate(batch_formatted['input_ids']):
                predicted_entities_row = []
                for j,v in enumerate(input_ids_sentence):
                    res = 0.
                    if (v in list(batch_formatted['predicted_entity_ids'][i].keys())):
                        v_offset = batch_formatted['predicted_entity_ids'][i][v.tolist()]['offset']
                        v_idx = len(self.tokenizer.encode(batch[i]['text'][:v_offset-1]))
                        if (j >= v_idx - error_margin) and (j <= v_idx + error_margin):
                            res = 1.
                    predicted_entities_row.append(res)
                batch_formatted['predicted_entities'].append(predicted_entities_row)
            batch_formatted['predicted_entities'] = torch.as_tensor(batch_formatted['predicted_entities'])

            if 'is_coref' in batch[0]['entities'][0]:

                batch_formatted['gold_entity_ids'] = []
                for sentence in batch:
                    gold_entity_ids_row = {}
                    for e in sentence['entities']:
                        if e['is_coref'] == 'TRUE':
                            gold_entity_ids_row[ self.get_tok_embedding_ids(e['entity'])[0] ] = {'name':e['entity'], 'offset':e['offset']}
                    batch_formatted['gold_entity_ids'].append( gold_entity_ids_row )

                batch_formatted['gold_entities'] = []
                for i, input_ids_sentence in enumerate(batch_formatted['input_ids']):
                    gold_entities_row = []
                    for j,v in enumerate(input_ids_sentence):
                        res = 0.
                        if (v in list(batch_formatted['gold_entity_ids'][i].keys())):
                            v_offset = batch_formatted['gold_entity_ids'][i][v.tolist()]['offset']
                            v_idx = len(self.tokenizer.encode(batch[i]['text'][:v_offset-1]))
                            if (j >= v_idx - error_margin) and (j <= v_idx + error_margin):
                                res = 1.
                        gold_entities_row.append(res)
                    batch_formatted['gold_entities'].append(gold_entities_row)
                batch_formatted['gold_entities'] = torch.as_tensor(batch_formatted['gold_entities'])

            else:

                batch_formatted['gold_entities'] = torch.as_tensor([]*len(batch))


            return batch_formatted
    
        return collate_fn