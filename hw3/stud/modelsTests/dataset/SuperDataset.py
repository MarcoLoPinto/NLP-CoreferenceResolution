from torch.utils.data import Dataset

import csv
import torch
import numpy as np

class SuperDataset(Dataset):
    """ This class is needed in order to read and work with data for this homework """
    def __init__(self, data_path:str = None):
        """The dataset class used for the homework

        Args:
            data_path (str): the path to the dataset file
        """

        self.data = SuperDataset.read_dataset(data_path) if data_path is not None else None

    @staticmethod
    def read_dataset(file_path):
        """generates the data from a file path

        Args:
            file_path (str): the path to the dataset

        Returns:
            list: a list of samples
        """
        data = []
        with open(file_path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for sample in tsv_file:
                # E2E               (model123)  = text
                # Entity Iden + Res (model23)   = text, pronoun
                # Entity Res        (model3)    = text, pronoun and candidates
                if sample[3] == 'Pronoun-offset':
                    continue
                data_row = {
                    'id': sample[0], 
                    'text': sample[1], 
                    'pron': sample[2], 'p_offset': int(sample[3]), 
                    'entity_A': sample[4], 'offset_A': int(sample[5]), 'is_coref_A': sample[6], # candidate 1
                    'entity_B': sample[7], 'offset_B': int(sample[8]), 'is_coref_B': sample[9], # candidate 2
                    # 'url': sample[10], # not useful
                }
                data.append(data_row)
                # the input for the wrapped model: a sentence dictionary with infos
                # the output for each sentence (so it's a list of these things):
                # (
                #   ('her',     274), # this is the ambiguos pronoun identified
                #   ('Pauline', 418), # this is the coreferent selected
                # )
                # if the entities are both FALSE, then the answer is: 
                # (
                #   ('her',     274), # this is the ambiguos pronoun identified
                #   (None,      None),# no pronouns!
                # )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def save_dict(file_path, dict_values):
        """saves the variable in a file

        Args:
            file_path (str): save file path
            dict_values (any): the value
        """
        np.save(file_path, dict_values)

    @staticmethod
    def load_dict(file_path):
        """returns the loaded variable

        Args:
            file_path (str): saved file path

        Returns:
            any: the variable
        """
        return np.load(file_path, allow_pickle=True).tolist()

