try:
    from .modelsTests.model_1.model1_transformer_simple_multilogits import Model1
    from .modelsTests.model_2.model2_transformer_simple_multilogits import Model2
    from .modelsTests.model_3.model3_transformer_simple_multilogits import Model3
except: # notebooks
    from stud.modelsTests.model_1.model1_transformer_simple_multilogits import Model1
    from stud.modelsTests.model_2.model2_transformer_simple_multilogits import Model2
    from stud.modelsTests.model_3.model3_transformer_simple_multilogits import Model3

from transformers import AutoTokenizer

import os
import numpy as np

class FinalModel():

    def __init__(
        self, 
        device,
        model_type: int = 0,
        root_path = '../../../../',
        tokenizer = None,
        use_ner_model = True,
    ):
        """the wrapper model for all the steps

        Args:
            device (str, optional): the device in which the model needs to be loaded. Defaults to None.
            model_type (int, optional): the type of model to be initialized. [ 0 = model3, 1 = model23, 2 = model123 ].
            root_path (str, optional): root of the current environment. Defaults to '../../../../'. 
            tokenizer (any, optional): The optional instance of the tokenizer. Defaults to None.
            use_ner_model (bool, optional): if the model to be used for Entity Identification is the NER transformer-based or only the Entity Resolution part. Defaluts to True.
            tokenizer (any, optional): the tokenizer to use for each model. Defaults to None.
        """

        # root:

        saves_path = os.path.join(root_path, f'model/test2/')

        # load hparams:

        self.hparams = np.load(
            os.path.join(saves_path,'global_params.npy'), 
            allow_pickle=True
        ).tolist()

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams['transformer_name'])  

        if model_type not in [0,1,2]:
            raise Exception('0 = model3, 1 = model23, 2 = model123')

        if model_type > 0:
            if model_type == 2:
                self.model1 = Model1(device=device,root_path=root_path,tokenizer=tokenizer)
            else:
                self.model1 = None
            if use_ner_model:
                self.model2 = Model2(device=device,root_path=root_path,tokenizer=tokenizer)
                self.model3 = Model3(device=device,root_path=root_path,tokenizer=tokenizer)
            else:
                self.model2 = None
                self.model3 = Model3(device=device,root_path=root_path,tokenizer=tokenizer, use_entities=False)
        else:
            self.model3 = Model3(device=device,root_path=root_path,tokenizer=tokenizer)
            self.model2 = None
            self.model1 = None

        

    def predict(self, sentence):
        """predict a sentence output

        Args:
            sentence (dict): sentence input

        Returns:
            dict: the formatted output from the pipeline
        """

        result = sentence

        if self.model1 is not None:
            result = self.model1.predict(result)
        if self.model2 is not None:
            result = self.model2.predict(result)
        result = self.model3.predict(result)

        return result