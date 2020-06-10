import numpy as np
from transformers import *
import tokenizers
import re
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


from constants import TYPE2MODEL, TOKENIZERS, MODELS, WITH_THIRD_PARAM 

PUNCT_PATTERN = "[.,/|\!?\{\}\(\)\[\]]"


class BaseModel:
    def __init__(self, model, with_third_param=True):
        self.model = model
        self.with_third_param = with_third_param

    def __call__(self, ids, attention_mask, token_type_ids):
        if self.with_third_param:
            return self.model(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.model(ids, attention_mask=attention_mask)
       

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_shifts()
    
    def preprocessing(self, text):
        text = text.lower()
#         text = re.sub(PUNCT_PATTERN, "", text)
#         if text == "":
#             text = "#"
#         text = wordnet_lemmatizer.lemmatize(text, pos = "v")
        return text

    def encode(self, text):
        text = self.preprocessing(text)
        return self.tokenizer.encode(text)[self.l:-self.r]

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def set_shifts(self):
        tokens1 = self.tokenizer.encode("1")
        tokens2 = self.tokenizer.encode("2")

        l, r = 1, 1
        while np.all(tokens1[:l] == tokens2[:l]): l += 1
        while np.all(tokens1[-r:] == tokens2[-r:]): r += 1
        self.l, self.r = l-1, r-1


def get_base_model(params):
    for type_model in TYPE2MODEL.keys():
        for model in TYPE2MODEL[type_model]:
            if model == params["base_model"]:
                print(model)
                tokenizer = Tokenizer(TOKENIZERS[type_model].from_pretrained(model))
                base_model =  BaseModel(
                    MODELS[type_model].from_pretrained(model),
                    with_third_param=WITH_THIRD_PARAM[type_model]
                )
                return tokenizer, base_model
    assert False, "unknown base_model param"

        
        
        
        
        
        
        
        