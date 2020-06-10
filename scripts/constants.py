from transformers import *

SEED = 42
N_SPLIT = 5
N_EPOCH = 5
BATCH_SIZE = 8

PAD_ID = 1
MAX_LEN = 113

LEFT_PAD_LEN = 1

N_TRAIN = 27481
N_TEST = 3534

SENTIMENT_ID = {'positive': 1313, 'negative': 2430, 'neutral': 7974}


TYPE2MODEL = {
    "bert" : [
        "bert-base-chinese",
        "bert-base-uncased",
        "bert-base-uncased",
        "bert-base-finnish-uncased-v1",
        "bert-base-dutch-cased",
        "bert-base-cased-finetuned-mrpc",
        "bert-base-finnish-cased-v1",
        "bert-base-german-cased",
        "bert-base-cased"
    ],
    "roberta": [
        "distilroberta-base",
        "roberta-base"
    ],
    "xlnet": [
        "xlnet-base-cased"
    ],
    "xlm": [
        "xlm-mlm-ende-1024",
        "xlm-mlm-enfr-1024",
        "xlm-mlm-enro-1024",
        "xlm-clm-ende-1024",
        "xlm-clm-enfr-1024"
    ],
    "distill_bert": [
        "distilbert-base-uncased-distilled-squad",
        "distilbert-base-multilingual-cased",
        "distilbert-base-uncased",
        "distilbert-base-cased-distilled-squad",
        "distilbert-base-cased"
    ],
    "albert": [
        "albert-base-v2",
        "albert-large-v2",
        "albert-xlarge-v2",
        "albert-base-v1",
        "albert-large-v1"
    ],
    "electra": [
        "google/electra-large-discriminator",
        "google/electra-small-discriminator",
        "google/electra-small-generator",
        "google/electra-base-generator",
        "google/electra-base-discriminator",
        "google/electra-large-generator"
    ]
}

TOKENIZERS = {
    "bert": BertTokenizer,
    "roberta": RobertaTokenizer,
    "xlnet": XLNetTokenizer,
    "xlm": XLMTokenizer,
    "distill_bert": DistilBertTokenizer,
    "albert": AlbertTokenizer,
    "electra": ElectraTokenizer 
}

MODELS = {
    "bert": TFBertModel,
    "roberta": TFRobertaModel,
    "xlnet": TFXLNetModel,
    "xlm": TFXLMModel,
    "distill_bert": TFDistilBertModel,
    "albert": TFAlbertModel,
    "electra": TFElectraModel 
}

WITH_THIRD_PARAM = {
    "bert": True,
    "roberta": True,
    "xlnet": True,
    "xlm": True,
    "distill_bert": False,
    "albert": True,
    "electra": True
}