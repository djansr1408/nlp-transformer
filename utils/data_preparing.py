from torchtext.data import Field, BucketIterator, Dataset
from torchtext.data.example import Example
from torchtext.data.utils import interleave_keys
import torchtext.datasets
import torch
import spacy
import os

import utils.constants as C

class LangModel:
    en = None
    de = None

"""English -> Deutsch"""
en = spacy.load("en_core_web_sm")
de = spacy.load("de_core_news_sm")

LangModel.en = en
LangModel.de = de

dataset = torchtext.datasets.IWSLT

def tokenize_src(text): 
    return [tok.text for tok in LangModel.en.tokenizer(text) if not tok.is_space]

def tokenize_trg(text): 
    return [tok.text for tok in LangModel.de.tokenizer(text) if not tok.is_space]

def filter_long_examples(batch):
    return len(batch.src) <= C.MAX_SEQ_LEN and len(batch.trg) <= C.MAX_SEQ_LEN

def get_data_loaders(dataset_path, 
                     batch_size, 
                     device, 
                     load_cached=True):
    """
        This function splits the dataset to train, val and test data loaders. Each dataloader contains batches of tokenized sentences.
    """
    src_tokenizer = Field(tokenize = tokenize_src, init_token=C.SOS_WORD, eos_token=C.EOS_WORD, pad_token=C.PAD_WORD, batch_first=True)
    trg_tokenizer = Field(tokenize = tokenize_trg, init_token=C.SOS_WORD, eos_token=C.EOS_WORD, pad_token=C.PAD_WORD, batch_first=True)
    fields = [('src', src_tokenizer), ('trg', trg_tokenizer)]

    train_cache_path = os.path.join(dataset_path, "train_cached_dataset.txt")
    val_cache_path =  os.path.join(dataset_path, "val_cached_dataset.txt")
    test_cache_path =  os.path.join(dataset_path, "test_cached_dataset.txt")

    if not load_cached:
        train_dataset, val_dataset, test_dataset = dataset.splits(root=dataset_path, exts=(".en", ".de"), 
                                                                  fields=fields, 
                                                                  filter_pred=filter_long_examples)
        save_dataset(train_cache_path, train_dataset)
        save_dataset(val_cache_path, val_dataset)
        save_dataset(test_cache_path, test_dataset)
    
    else:
        train_dataset = FastDataset(train_cache_path, fields)
        val_dataset = FastDataset(val_cache_path, fields)
        test_dataset = FastDataset(test_cache_path, fields)
            
    src_tokenizer.build_vocab(train_dataset.src, min_freq=2)
    trg_tokenizer.build_vocab(train_dataset.trg, min_freq=2)

    train_loader, val_loader, test_loader = BucketIterator.splits(datasets=(train_dataset, val_dataset, test_dataset), 
                                                                  batch_size=batch_size, 
                                                                  device=device, 
                                                                  sort_within_batch=True)
    
    return train_loader, val_loader, test_loader, src_tokenizer, trg_tokenizer


def save_dataset(output_path, dataset):
    with open(output_path, 'w') as file_:
        for ex in dataset.examples:
            file_.write(' '.join(ex.src) + '\n')
            file_.write(' '.join(ex.trg) + '\n')


class FastDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, cache_path, fields):
        with open(cache_path, 'r') as file_:
            data = [line.split() for line in file_.readlines()]
            src_tokens_data = data[0::2]
            trg_tokens_data = data[1::2]

            examples = []
            for src_tokenized, trg_tokenized in zip(src_tokens_data, trg_tokens_data):
                example = Example()
                setattr(example, 'src', src_tokenized)
                setattr(example, 'trg', trg_tokenized)
                examples.append(example)
            
            super().__init__(examples, fields)
