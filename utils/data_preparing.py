from torchtext.data import Field, BucketIterator
import torchtext.datasets
import torch
import spacy

import utils.constants as C

class LangModel:
    en = None
    de = None

"""English -> Deutsch"""
en = spacy.load("en_core_web_sm")
de = spacy.load("de_core_news_sm")

LangModel.en = en
LangModel.de = de

def tokenize_src(text): 
    return [tok.text for tok in LangModel.en.tokenizer(text) if not tok.is_space]

def tokenize_trg(text): 
    return [tok.text for tok in LangModel.de.tokenizer(text) if not tok.is_space]

dataset = torchtext.datasets.IWSLT

def get_data_loaders(dataset_path, device):
    src_tokenizer = Field(tokenize = tokenize_src, init_token=C.SOS_WORD, eos_token=C.EOS_WORD, pad_token=C.PAD_WORD, batch_first=True)
    trg_tokenizer = Field(tokenize = tokenize_trg, init_token=C.SOS_WORD, eos_token=C.EOS_WORD, pad_token=C.PAD_WORD, batch_first=True)

    train_dataset, val_dataset, test_dataset = dataset.splits(root=dataset_path, exts=(".en", ".de"), fields=[('src', src_tokenizer), ('trg', trg_tokenizer)])

    src_tokenizer.build_vocab(train_dataset.src, min_freq=2)
    trg_tokenizer.build_vocab(train_dataset.trg, min_freq=2)

    train_loader, val_loader, test_loader = BucketIterator.splits(datasets=(train_dataset, val_dataset, test_dataset), 
                                                        batch_size=C.BATCH_SIZE, 
                                                        device=device, 
                                                        sort_within_batch=True)
    
    return train_loader, val_loader, test_loader, src_tokenizer, trg_tokenizer



# train_loader, val_loader, test_loader, src_tokenizer, trg_tokenizer = get_data_loaders(dataset_path=".data")
# print(len(src_tokenizer.vocab))
