# NLP-Transformer
This repository contains Pytorch implementation of the original NLP transformer from the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf). The model is trained on [IWSLT](https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/iwslt.html) dataset for English -> German translation which contains ~200k sentence pairs collected mainly from [TEDx talks](https://www.ted.com/) and [QED corpus](https://alt.qcri.org/resources/qedcorpus/). 

## Setup the environment
Download it using ```git clone``` command and navigate into the main directory:
```python
git clone https://github.com/djansr1408/nlp-transformer.git
cd nlp-transformer
```
Install Python environment:

```python
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Create ```.env``` file which contains environment variables such as ```BASE_DIR``` and ```DATASET_DIR```.
```python
BASE_DIR="./path_to_the_repository/nlp-transformer"
DATASET_DIR="./path_to_the_repository/nlp-transformer/data/.data"  # Here it is already inside the base directory (but doesn't necessary needs to be)
```

Create ```config.json``` file which contains parameters of the model, as well as training parameters:
```python
{
    "model_alias": "transformer_2",
    "batch_size": 16,   
    "load_cached": true, 
    "embedding_size": 512, 
    "num_parts_encoder": 6, 
    "num_parts_decoder": 6, 
    "d_model": 512, 
    "d_k": 64, 
    "d_v": 64, 
    "n_heads": 8, 
    "inner_layer_size": 2048, 
    "dropout": 0.1, 
    "device": "cuda", 
    "warmup_steps": 4000, 
    "num_epochs": 5, 
    "smoothing": true, 
    "smoothing_coeff": 0.1, 
    "use_xavier_init": true, 
    "log_every_n_steps": 600
}
```
```model_alias```: Name of the model specified for the specific version of the transformer that is trained.

```batch_size```: Batch size (Default: 16)

```load_cached```: Whether to use cached dataset already split to train, validation and test part (Default: true). 
```embedding_size```: Embedding size for the input (Default: 512, specified in the original paper) 
```num_parts_encoder```: 6, 
    "num_parts_decoder": 6, 
    "d_model": 512, 
    "d_k": 64, 
    "d_v": 64, 
    "n_heads": 8, 
    "inner_layer_size": 2048, 
    "dropout": 0.1, 
    "device": "cuda", 
    "warmup_steps": 4000, 
    "num_epochs": 5, 
    "smoothing": true, 
    "smoothing_coeff": 0.1, 
    "use_xavier_init": true, 
    "log_every_n_steps": 600

## Run training with the command:
```python
python train.py
```

## Run ```tensorboard``` for monitoring the train and validation loss.
```python
tensorboard --logidr=./runs
```
