import os
from dotenv import load_dotenv
load_dotenv()

MAX_SEQ_LEN = 100

SOS_WORD = "<sos>"
EOS_WORD = "<eos>"
PAD_WORD = "<pad>"

BASE_DIR = os.getenv("BASE_DIR")
DATASET_DIR = os.getenv("DATASET_DIR")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")