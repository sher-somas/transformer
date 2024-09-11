import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split

def get_all_sentences(dataset, language):

    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    """
    "[UNK]" --> For a word that doesn't exist in the vocabulary
    "[PAD]" --> Padding
    "[EOS]" --> End of sentence
    "[SOS]" --> Start of sentence
    min_frequencey --> a word should exist atleast twice in the corpus
    """

    tokenizer_path = Path(config['tokenizer_file'].format(language))

    if not Path.exists(tokenizer_path):
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequencey=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

# Get and load the dataset
def get_dataset(config):

    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # Split data for training and validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

    