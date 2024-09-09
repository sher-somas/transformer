import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int) -> None:

        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.model_dimension)
    
class PositionalEmbedding(nn.Module):

    def __init__(self, model_dimension: int, sequence_length, dropout: float) -> None:
        """
        sequence_length: Maximum length of the sentence.

        """

        super().__init__()
        self.model_dimension = model_dimension
        self.sequence_length = sequence_length
        self.dropout = dropout