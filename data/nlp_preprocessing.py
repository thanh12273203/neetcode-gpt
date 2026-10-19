import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        strings = positive + negative
        words = set()

        for sentence in strings:
            for word in sentence.split():
                words.add(word)

        all_words = sorted(words)
        vocab = {word: i + 1 for i, word in enumerate(all_words)}
        
        encoded_string = [torch.tensor([vocab[word] for word in sentence.split()], dtype=int) for sentence in strings]
        return torch.nn.utils.rnn.pad_sequence(encoded_string, batch_first=True)
        pass
