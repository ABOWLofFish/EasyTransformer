import collections
import re
class Vocab:
    def __init__(self, sentences=None, min_freq=0, reserved_tokens=None):
        '''根据出现频率排序'''
        tokens = [token for item in sentences for token in re.split(r" |\n", item) if token != '']
        counter = collections.Counter(tokens)
        token_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        '''<unk>索引为0,<pad>索引1'''
        self.unk = 0
        self.pad = 1
        tokens = ['<unk>', '<pad>']
        if reserved_tokens is not None:
            tokens += reserved_tokens
        tokens += [tur[0] for tur in token_freq if tur[1] >= min_freq]
        self.unique_tokens = dict()
        self.reversed_unique_tokens = dict()

        for i in range(len(tokens)):
            self.unique_tokens[tokens[i]] = i
        self.reversed_unique_tokens = self.unique_tokens.__reversed__()
        print("vocab size:",len(self.unique_tokens))

    def __len__(self):
        return len(self.unique_tokens)

    def to_idx(self, words, maxLen):
        """list[tokens] to list[idx]"""
        if not isinstance(words, (list, tuple)):
            return self.unique_tokens.get(words, self.unk)
        seq = [self.to_idx(token, maxLen) for token in words]
        return seq + (maxLen - len(seq)) * [self.pad] if len(seq) < maxLen else seq[:maxLen]


    def to_tokens(self, indices):
        """list[idx] to list[tokens]"""
        if not isinstance(indices, (list, tuple)):
            return self.reversed_unique_tokens[indices]
        return [self.reversed_unique_tokens[index] for index in indices]

