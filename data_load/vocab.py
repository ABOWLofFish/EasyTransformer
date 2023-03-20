import collections
import re
class Vocab:
    def __init__(self, sentences=None, min_freq=0, reserved_tokens=None):
        '''根据出现频率排序'''
        tokens = [token for item in sentences for token in re.split(r" |\n", item) if token != '']
        counter = collections.Counter(tokens)
        self.token_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        '''未知标记的索引为0'''
        self.unk = 0
        if reserved_tokens is not None:
            tokens = ['<unk>']+reserved_tokens
        tokens += [tur[0] for tur in self.token_freq if tur[1] >= min_freq and tur[0] not in tokens]
        self.unique_tokens = dict()
        self.reversed_unique_tokens = dict()

        for i in range(len(self.unique_tokens)):
            self.unique_tokens += {tokens[i]: i}
        self.reversed_unique_tokens = [{v, k} for k, v in self.unique_tokens]

    def __len__(self):
        return len(self.unique_tokens)

    def __getitem__(self, tokens):
        """转换到一个一个的item进行输出"""
        if not isinstance(tokens, (list, tuple)):
            return self.unique_tokens.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """如果是单个index直接输出，如果是list或者tuple迭代输出"""
        if not isinstance(indices, (list, tuple)):
            return self.reversed_unique_tokens[indices]
        return [self.reversed_unique_tokens[index] for index in indices]

