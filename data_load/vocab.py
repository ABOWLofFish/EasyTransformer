import collections

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 根据出现频率排序
        tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知标记的索引为0
        self.unk = 0
        uniq_tokens = ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freq
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token = []
        self.token_to_idx = dict()  # 根据索引找标记和根据标记找索引

        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """转换到一个一个的item进行输出"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """如果是单个index直接输出，如果是list或者tuple迭代输出"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

