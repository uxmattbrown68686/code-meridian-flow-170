# -*- coding: utf-8 -*-

# 小龙智脑 (XiaoLong Brain) - 全新原创项目

# 作者 / 版权人: 小龙 (XiaoLong)

# License: MIT。本项目所有代码均为原创，保留署名即可自由使用。



import math

import re





class Embeddings:

    def embed(self, text):

        raise NotImplementedError



    def embed_batch(self, texts):

        return [self.embed(t) for t in texts]





class HashEmbeddings(Embeddings):

    # 基于词哈希的确定性嵌入，无需任何模型文件，适合演示与基线

    def __init__(self, dim=256):

        self.dim = dim



    def embed(self, text):

        vec = [0.0] * self.dim

        tokens = re.findall(r"[a-z0-9]+|[一-鿿]", (text or "").lower())

        for tok in tokens:

            h = hash(tok) % self.dim

            vec[h] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))

        if norm > 0:

            vec = [v / norm for v in vec]

        return vec

