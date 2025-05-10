import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model, **factory_kwargs)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, **factory_kwargs)
        # 单词位置
        position = torch.arange(0.0, max_len, **factory_kwargs)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, **factory_kwargs) * (
                - math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练
        x += self.pe[:, : x.size(1), :]
        return self.dropout(x)


def clones(module, N):
    """
    克隆基本单元，克隆的单元之间参数不共享
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(N)
    ])


def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention（方程（4））
    """
    # q、k、v向量长度为d_k
    d_k = query.size(-1)
    # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意力掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 注意力矩阵softmax归一化
    p_attn = F.softmax(scores, dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力对v加权
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention（编码器（2））
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout=0.1, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiHeadedAttention, self).__init__()
        """
        `h`：注意力头的数量
        `d_model`：词向量维数
        """
        # 确保整除
        assert d_model % nhead == 0
        # q、k、v向量维数
        self.d_k = d_model // nhead
        # 头的数量
        self.h = nhead
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model, **factory_kwargs), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 批次大小
        nbatches = query.size(0)
        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 注意力加权
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 多头注意力加权拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        # α、β分别初始化为1、0
        self.a_2 = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
        self.b_2 = nn.Parameter(torch.zeros(normalized_shape, **factory_kwargs))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 沿词向量方向计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 沿词向量和语句序列方向计算均值和方差
        # mean = x.mean(dim=[-2, -1], keepdim=True)
        # std = x.std(dim=[-2, -1], keepdim=True)
        # 归一化
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2


class SublayerConnection(nn.Module):
    """
    通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
    """

    def __init__(self, d_model, dropout, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward: int = 2048, dropout=0.1, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout=0.1,
                 norm_first: bool = False,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadedAttention(d_model, nhead, dropout, **factory_kwargs)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, **factory_kwargs)
        # SublayerConnection作用连接multi和ffn
        self.sublayer = clones(SublayerConnection(d_model, dropout, **factory_kwargs), 2)

    def forward(self, x, src_mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        # attn的结果直接作为下一层输入
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self,
                 encoder_layer: "EncoderLayer",
                 num_layers: int,
                 norm: "LayerNorm"
                 ):
        """
        layer = EncoderLayer
        """
        super(Encoder, self).__init__()
        # 复制N个编码器基本单元
        self.layers = clones(encoder_layer, num_layers)
        # 层归一化
        self.norm = norm

    def forward(self, x, mask):
        """
        循环编码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout=0.1,
                 device=None,
                 dtype=None
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, nhead, dropout, **factory_kwargs)
        # 上下文注意力机制
        self.src_attn = MultiHeadedAttention(d_model, nhead, dropout, **factory_kwargs)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, **factory_kwargs)
        self.sublayer = clones(SublayerConnection(d_model, dropout,**factory_kwargs), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory为编码器输出隐表示
        m = memory
        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,
                 decoder_layer: "DecoderLayer",
                 num_layers: int,
                 norm: "LayerNorm"
                 ):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        循环解码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab,
                 device=None,
                 dtype=None
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab,**factory_kwargs)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 src_vocab: int = 2048,
                 tgt_vocab: int = 2048,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, **factory_kwargs)
        encoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )
        self.encoder = Encoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout, **factory_kwargs)
        decoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )
        self.decoder = Decoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )
        self.src_embed = nn.Sequential(
            Embeddings(d_model,src_vocab,**factory_kwargs),
            PositionalEncoding(d_model,dropout,**factory_kwargs)
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(d_model,tgt_vocab,**factory_kwargs),
            PositionalEncoding(d_model,dropout,**factory_kwargs)
        )
        self.generator = Generator(d_model, tgt_vocab, **factory_kwargs)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)