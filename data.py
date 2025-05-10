from nltk import word_tokenize
from collections import Counter
import json
import numpy as np
import torch
from torch.autograd import Variable

PAD = 0  # padding占位符的索引
UNK = 1  # 未登录词标识符的索引
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pad_mask(seq: torch.tensor, pad_idx: int = PAD) -> torch.tensor:
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Batch:
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """

    def __init__(self, src, trg=None, pad=PAD,device=None):
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(device).long()
        trg = torch.from_numpy(trg).to(device).long()
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = get_pad_mask(src,pad)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]  #（去掉了最后一个词）
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]   # (去掉了第一个词）# 目的是(src + trg) 来预测出来(trg_y)，
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = get_pad_mask(tgt,pad)
        tgt_mask = tgt_mask & get_subsequent_mask(tgt)
        return tgt_mask


class TranslationData:
    def __init__(self, train_file='data/train_data_dict.json', dev_file="data/dev_data_dict.json", batch_size=128,device=None):
        self.device = device
        self.train_file = train_file
        self.dev_file = dev_file
        self.batch_size = batch_size
        # 读取数据、分词
        self.train_en, self.train_cn = self._load_data(self.train_file)
        self.dev_en, self.dev_cn = self._load_data(self.dev_file)
        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = \
            self._build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = \
            self._build_dict(self.train_cn)
        # 单词映射为索引
        self.train_en_idx, self.train_cn_idx = self._word2id(self.train_en, self.train_cn, self.en_word_dict,
                                                             self.cn_word_dict)
        self.dev_en_idx, self.dev_cn_idx = self._word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        #test
        # self.train_batch = self._split(self.train_en_idx, self.train_cn_idx, self.batch_size)

        # 划分批次、填充、掩码
        self.train_data = self._split_batch(self.train_en_idx, self.train_cn_idx, self.batch_size)
        self.dev_data = self._split_batch(self.dev_en_idx, self.dev_cn_idx, self.batch_size)

    @staticmethod
    def _load_data(file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            data_li = json.load(f)
            en = [["BOS"] + word_tokenize(item["en-US"].lower()) + ["EOS"] for item in data_li]
            cn = [["BOS"] + [char for char in item["zh-CN"]] + ["EOS"] for item in data_li]
        # if len(en)> 2000:
        #     en = en[:1000]
        #     cn = cn[:1000]
        return en, cn

    @staticmethod
    def _build_dict(sentence_li, max_words=5e4):
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentence_li for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加UNK和PAD两个单词
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2
        word_dict = {'PAD': PAD, 'UNK': UNK}
        word_dict.update({w[0]: index + 2 for index, w in enumerate(ls)})
        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    @staticmethod
    def _word2id(en, cn, en_dict, cn_dict, sort=True):
        """
        传入一系列语句数据(分好词的列表形式)，
        按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
        """
        out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in cn]

        def arg_sort(seq):
            """
            传入一系列语句数据(分好词的列表形式)，
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 按相同顺序对中文、英文样本排序
        if sort:
            # 以英文语句长度排序
            sorted_index = arg_sort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    @staticmethod
    def seq_padding(sentence_li, padding=PAD):
        """
        按批次（batch）对数据填充、长度对齐
        """
        # 计算该批次各条样本语句长度
        sentence_len = [len(x) for x in sentence_li]
        # 获取该批次样本中语句长度最大值
        max_len = max(sentence_len)
        # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
        return np.array([
            np.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in sentence_li
        ])


    def _split(self,en, cn, batch_size, shuffle=True):
        """
               划分批次
               `shuffle=True`表示对各批次顺序随机打乱
               """
        # 每隔batch_size取一个索引作为后续batch的起始索引
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_index = []
        for idx in idx_list:
            """
            形如[array([4, 5, 6, 7]), 
                     array([0, 1, 2, 3]), 
                     array([8, 9, 10, 11]),
                     ...]
            """
            # 起始索引最大的批次可能发生越界，要限定其索引
            batch_index.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_index:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            # 维度为：batch_size * 当前批次中语句的最大长度
            batch_cn = self.seq_padding(batch_cn)
            batch_en = self.seq_padding(batch_en)
            batches.append((batch_en, batch_cn))
        return batches
    def _split_batch(self,en, cn, batch_size, shuffle=True):
        """
               划分批次
               `shuffle=True`表示对各批次顺序随机打乱
               """
        # 每隔batch_size取一个索引作为后续batch的起始索引
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_index = []
        for idx in idx_list:
            """
            形如[array([4, 5, 6, 7]), 
                     array([0, 1, 2, 3]), 
                     array([8, 9, 10, 11]),
                     ...]
            """
            # 起始索引最大的批次可能发生越界，要限定其索引
            batch_index.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_index:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            # 维度为：batch_size * 当前批次中语句的最大长度
            batch_cn = self.seq_padding(batch_cn)
            batch_en = self.seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn,device=self.device))
        return batches


# data = TextData()
# b = data.train_batch[0]
# src = b[0]
# trg = b[1]
# src = torch.from_numpy(src).to(DEVICE).long()
# trg = torch.from_numpy(trg).to(DEVICE).long()
# src_mask = get_pad_mask(src,PAD)
# trg_mask = get_pad_mask(trg,PAD)
# t = get_subsequent_mask(trg)
# tt = trg_mask & t
