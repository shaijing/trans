TRAIN_FILE = '../nmt/en-cn/train.txt'  # 训练集
DEV_FILE = "../nmt/en-cn/dev.txt"  # 验证集
from langconv import Converter
from nltk import word_tokenize
import json
def cht_to_chs(sent):
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent

data_dict = []

en = []
cn = []
path = DEV_FILE
with open(path, mode="r", encoding="utf-8") as f:
    for line in f.readlines():
        sent_en, sent_cn = line.strip().split("\t")
        sent_cn = cht_to_chs(sent_cn)
        data_dict.append({"en-US":sent_en, "zh-CN":sent_cn})
        sent_en = sent_en.lower()
        sent_en = ["BOS"] + word_tokenize(sent_en) + ["EOS"]
        # 中文按字符切分
        sent_cn = ["BOS"] + [char for char in sent_cn] + ["EOS"]
        en.append(sent_en)
        cn.append(sent_cn)
with open("../data/dev_data_dict.json", mode="w", encoding="utf-8") as f:
    json.dump(data_dict, f)