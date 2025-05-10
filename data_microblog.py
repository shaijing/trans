import json

from nltk import word_tokenize
train_cn_file = "data/Bi-Microblog_zh.tok"
train_en_file = "data/Bi-Microblog_en.tok"


train_cn = []
with open(train_cn_file, mode="r", encoding="utf-8") as f:
    for line in f.readlines():
        sen = line.strip().replace(' ','').split("\t")[0]
        train_cn.append(sen)
train_en = []
with open(train_en_file, mode="r", encoding="utf-8") as f:
    for line in f.readlines():
        train_en.append(line.strip().lower().split("\t")[0])

data_dict = []

for cn ,en in zip(train_cn,train_en):
    data_dict.append({"zh-CN":cn,"en-US":en})

with open("data/microblog-train.json", mode="w", encoding="utf-8") as f:
    json.dump(data_dict, f)