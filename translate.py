import time
import torch
from data import TranslationData
from modules import Transformer
from util import evaluate_trans
from science4ai import load_checkpoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = TranslationData(device=device)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print(f"src_vocab {src_vocab}")
print(f"tgt_vocab {tgt_vocab}")
D_MODEL = 512                      # 输入、输出词向量维数

model = Transformer(d_model=D_MODEL,src_vocab=src_vocab,tgt_vocab=tgt_vocab,norm_first=True,device=device)
checkpoint = load_checkpoint()
model.load_state_dict(checkpoint["model"])

print(">>>>>>> start evaluate")
evaluate_start  = time.time()
evaluate_trans(data, model,device)
print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")
# en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en_idx[0]])
# cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn_idx[0]])
# src = torch.from_numpy(np.array(data.dev_en_idx[0])).long().to(DEVICE)
# src = src.unsqueeze(0)
# src_mask = (src != 0).unsqueeze(-2)
# memory = model.encode(src, src_mask)
# ys = torch.ones(1, 1).fill_(data.cn_word_dict["BOS"]).type_as(src.data)
# for i in range(MAX_LENGTH - 1):
#     # decode得到隐层表示
#     out = model.decode(memory,
#                        src_mask,
#                        Variable(ys),
#                        Variable(get_subsequent_mask(ys).type_as(src.data)))
#     # 将隐藏表示转为对词典各词的log_softmax概率分布表示
#     prob = model.generator(out[:, -1])
#     # 获取当前位置最大概率的预测词id
#     _, next_word = torch.max(prob, dim=1)
#     next_word = next_word.data[0]
#     # 将当前位置预测的字符id与之前的预测内容拼接起来
#     ys = torch.cat([ys,
#                     torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
# out = model.decode(memory,
#                    src_mask,
#                    Variable(ys),
#                    Variable(get_subsequent_mask(ys).type_as(src.data)))
# prob = model.generator(out[:, -1])
# 开始预测
