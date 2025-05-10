import time
import torch
from science4ai.nn.modules.transformer import Transformer
from data import TranslationData
from util import train, LabelSmoothing,NoamOpt


EPOCHS = 25                         # 训练轮数
D_MODEL = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = TranslationData(device=device)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print(f"src_vocab {src_vocab}")
print(f"tgt_vocab {tgt_vocab}")

model = Transformer(d_model=D_MODEL,src_vocab=src_vocab,tgt_vocab=tgt_vocab,norm_first=True,device=device)

print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

train(data, model, criterion, optimizer,EPOCHS)
print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")