import torch
# 分词工具
m = torch.nn.modules.transformer.Transformer()
def funtest(a,b=1,c=2,d=3,e=4):
    print(f"{a} {b} {c} {d} {e}")

p = {"d":4,"e":5}
funtest(1,**p)