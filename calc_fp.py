from thop import profile
from thop import clever_format
import torch
import final_net
model=final_net.OursNet(is_train=False)
model.eval()
h=270
w=480
r=4
cur_lr=torch.randn(1,3,h,w)
cur_hr=torch.randn(1,7,h*r,w*r)
prev_hr=torch.randn(1,7,h,w)
mv=torch.randn(1,2,h*r,w*r)
jitter=torch.zeros(1,2)
r=torch.tensor([r])
print(r)
flops, params = profile(model, inputs=(cur_lr,cur_hr,prev_hr,mv,jitter,r))
print('Flops: % .4fG'%(flops / 1000000000))# 计算量
print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值
# 将结果转换为更易于阅读的格式
flops, params = clever_format([flops, params], '%.3f')

print(f"运算量：{flops}, 参数量：{params}")
