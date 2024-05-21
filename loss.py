import numpy as np
import torch.nn as torch
import vgg
import torch
class LossNetwork(nn.Module):

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '15': "relu3",
            '22': "relu4",
            '29': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
class myLoss(nn.Module):
    def __init__(self,relative_weight=0.2):
        super(myLoss,self).__init__()
        self.weight=relative_weight
        self.loss_fn_vgg = LossNetwork().cuda()
        self.loss_fn_vgg.eval()
        self.loss_ssim=pytorch_ssim.SSIM(window_size=11)
    
    def forward(self,pred,label):
        ssimloss=1-self.loss_ssim(pred,label)
        
        pred_features = self.loss_fn_vgg(pred*2-1)
        label_features = self.loss_fn_vgg(label*2-1)
        perceptualloss=0
        
        for i in range(len(label_features)):
            perceptualloss+=self.weight*pred_features[i].dist(label_features[i], p=2) / torch.numel(pred_features[i])
        return ssimloss+perceptualloss