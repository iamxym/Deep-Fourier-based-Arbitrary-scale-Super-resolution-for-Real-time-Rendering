
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from commons import *
#code from https://github.com/eduardzamfir/RT4KSR
class ResBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(ResBlock, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, int(ratio*n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv2d(int(ratio*n_feats), int(ratio*n_feats), 3, 1, 0)
        self.reduce_conv = nn.Conv2d(int(ratio*n_feats), n_feats, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return  out


class RepResBlock(nn.Module):
    def __init__(self, n_feats):
        super(RepResBlock, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out  

class RBlock(nn.Module):
    def __init__(self, in_c, is_train,act='gelu',layernorm=False, residual=True) -> None:
        super(RBlock,self).__init__()
        self.residual = residual
        if is_train:
            self.conv1 = ResBlock(in_c)
        else:
            self.conv1 = RepResBlock(in_c)
        
        # activation
        if act=='gelu':
            self.act = nn.GELU()
        elif act=='relu':
            self.act = nn.ReLU(inplace=True)
        elif act=='softmax':
            self.act = nn.Softmax(dim=1)
        elif act=='softmax2d':
            self.act = nn.Softmax2d()
        elif act=='sigmoid':
            self.act = nn.Sigmoid()
        elif act=='tanh':
            self.act = nn.Tanh()
        elif act=='lrelu':
            self.act=nn.LeakyReLU(0.2)
        else:
            print(act+'is NOT implemented in RBlock')
            assert(0)
        # channel attention
        
        # LayerNorm
        if layernorm:
            self.norm = LayerNorm2d(in_c)
        else:
            self.norm = None
        
    def forward(self, x):
        return x+self.act(self.conv1(x))
        # res = x.clone()
        
        # if self.norm is not None:
        #     x = self.norm(x)    
        # x = self.conv1(x)
        
        # x = self.act(x)
        
        # if self.residual:
        #     x += res
        # else:
        #     x = x
        # return x
    
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, bias=bias,padding=1))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
class LWGatedConv2D(nn.Module):
    def __init__(self, input_channel1, output_channel, pad, kernel_size, stride):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=input_channel1, out_channels=output_channel, kernel_size=kernel_size,
                                      stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=input_channel1, out_channels=1, kernel_size=kernel_size, stride=stride,
                      padding=pad),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # inputs = inputs * mask
        newinputs = self.conv_feature(inputs)
        mask = self.conv_mask(inputs)

        return newinputs*mask
## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self,  inc,outc, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                 inc, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(inc, outc, kernel_size,padding=kernel_size//2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
class doubleResidualConv(nn.Module):
    def __init__(self,outc,kernel_size=3,padding=1):
        super(doubleResidualConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)+x
class FeatureExtraction(nn.Module):
      def __init__(self,inc,outc,is_train,act='relu',midc=[32,32,32],num_blocks=3,need_RG=True,need_lwg=False,kernel_size=3,padding=1):
            super(FeatureExtraction,self).__init__()
            netlist=[]
            if need_RG:
                c1=inc
                netlist.append(nn.Conv2d(inc,midc[0],kernel_size=1,padding=0))
                for i in range(num_blocks):
                    netlist.append(ResidualGroup(midc[i],midc[i], kernel_size, 16, n_resblocks=2))
                    if i==num_blocks-1:
                        continue
                    if need_lwg:
                        netlist.append(LWGatedConv2D(midc[i],midc[i+1],pad=padding,kernel_size=kernel_size,stride=1))
                    else:
                        netlist.append(nn.Conv2d(midc[i],midc[i+1],kernel_size=1,padding=0))
                netlist.append(nn.Conv2d(midc[-1],outc,kernel_size=1,padding=0))
            else:
                c1=inc
                for i in range(num_blocks):
                    if i==num_blocks-1:
                        c2=outc
                    else:
                        c2=midc[i]
                    if need_lwg:
                        netlist.append(LWGatedConv2D(c1,c2,pad=padding,kernel_size=kernel_size,stride=1))
                    else:
                        netlist.append(nn.Conv2d(c1,c2,kernel_size=kernel_size,padding=padding))
                    # netlist.append(nn.Conv2d(c1,c2,kernel_size=3,padding=1))
                    if act=='gelu':
                        netlist.append(nn.GELU())
                    elif act=='relu':
                        netlist.append(nn.ReLU(inplace=True))
                    elif act=='softmax':
                        netlist.append(nn.Softmax(dim=1))
                    elif act=='softmax2d':
                        netlist.append(nn.Softmax2d())
                    elif act=='sigmoid':
                        netlist.append(nn.Sigmoid())
                    elif act=='tanh':
                        netlist.append(nn.Tanh())
                    elif act=='lrelu':
                        netlist.append(nn.LeakyReLU(0.2))
                    else:
                        print(act+'is NOT implemented in RBlock')
                        assert(0)
                    c1=c2        
                    
                
            self.net=nn.Sequential(*netlist)
      def forward(self,x):
          ret=self.net(x)
          return ret
  
class OursNet(nn.Module):
     def __init__(self,is_train,lr_hidc=32,hr_hidc=32,mlpc=64) -> None:
        super(OursNet,self).__init__()
        self.hidden=None
        self.resimg=None
        self.is_train=is_train
        self.curExtraction=FeatureExtraction(3,lr_hidc*2,is_train,midc=[32,48,48],num_blocks=4,need_RG=False)
        if config.Pixelshuffle:
            r=4
        else:
            r=1
        self.gbufferConv=nn.Sequential(
            nn.Conv2d((10-3+1)*r,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            doubleResidualConv(64),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.ReLU(inplace=True),
            doubleResidualConv(64),
            nn.Conv2d(64,hr_hidc*2,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.ReLU(inplace=True),
            doubleResidualConv(hr_hidc*2)
        )
        self.tempConv=FeatureExtraction((5+3)*r,lr_hidc//2,is_train,midc=[32,32],num_blocks=3,need_RG=False,need_lwg=True)
        self.imgConv=FeatureExtraction(lr_hidc//2+lr_hidc+hr_hidc,3*r,is_train,midc=[64,48],num_blocks=3,need_RG=False)
        self.lastConv=nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,padding=1)
        )
        self.HFConv=nn.Conv2d(3*r,3*r,kernel_size=3,padding=1)
        self.coef = nn.Conv2d(lr_hidc, mlpc, 3, padding=1)
        self.freq = nn.Conv2d(hr_hidc, mlpc, 3, padding=1)
        self.phase= nn.Conv2d(1, mlpc//2, kernel_size=1,bias=False)
        self.mlp=nn.Sequential(
            nn.Conv2d(mlpc,mlpc,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc,mlpc,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc,mlpc,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlpc,3*r,kernel_size=1),
        )
        self.expand_func = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
     def reset_hidden(self,n,th,tw):
        self.hidden=None
        self.resimg=None
        if not self.is_train:
            self.h_w_invr=None
            self.coord=None
     def set_hidden(self,hid,res):
        self.hidden=hid
        self.resimg=res
     def forward(self,cur_lr,cur_hr,prev_hr,mv,jitter,r):
        #cur_lr:[img]
        #cur_hr:[depth,normal,albedo,spat_mask]
        #prev_hr:[depth,normal,albedo]
        
        n,_,ih,iw=cur_lr.shape
        th,tw=cur_hr.shape[-2:]
        lr_features=self.curExtraction(cur_lr[:,0:3])
        #kernel applying
        if config.Pixelshuffle:
            th=th//2
            tw=tw//2
        mp_r=r.view(1,1,1,1).repeat(n,1,th,tw).to(torch.float32)
        inv_r=torch.ones(n,1,th,tw,device=cur_lr.device,dtype=torch.float32)/mp_r
        jitter_w=(jitter[:,0:1]/tw).repeat(1,th*tw).unsqueeze(-1)
        jitter_h=(jitter[:,1:2]/th).repeat(1,th*tw).unsqueeze(-1)
        coord=make_coord((th,tw),device=lr_features.device).unsqueeze(0).repeat(n,1,1)
        feat_coord = make_coord((ih,iw),device=lr_features.device,flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(lr_features.shape[0], 2, ih,iw)
        q_coord = F.grid_sample(
            feat_coord, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        
        rel_coord = coord - q_coord+torch.cat([jitter_h,jitter_w],dim=-1)
        rel_coord[:, :, 0] *= ih
        rel_coord[:, :, 1] *= iw
        my_rel_coord = rel_coord.permute(0, 2, 1).view(rel_coord.shape[0], rel_coord.shape[2],th,tw)
        up_img=F.interpolate(cur_lr[:,0:3],size=cur_hr.shape[-2:],mode='bilinear',align_corners=False)
        amp_feat,recon_lr_feat=torch.split(lr_features,lr_features.shape[1]//2,dim=1)
        up_lr_features=F.interpolate(recon_lr_feat,(th,tw),mode='bilinear',align_corners=False)
        hr_inp=torch.cat([cur_hr[:,0:7],cur_hr[:,-1:]],dim=1)
        if config.Pixelshuffle:
            hr_inp=F.pixel_unshuffle(hr_inp,2)
        gbuffer_features=self.gbufferConv(torch.cat([hr_inp],dim=1))
        freq_feat,recon_hr_feat=torch.split(gbuffer_features,gbuffer_features.shape[1]//2,dim=1)
        if self.resimg!=None:
            prev=torchImgWrap(torch.cat([prev_hr[:,0:7],self.resimg],dim=1),mv)
            temp_inp=torch.cat([calc_diff(cur_hr[:,0:7],prev[:,0:7],None),prev[:,-3:]],dim=1)
        else:
            temp_inp=torch.cat([torch.zeros(n,5,cur_hr.shape[-2],cur_hr.shape[-1],device=up_img.device),up_img],dim=1)
        if config.Pixelshuffle:
            temp_inp=F.pixel_unshuffle(temp_inp,2)
        temp_features=self.tempConv(temp_inp)
        coef=self.coef(amp_feat)
        freq=self.freq(freq_feat)
        q_coef = F.grid_sample(
                    coef, coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .view(n,-1,th,tw).contiguous()
        q_coef=torch.cat([q_coef],dim=1)
        q_freq=freq
        q_freq = torch.stack(torch.split(q_freq, 2, dim=1), dim=1)
        q_freq = torch.mul(q_freq, my_rel_coord.unsqueeze(1)).sum(2)        
        q_freq += self.phase(inv_r)
        q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=1)
        inp_origin = torch.mul(q_coef, q_freq)#
        ret=self.HFConv(self.mlp(inp_origin))
        if config.Pixelshuffle:
            self.resimg=self.lastConv(F.pixel_shuffle(ret,2)+F.pixel_shuffle(self.imgConv(torch.cat([up_lr_features,temp_features,recon_hr_feat],dim=1)),2))
        else:
            self.resimg=self.lastConv(ret+self.imgConv(torch.cat([up_lr_features,temp_features,recon_hr_feat],dim=1)))

        return self.resimg
    