import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import torch.nn as nn
import torch.utils.data as data
import config
import numpy as np
import cv2 as cv
from Loaders import BigDataset
import time
import torch.nn.functional as F
from commons import calcPSNR,iter,tonemap
import final_net as OursNet
import lpips
def calcSSIM(prediction, target):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def reparameterize(pretrained_model, rep_model,device=config.device, save_rep_checkpoint=False):
    rep_model = rep_model.to(device)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    
    for k, v in rep_state_dict.items():            
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]
            
            bias_str = k.replace(".weight", ".bias")
            print(bias_str)
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]
            
            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0
        
            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3,device=config.device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0
            
            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()
            
        elif "rep_conv.bias" in k:
            pass
            
        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if save_rep_checkpoint:
        torch.save(rep_state_dict, f"rep_model_{config.checkpoint_id}.pth")
        
    return rep_model
def infer(net, test_scale,pretrained_model,modelPath,testpath,device=config.device):
    # n_steps 就是公式里的 T
    # net 是某个继承自 torch.nn.Module 的神经网络
    resPath='./res/'
    if not os.path.exists(resPath):
        os.mkdir(resPath)
    dataset=BigDataset([testpath],[test_scale],is_test=True,isPatch=True,seq_length=1,lr_mode='read')
    dataloader =  data.DataLoader(dataset,1,shuffle=False,num_workers=1)
    net = net.to(device)
    psnr1=0
    chkpoint = torch.load(modelPath,map_location=device)
    pretrained_model.load_state_dict(chkpoint['state_dict'],strict=False)
    net=reparameterize(pretrained_model=pretrained_model,rep_model=net)
    global iter
    iter=0
    fid=-2
    ssim1=0
    lp=0
    loss_net=lpips.LPIPS(net='vgg').to('cuda')
    loss_net.eval()
    startTime = time.time()
    with torch.no_grad():
        net.eval()
        for sequence in dataloader:
            iter+=1
            gt = sequence['gt'][:, 0, :, :, :]
            cur_lr_feature=sequence['cur_lr'][:, 0, :, :, :]
            cur_hr_feature=sequence['cur_hr'][:, 0, :, :, :]
            prev_hr_feature=sequence['prev_hr'][:, 0, :, :, :]
            cur_hr_brdf=sequence['cur_hr_brdf'][:, 0, :, :, :]
            hr_mv=sequence['hr_mv'][:, 0, :, :, :]
            jitter=sequence['jitter'][:,0]
            scale=sequence['scale']            
            if int(sequence['frameid'][0])!=fid+1:
                net.reset_hidden(gt.shape[0],gt.shape[-2],gt.shape[-1])
                print('Reset')
            fid=int(sequence['frameid'][0])
            cur_lr_feature=cur_lr_feature.to(config.device)
            cur_hr_feature=cur_hr_feature.to(config.device)
            prev_hr_feature=prev_hr_feature.to(config.device)
            hr_mv=hr_mv.to(config.device)
            gt=gt.to(config.device)
            jitter=jitter.to(config.device)
            scale=scale.to(config.device)
            torch.cuda.synchronize()
            start = time.time()
            res= net(cur_lr_feature,cur_hr_feature,prev_hr_feature,hr_mv,jitter,scale)
            torch.cuda.synchronize()
            end = time.time()
            fid=int(sequence['frameid'][0])
            if iter<=1000000:
                res=res.clip(0)
                albedo=cur_hr_brdf[0].detach().cpu().numpy().transpose([1,2,0])
                if config.Demodulate:
                    
                    r1=tonemap(res[0].detach().cpu().numpy().transpose([1,2,0]),TYPE='log_inv')
                    r2=tonemap(cur_lr_feature[0].detach().cpu().numpy().transpose([1,2,0]),TYPE='log_inv')
                    r3=tonemap(gt[0].detach().cpu().numpy().transpose([1,2,0]),TYPE='log_inv')
                    
                    pos=(albedo<=1e-6)
                    r1*=albedo
                    r1[pos]=r3[pos]
                    
                    if config.InferToFile:
                        cv.imwrite('%s/%s-%s-%d.exr'%(resPath,config.scene,str(scale[0].item()),fid),r1)
                        # cv.imwrite('%s/%s-%s-%d.exr'%(resPath,'lr',str(scale[0].item()),fid),r2)
                        # cv.imwrite('%s/%s-%s-%d.exr'%(resPath,'gt',str(scale[0].item()),fid),r3)
                    
                else:
                    r1=res[0].detach().cpu().numpy().transpose([1,2,0]).clip(0,1)
                    r2=cur_lr_feature[0].detach().cpu().numpy().transpose([1,2,0])
                    r3=gt[0].detach().cpu().numpy().transpose([1,2,0])
                                        
                r1=(r1**(1/2.2)).clip(0,1)
                r3=(r3**(1/2.2)).clip(0,1)
                lp+=loss_net(torch.from_numpy(r1*2-1).permute(2,0,1).unsqueeze(0).to(device),torch.from_numpy(r3*2-1).permute(2,0,1).unsqueeze(0).to(device)).mean().item()
                psnr=calcPSNR(r1,r3,MAXP=1.0)
                psnr1+=psnr
                ssim=calcSSIM(r1,r3)
                ssim1+=ssim     
            torch.cuda.empty_cache()
    endTime = time.time()
    print("inference time is {}".format(endTime - startTime))
    print('PSNR: ',psnr1/iter)
    print('SSIM: ',ssim1/iter)
    print('LPIPS: ',lp/iter)

if __name__ == '__main__':

    net=OursNet.OursNet(is_train=False)

    pretrained_model=OursNet.OursNet(is_train=False).to(config.device)
    for i in range(len(config.test_paths)):
        for _,item in enumerate(config.test_scale_list):
            print('start: ',item,',',config.test_paths[i])
            infer(net,item,pretrained_model,config.modelPath+'Model31_mod_234_finalloss_100.pt',config.test_paths[i])#