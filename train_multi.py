#python -m torch.distributed.launch --nproc_per_node=4  --use_env train_multi.py
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
os.environ['OMP_NUM_THREADS']='1'
os.environ['NCCL_LL_THRESHOLD']='0'
os.environ['NCCL_P2P_DISABLE']='1'
os.environ['NCCL_IB_DISABLE']='1'
os.environ['NCCL_DEBUG']='INFO'
import torch
import torch.nn as nn
import torch.utils.data as data
import config
import numpy as np
import cv2 as cv
from Loaders import BigDataset
import torch.nn.functional as F
import final_net as OursNet
import time
from commons import get_temporal_data,l1_norm
import pytorch_ssim
import lpips
# 初始化
class myLoss(nn.Module):
    def __init__(self,loss_net,relative_weight=0.1):
        super(myLoss,self).__init__()
        self.weight=relative_weight
        self.loss_net = loss_net
        self.loss_net.eval()
        self.loss_ssim=pytorch_ssim.SSIM(window_size=11)
    
    def forward(self,output, temporal_output, target, temporal_target,mask):
        ssimloss=(1-self.loss_ssim(output,target))*0.4
        
        ls = l1_norm(output, target)
        lm=(mask*torch.abs(output-target)).sum()/(mask.sum()+1)*0.1
		
        lp=self.loss_net((output*2-1),(target*2-1)).mean()*config.perceptual_weight
        lt = l1_norm(temporal_output, temporal_target)*0.3
        total_loss=ssimloss+ls+lm+lp+lt
        return total_loss,ls,lm,lt,lp,ssimloss
def set_multiGPU():
	local_rank=int(os.environ['LOCAL_RANK'])
	world_size=int(os.environ['WORLD_SIZE'])
	torch.distributed.init_process_group(backend="nccl",
								rank=local_rank,
								world_size=world_size)
	torch.distributed.barrier()	
	torch.cuda.set_device(local_rank)
	torch.distributed.is_initialized()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model=OursNet.OursNet(is_train=True)
	model = model.to(device)
	criterion=myLoss(lpips.LPIPS(net='vgg').to(device))
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config.learning_rate*world_size)
	scheduler=torch.optim.lr_scheduler.StepLR(optimizer, config.n_epochs//5, 0.5)
	epoch_start=0
	if config.load_dir != "":
		chkpoint = torch.load(config.load_dir,map_location=device)
		epoch_start = int(chkpoint['epoch'])
		model.load_state_dict(chkpoint['state_dict'])
		optimizer.load_state_dict(chkpoint['optimizer'])
		for i in range(epoch_start):
			scheduler.step()
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.parallel.DistributedDataParallel(model,
														device_ids=[local_rank],find_unused_parameters=True)
														# output_device=local_rank,
	dataset=BigDataset(config.train_paths,config.train_scale_list,lr_mode='read')

	if local_rank==1:
		print('dataset length:',len(dataset))

	train_sampler=torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=local_rank,shuffle=True)
	dataloader =data.DataLoader(dataset,sampler=train_sampler,batch_size=1,num_workers=8,pin_memory=True)
	return model,optimizer,epoch_start,scheduler,dataloader,device,criterion
def train_sequence(model, sequence,device,criterion):
	output_final = sequence['gt'].clone().to(device)
	output_final.fill_(0)
	target_final = sequence['gt'].clone().to(device)
	target_final.fill_(0)

	target = sequence['gt'].to(device)
	cur_lr_feature=sequence['cur_lr'].to(device)
	cur_hr_feature=sequence['cur_hr'].to(device)
	cur_hr_brdf=sequence['cur_hr_brdf'].to(device)
	prev_hr_feature=sequence['prev_hr'].to(device)
	hr_mv=sequence['hr_mv'].to(device)
	jitter=sequence['jitter'].to(device)
	scale=sequence['scale'].to(device)

	loss_final = 0
	loss_l1=0
	loss_temporal=0
	loss_mask=0
	loss_percept=0
	loss_ssim=0
	seq_length=config.seq_length
	hr_mv=hr_mv.to(device)
	for j in range(0, seq_length):
		cur_lr_featurei=cur_lr_feature[:, j, :, :, :]
		cur_hr_featurei=cur_hr_feature[:, j, :, :, :]
		cur_hr_brdfi=cur_hr_brdf[:, j, :, :, :]
		prev_hr_featurei=prev_hr_feature[:, j, :, :, :]
		hr_mvi=hr_mv[:, j, :, :, :]
		gti = target[:, j, :, :, :]
		jitteri=jitter[:,j]
		if j == 0:
			model.module.reset_hidden(gti.shape[0],gti.shape[-2],gti.shape[-1])

		output = model(cur_lr_featurei,cur_hr_featurei,prev_hr_featurei,hr_mvi,jitteri,scale)
		if config.Demodulate:
			output*=cur_hr_brdfi
		output_final[:, j, :, :, :] = output
		target_final[:, j, :, :, :] = gti

	temporal_output, temporal_target = get_temporal_data(output_final, target_final,hr_mv)

	for j in range(0, seq_length):
		output = output_final[:, j, :, :, :]
		target = target_final[:, j, :, :, :]
		t_output = temporal_output[:, j, :, :, :]
		t_target = temporal_target[:, j, :, :, :]

		l ,ls,lm,lt,lp,ssimloss= criterion(output, t_output, target, t_target,cur_hr_feature[:, j, -1:, :,:])
		loss_final += l
		loss_l1+=ls
		loss_percept+=lp
		loss_temporal+=lt
		loss_mask+=lm
		loss_ssim+=ssimloss
	

	return loss_final,loss_l1,loss_temporal,loss_percept,loss_mask,loss_ssim


def train(model, dataloader, optimizer, epoch,device,criterion):
	total_loss = 0
	total_loss_num = 0
	total_l1loss=0
	total_temporalloss=0
	total_perceptloss=0
	total_maskloss=0
	total_ssimloss=0
	for item in dataloader:
		for k, v in item.items():
			#v.shape=(N,T,...)
			if k=='frameid' or k=='scale':
				pass
				# print(k,item[k])
			else:
				# print(v.shape)
				item[k] = v.reshape(config.batch_size,config.seq_length,*v.shape[2:])
		optimizer.zero_grad()
		loss_final,ls,lt,lp,lm,ssim_loss= train_sequence(model, item,device,criterion)
		
		loss_final.backward(retain_graph=False)
		optimizer.step()

		total_loss += loss_final.item()
		total_l1loss+=ls.item()
		total_temporalloss+=lt.item()
		total_perceptloss+=lp.item()
		total_maskloss+=lm.item()
		total_ssimloss+=ssim_loss.item()
		total_loss_num += config.seq_length
		del item,loss_final
		torch.cuda.empty_cache()

	total_loss /= total_loss_num
	total_l1loss/=total_loss_num
	total_temporalloss/=total_loss_num
	total_perceptloss/=total_loss_num
	total_maskloss/=total_loss_num
	total_ssimloss/=total_loss_num

	return total_loss,total_l1loss,total_temporalloss,total_perceptloss,total_maskloss,total_ssimloss
if __name__=='__main__':
	torch.manual_seed(3407)
	ckpt_path=config.modelPath
	
	model,optimizer,epoch_start,scheduler,dataloader,device,criterion=set_multiGPU()	
	local_rank=int(os.environ['LOCAL_RANK'])
	for e in range(epoch_start,config.n_epochs):
		dataloader.sampler.set_epoch(e)
		startTime = time.time()
		iter=0
		model.train()
		loss,ls,lt,lp,lm,ssimloss=train(model,dataloader,optimizer,e,device,criterion)
		scheduler.step()
		endTime = time.time()
		
		if local_rank==1:
			print("{} epoch time is {}".format(e,endTime - startTime))
			print("loss-%f, L1-%f, temp-%f, perceptual-%f, mask-%f, (1-ssim)-%f."%(loss,ls,lt,lp,lm,ssimloss))
		
		if (local_rank==1 and e%(config.n_epochs//5)==(config.n_epochs//5)-1):
			torch.save({
						'epoch': e+1,
						'state_dict':model.module.state_dict(),
						'optimizer':optimizer.state_dict(),
					}, '%s/%s_%s.pt' % (ckpt_path,config.modelName, e+1))
	if local_rank==1:		
		torch.save({
						'epoch': config.n_epochs,
						'state_dict':model.module.state_dict(),
						'optimizer':optimizer.state_dict(),
					}, '%s/%s_%s.pt' % (ckpt_path,config.modelName, config.n_epochs))
  