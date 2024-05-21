import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch.utils.data as data
import torch
import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import config
import random
from commons import dilate,tonemap
class BigDataset(data.Dataset):
    def __init__(self, paths,scale_list,lr_mode,seq_length=config.seq_length,is_test=False, isPatch=True): 
        self.is_test = is_test
        self.seq_length=seq_length
        self.lr_mode=lr_mode
        self.isPatch=isPatch
        #the data dir is '/data/OurDataset/SceneName/SequenceName/TYPE_SCALE/Prefix&BufferName.IDX.exr
        #So we set paths '/data/OurDataset/SceneName/SequenceName/'
        self.paths=paths
        self.scale_list=scale_list
        self.totalNum = 0
        
        self.imgSet = []
        self.sidSet=[]
        self.sceneSet=[]
        for path in self.paths:#TODO
            scene_name=path.split('/')[-3]
            imgs = os.listdir(path+'Nojitter-1080P/')
            imgs.sort()
            self.sidSet.append(int(imgs[0].split(".")[-2]))
            self.sceneSet.append(scene_name)
            resimgs=imgs[1::seq_length][:-1]
            setNum = len(resimgs)
            self.imgSet.append(resimgs)
            self.totalNum += setNum

    def mapIndex2PathAndIndex(self, index):
        remain = index
        for setIndex,ims in enumerate(self.imgSet):
            if remain < len(ims):
                return self.paths[setIndex], ims[remain].split(".")[-2],ims[remain].split(".")[0][:-18],self.sidSet[setIndex],self.sceneSet[setIndex]
            else:
                remain -= len(ims)

        return None, -1
    def getScale(self):
        if self.is_test:
            return self.scale_list[0]
        else:
            return self.scale_list[random.randint(0,len(self.scale_list)-1)]
    def __getitem__(self, index):
        path, id,scene_prefix,start_id,scene_name = self.mapIndex2PathAndIndex(index)
        
        scale=self.getScale()
        scale_prefix=config.scale_dict[scale]
        
        cur_lr_list=[]
        cur_hr_list=[]
        prev_hr_list=[]
        hr_mv_list=[]
        gt_list=[]
        cur_brdf_list=[]
        jitter_list=[]
        gt_prefix='PreTonemapHDRColor'
        MV_M_R_prefix='MotionVectorAndMetallicAndRoughness'
        N_D_prefix='WorldNormalAndSceneDepth'
        albedo_prefix='BaseColor'
        brdf_prefix='Albedo'
        
        for i in range(self.seq_length):
            #NOTE:exr save:RGBA
            #opencv read:BGRA
            idx=str(int(id)+i).zfill(4)
            previdx=str(int(id)+i-1).zfill(4)
            gt=cv.imread(path+'Nojitter-1080P/%s%s.%s.exr'%(scene_prefix,gt_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
            gt=gt.clip(0,10)
            cur_hr_albedo=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,albedo_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
            cur_hr_brdf=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,brdf_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
            th,tw=cur_hr_albedo.shape[0:2]
            cur_hr_mvmr=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,MV_M_R_prefix,idx),cv.IMREAD_UNCHANGED)
            cur_hr_mv=cur_hr_mvmr[:,:,1:3]
            cur_hr_mr=np.concatenate([cur_hr_mvmr[:,:,0:1],cur_hr_mvmr[:,:,3:4]],axis=-1)
            
            
                
            if config.RandomMask and self.is_test==False:
                if self.lr_mode=='read':
                    assert(0)
                cnt=random.randint(0,6)
                ps=random.randint(32,64)
                for i in range(cnt):
                    rh=random.randint(0,th-ps)
                    rw=random.randint(0,tw-ps)
                    gt[rh:rh+ps,rw:rw+ps]=0
                
            
            cur_tmp=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,N_D_prefix,idx),cv.IMREAD_UNCHANGED)
            cur_hr_depth=cur_tmp[:,:,3:4]
            cur_hr_normal=cur_tmp[:,:,0:3]            
            cur_hr_depth[cur_hr_depth>100.0]=0
            cur_hr_depth=(cur_hr_depth-cur_hr_depth.min())/(1e-6+max(1.0,cur_hr_depth.max())-cur_hr_depth.min())
            cur_hr_nlen=np.sqrt(np.square(cur_hr_normal).sum(-1, keepdims=True)).repeat(3, axis=-1)+1e-6
            cur_hr_normal/=cur_hr_nlen
            cur_hr_feature=np.concatenate([cur_hr_depth,cur_hr_normal,cur_hr_albedo,cur_hr_mr],axis=-1)
            
            prev_hr_albedo=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,albedo_prefix,previdx),cv.IMREAD_UNCHANGED)[:,:,0:3]
            prev_tmp=cv.imread(path+'GBuffer-1080P/%s%s.%s.exr'%(scene_prefix,N_D_prefix,previdx),cv.IMREAD_UNCHANGED)
            prev_hr_depth=prev_tmp[:,:,3:4]
            prev_hr_normal=prev_tmp[:,:,0:3]            
            prev_hr_depth[prev_hr_depth>100.0]=0
            prev_hr_depth=(prev_hr_depth-prev_hr_depth.min())/(1e-6+max(1.0,prev_hr_depth.max())-prev_hr_depth.min())
            prev_hr_nlen=np.sqrt(np.square(prev_hr_normal).sum(-1, keepdims=True)).repeat(3, axis=-1)+1e-6
            prev_hr_normal/=prev_hr_nlen
            prev_hr_feature=np.concatenate([prev_hr_depth,prev_hr_normal,prev_hr_albedo],axis=-1)
            
            if self.is_test==False and self.isPatch==True:
                if self.lr_mode=='read':
                    cur_img=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,gt_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    cur_lr_tmp=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,N_D_prefix,idx),cv.IMREAD_UNCHANGED)
                    cur_lr_depth=cur_lr_tmp[:,:,3:4]
                    cur_lr_normal=cur_lr_tmp[:,:,0:3]            
                    cur_lr_depth[cur_lr_depth>100.0]=0
                    cur_lr_depth=(cur_lr_depth-cur_lr_depth.min())/(1e-6+max(1.0,cur_lr_depth.max())-cur_lr_depth.min())
                    cur_lr_nlen=np.sqrt(np.square(cur_lr_normal).sum(-1, keepdims=True)).repeat(3, axis=-1)+1e-6
                    cur_lr_normal/=cur_lr_nlen
                    cur_lr_albedo=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,albedo_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    cur_lr_brdf=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,brdf_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    ih,iw=cur_img.shape[0:2]
                    lr_patch=int(config.patch_size//scale)
                    sih=random.randint(0,ih-lr_patch)
                    siw=random.randint(0,iw-lr_patch)
                    sth=int(sih*scale)
                    stw=int(siw*scale)
                    
                    prev_hr_feature=prev_hr_feature[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    gt=gt[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_feature=cur_hr_feature[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_brdf=cur_hr_brdf[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_mv=cur_hr_mv[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_img=cur_img[sih:sih+lr_patch,siw:siw+lr_patch]
                    cur_lr_depth=cur_lr_depth[sih:sih+lr_patch,siw:siw+lr_patch]
                    cur_lr_normal=cur_lr_normal[sih:sih+lr_patch,siw:siw+lr_patch]
                    cur_lr_albedo=cur_lr_albedo[sih:sih+lr_patch,siw:siw+lr_patch]
                    cur_lr_brdf=cur_lr_brdf[sih:sih+lr_patch,siw:siw+lr_patch]
                elif self.lr_mode=='downsample':
                    sth=random.randint(0,th-config.patch_size)
                    stw=random.randint(0,tw-config.patch_size)
                    prev_hr_feature=prev_hr_feature[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_feature=cur_hr_feature[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_brdf=cur_hr_brdf[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    cur_hr_mv=cur_hr_mv[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    gt=gt[sth:sth+config.patch_size,stw:stw+config.patch_size]
                    ih=int((config.patch_size+1e-7)/scale)
                    iw=ih
                    th,tw=config.patch_size,config.patch_size
                    cur_img=cv.resize(gt,(iw,ih),interpolation=cv.INTER_AREA)
                    cur_lr_depth=cv.resize(cur_hr_feature[...,0:1],(iw,ih),interpolation=cv.INTER_AREA)[:,:,np.newaxis]
                    cur_lr_normal=cv.resize(cur_hr_feature[...,1:4],(iw,ih),interpolation=cv.INTER_AREA)
                    cur_lr_albedo=cv.resize(cur_hr_feature[...,4:7],(iw,ih),interpolation=cv.INTER_AREA)
            else:
                
                if self.lr_mode=='read':
                    cur_img=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,gt_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    cur_lr_albedo=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,albedo_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    cur_lr_brdf=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,brdf_prefix,idx),cv.IMREAD_UNCHANGED)[:,:,0:3]
                    cur_lr_tmp=cv.imread(path+'Nojitter-%s/%s%s.%s.exr'%(scale_prefix,scene_prefix,N_D_prefix,idx),cv.IMREAD_UNCHANGED)
                    cur_lr_depth=cur_lr_tmp[:,:,3:4]
                    cur_lr_normal=cur_lr_tmp[:,:,0:3]            
                    cur_lr_depth[cur_lr_depth>100.0]=0
                    cur_lr_depth=(cur_lr_depth-cur_lr_depth.min())/(1e-6+max(1.0,cur_lr_depth.max())-cur_lr_depth.min())
                    cur_lr_nlen=np.sqrt(np.square(cur_lr_normal).sum(-1, keepdims=True)).repeat(3, axis=-1)+1e-6
                    cur_lr_normal/=cur_lr_nlen
                elif self.lr_mode=='downsample':
                    th,tw=gt.shape[0:2]
                    ih,iw=int(th/scale),int(tw/scale)
                    cur_img=cv.resize(gt,(iw,ih),interpolation=cv.INTER_AREA)
                    cur_lr_depth=cv.resize(cur_hr_feature[...,0:1],(iw,ih),interpolation=cv.INTER_AREA)[:,:,np.newaxis]
                    cur_lr_normal=cv.resize(cur_hr_feature[...,1:4],(iw,ih),interpolation=cv.INTER_AREA)
                    cur_lr_albedo=cv.resize(cur_hr_feature[...,4:7],(iw,ih),interpolation=cv.INTER_AREA)
                    cur_lr_brdf=cv.resize(cur_hr_brdf,(iw,ih),interpolation=cv.INTER_AREA)
                else:
                    print('NO LR MODE IN LOADER!')
                    assert(0)
            th,tw=gt.shape[:2]
            ih,iw=cur_img.shape[:2]
            if config.Demodulate:
                pos=(cur_hr_feature[:,:,4:7]<=1e-6)
                cur_hr_feature[:,:,4:7][pos]=gt[pos]
                cur_hr_brdf[pos]=gt[pos]
                pos=(cur_lr_albedo<=1e-6)
                cur_lr_albedo[pos]=cur_img[pos]
                cur_lr_brdf[pos]=cur_img[pos]
                
                
                pos=(cur_lr_brdf<=1e-6)
                cur_lr_brdf[pos]=1.0
                cur_img=cur_img/cur_lr_brdf
                cur_img[pos]=0.0
                cur_lr_brdf[pos]=0
                gt=tonemap(gt,'log')
                cur_img=tonemap(cur_img,'log')
            else:
                gt=tonemap(gt)
                cur_img=tonemap(cur_img)
            diff_dep=np.abs(cur_hr_feature[...,0:1]-cv.resize(cur_lr_depth,(tw,th))[:,:,np.newaxis])>0.01
            up_norm=cv.resize(cur_lr_normal,(tw,th))
            up_nlen=np.sqrt(np.square(up_norm).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)+1e-6
            up_norm/=up_nlen
            diff_nor=(cur_hr_feature[...,1:4]*up_norm).sum(-1,keepdims=True) < 0.9
            diff_albedo=np.abs(cur_hr_feature[...,4:7]-cv.resize(cur_lr_albedo,(tw,th))).mean(axis=-1,keepdims=True)>0.1
            diff=np.logical_or(diff_dep,diff_nor)
            diff=np.logical_or(diff,diff_albedo)
            cur_hr_feature=np.concatenate([cur_hr_feature,diff],axis=-1)
            if config.Demodulate:
                cur_lr_feature=np.concatenate([cur_img],axis=-1)
            else:
                cur_lr_feature=np.concatenate([cur_img],axis=-1)
            
            
            cur_lr_list.append(cur_lr_feature)
            cur_hr_list.append(cur_hr_feature)
            cur_brdf_list.append(cur_hr_brdf)
            hr_mv_list.append(cur_hr_mv)
            prev_hr_list.append(prev_hr_feature)
            gt_list.append(gt)
            jitter=(0,0)
            
            jitter_list.append(jitter)
            
            scale=(tw/iw)
              
        
        cur_lr_feature=np.stack(cur_lr_list)
        cur_hr_feature=np.stack(cur_hr_list)
        prev_hr_feature=np.stack(prev_hr_list)
        jitter=np.stack(jitter_list)
        cur_hr_brdf=np.stack(cur_brdf_list)
        hr_mv=np.stack(hr_mv_list)
        gt=np.stack(gt_list)
            
        _,h,w,c=cur_lr_feature.shape
        
        
        
        return {
            'cur_lr':torch.from_numpy(cur_lr_feature.transpose([0,3,1,2])).to(torch.float32),
            'prev_hr':torch.from_numpy(prev_hr_feature.transpose([0,3,1,2])).to(torch.float32),
            'cur_hr':torch.from_numpy(cur_hr_feature.transpose([0,3,1,2])).to(torch.float32),
            'cur_hr_brdf':torch.from_numpy(cur_hr_brdf.transpose([0,3,1,2])).to(torch.float32),
            'hr_mv':torch.from_numpy(hr_mv.transpose([0,3,1,2])).to(torch.float32),
            'gt':torch.from_numpy(gt.transpose([0,3,1,2])).to(torch.float32),
            'frameid':id,
            'scale':float(scale),
            'jitter':torch.tensor(jitter).to(torch.float32),
        }
    def __len__(self):
        return self.totalNum
    
    