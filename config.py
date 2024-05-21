device='cuda:0'
batch_size=1
ToYCrCb=False
Demodulate=True
Pixelshuffle=True
RandomMask=False
patch_size=240
seq_length=8
num_workers=6
modelPath='./Models/'
scene='RedwoodForest'
load_dir=''#modelPath+'Model24(test)_NoMod_80.pt'
scale=2
perceptual_weight=0.2
modelName='Model31_mod_234_finalloss_total'
n_epochs=100
learning_rate=1e-4
scale_dict={
    1.6:'675P',
    2:'540P',
    3:'360P',
    3.75:'288P',
    4:'270P',
    5:'216P',
    6:'180P',
    8:'135P',
    1.7:None
}
train_scale_list=[2,3,4]
test_scale_list=[4]
train_paths=[
    '/data/zjwdataset/Bunker/Bunker_1/',
    '/data/zjwdataset/Bunker/Bunker_2/',
    '/data/zjwdataset/MedievalDocks/MedievalDocks_1/',
    '/data/zjwdataset/MedievalDocks/MedievalDocks_2/',
    '/data/zjwdataset/RedwoodForest/RedwoodForest_1/',
    '/data/zjwdataset/RedwoodForest/RedwoodForest_2/',
    '/data/zjwdataset/WesternTown/WesternTown_1/',
]
test_paths=[
    
    # '/data/zjwdataset/Bunker/Bunker_Train_1/',
    # '/data/zjwdataset/Infiltrator/Infiltrator_1/'
    # '/data/zjwdataset/MedievalDocks/MedievalDocks_Train_1/'
    # '/data/zjwdataset/ShowDown/ShowDown_5.0_10.85s/'
    
    # '/data/zjwdataset/RedwoodForest/RedwoodForest_Train_2/'
    '/data/zjwdataset/Bunker/Bunker_3/',
    '/data/zjwdataset/RedwoodForest/RedwoodForest_Train_6/',
    '/data/zjwdataset/MedievalDocks/MedievalDocks_3_0/',
    '/data/zjwdataset/WesternTown/WesternTown_Test_1/',
    '/data/zjwdataset/EasternVillage/EasternVillage_2/',
    # "/data/zjwdataset/Particle/Bunker_3/"
]
def Halton(id,base):
    ret=0.0
    invb=1.0/base
    frac=invb
    while id>0:
        ret+=(id%base)*frac
        id=id//base
        frac*=invb
    return ret
#低分辨率数据的抖动采用 Halton16 序列, X 方向上以 2 为底, Y 方向上以 3 为底
#numpy与UE4产生/处理图像时均以从左向右为y轴(图宽width)；
#而原点方面，UE4在图像左下角，numpy在图像左上角，因此UE4以从下到上为x轴（图长height），而numpy则是从上到下，所以在计算时需要注意x坐标的正负转化。

'''NOTE:
在UE中修改增加x,y均为正值的jitter offset时，渲染得到的画面会向左下角移动，此时jx,jy>0
而画面往左下角移动时，新画面（同时也是LR）的左上角原点实际在老画面（同时也是HR）坐标系的-x,+y处，
因此对老画面重建时，采样的坐标系是基于新画面的，故应该减去(+jx,-jy)

默认屏幕空间大小是[-1,1]，也即（考虑X坐标）jitter=0时画面在正中央；jitter=-1时，画面向右移动半个屏幕；jitter=1时，画面向左移动半个屏幕；

所以当屏幕空间与numpy坐标的单位换算是X(1:W/2),Y(1:H/2)时，而UE添加的抖动量是(Halton-0.5)/W,(Halton-0.5)/H，所以在这里都要/2
'''
Halton16=[((Halton(i,2)-0.5)/2,-(Halton(i,3)-0.5)/2) for i in range(16)]
scene_jitter_start_pos={
    'Bunker':4,
    'RedwoodForest':4,
    'MedievalDocks':4,
    'WestTown':4,
    'ShowDown':1,
    'Infiltrator':2
}