"""
Loss Functions for Skeleton-Aware U-Net + clDice
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,prediction,target):
        smooth=1e-6
        prediction=prediction.view(-1)
        target=target.view(-1)
        inter=(prediction*target).sum()
        dice=(2*inter+smooth)/(prediction.sum()+target.sum()+smooth)
        return 1-dice

class SkeletonLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.L1Loss()
    def forward(self,prediction,skeleton):
        return self.l1(prediction,skeleton)

def soft_erode(img):
    p1=-F.max_pool2d(-img,(3,1),1,(1,0))
    p2=-F.max_pool2d(-img,(1,3),1,(0,1))
    return torch.min(p1,p2)

def soft_dilate(img):
    return F.max_pool2d(img,3,1,1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img,iters=10):
    img1=soft_open(img)
    skel=F.relu(img-img1)
    for _ in range(iters):
        img=soft_erode(img)
        img1=soft_open(img)
        delta=F.relu(img-img1)
        skel=skel+F.relu(delta-skel*delta)
    return skel

def soft_dice(pred,target,smooth=1.0):
    inter=(pred*target).sum((1,2,3))
    union=pred.sum((1,2,3))+target.sum((1,2,3))
    return 1-((2*inter+smooth)/(union+smooth)).mean()

def soft_cldice(pred,target,iters=10,smooth=1.0):
    sp=soft_skeletonize(pred,iters)
    st=soft_skeletonize(target,iters)
    tprec=((sp*target).sum((1,2,3))+smooth)/(sp.sum((1,2,3))+smooth)
    tsens=((st*pred).sum((1,2,3))+smooth)/(st.sum((1,2,3))+smooth)
    cl=2*(tprec*tsens)/(tprec+tsens+1e-8)
    return 1-cl.mean()

class SoftDiceclDiceLoss(nn.Module):
    def __init__(self,iters=10,alpha=0.5):
        super().__init__()
        self.iters=iters
        self.alpha=alpha
    def forward(self,pred,target):
        return self.alpha*soft_cldice(pred,target,self.iters)+(1-self.alpha)*soft_dice(pred,target)

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCELoss()
        self.dice=DiceLoss()
        self.skeleton=SkeletonLoss()
        self.cldice=SoftDiceclDiceLoss()
    def forward(self,prediction,target,skeleton):
        bce=self.bce(prediction,target)
        dice=self.dice(prediction,target)
        sk=self.skeleton(prediction,skeleton)
        cl=self.cldice(prediction,target)
        return 0.30*bce+0.30*dice+0.20*sk+0.20*cl
