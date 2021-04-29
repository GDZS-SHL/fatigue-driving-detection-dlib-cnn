from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
from  torch.utils.data import DataLoader as dt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
#define
lr=0.0001
batchsize=16
warmup=5
max_epoch=100
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root1=r'D:/学习相关/大三下/cv/fd/eye/train'
root2=r'D:/学习相关/大三下/cv/fd/eye/test'
transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((40,40)),transforms.ToTensor()])
trainset=datasets.ImageFolder(root1,transform=transform)
testset=datasets.ImageFolder(root2,transform=transform)
train_loader=dt(dataset=trainset,batch_size=batchsize,shuffle=True)
test_loader=dt(dataset=testset,batch_size=batchsize,shuffle=False)

class My_ResidualBlock(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.c1=nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(channel, channel,kernel_size=3, padding=1)
    def forward(self,x):
        x=nn.functional.relu(self.c1(x))
        y=self.c2(x)
        return nn.functional.relu(x+y)
class My_CNN_withRB_eye(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(1,64,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.c5=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.c8=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.rb1=My_ResidualBlock(512)
        self.pooling=nn.MaxPool2d(2)
        self.linear1=nn.Linear(2048,10)
        self.linear2=nn.Linear(10,2)
        self.do=nn.Dropout(0.5)
    def forward(self,x):
        x=nn.functional.relu(self.c1(x))
        x=nn.functional.relu(self.c2(x))
        x=nn.functional.relu(self.pooling(self.c3(x)))
        x=nn.functional.relu(self.c4(x))
        x=nn.functional.relu(self.pooling(self.c5(x)))
        x=nn.functional.relu(self.c6(x))
        x=nn.functional.relu(self.c6(x))
        x=nn.functional.relu(self.pooling(self.c7(x)))
        x=nn.functional.relu(self.c8(x))
        x=nn.functional.relu(self.c8(x))
        x=self.rb1(x)
        x=nn.functional.relu(self.pooling(self.c8(x)))
        x=nn.functional.relu(self.c8(x))
        x=nn.functional.relu(self.c8(x))
        x=x.view(-1,2048)  
        x=self.do(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x

#prepare
model=My_CNN_withRB_eye()
model.to(device)
loss_fun=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
scheduler_1 = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/warmup)
scheduler_2 = CosineAnnealingLR(optimizer,4*max_epoch)
def train(epoch):

    for i,data in enumerate(train_loader,0):
        x,target=data
        x,target=x.to(device),target.to(device)
        optimizer.zero_grad()
        #model.train()
        y_pred=model(x)
        loss=loss_fun(y_pred,target)
        loss.backward()
        optimizer.step()
        
def test(epoch):
    with torch.no_grad():
        correct=0
        total=0
        for i,data in enumerate(test_loader,0):
            inputs, target = data
            inputs,target=inputs.to(device),target.to(device)
            model.eval()
            y_pred = model(inputs)
            _,pred=torch.max(y_pred,dim=1)
            total+=target.size(0)
            correct+=(pred==target).sum().item()   
        print(epoch+1,correct/total)
        
if __name__ == '__main__':
    for epoch in range(max_epoch):
        train(epoch)
        if epoch<5:
            scheduler_1.step()
        else:
            scheduler_2.step()
        test(epoch)
    #torch.save(model,'cnn_for_eye_16.pkl')
