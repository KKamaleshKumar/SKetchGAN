import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw


class DownSampleBlock(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,padding=1,batch_norm=True):

        super(DownSampleBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.LReLU=nn.LeakyReLU(0.2)
        self.init_weight(self.conv)
        if batch_norm:
            self.bn=nn.BatchNorm2d(out_channels)
            self.out=nn.Sequential(self.conv,self.bn,self.LReLU)

        else:

            self.out=nn.Sequential(self.conv,self.LReLU)

    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self,x):

            return self.out(x)

class UpSampleBlock(nn.Module):

    

    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,padding=1,batch_norm=True,drop_out=False):
    
        super(UpSampleBlock,self).__init__()
        self.Tconv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding) #check out padding
        self.dropout=nn.Dropout(0.5)
        self.bn=nn.BatchNorm2d(out_channels)
        self.init_weight(self.Tconv)
        if drop_out:
        
            self.out=nn.Sequential(self.Tconv,self.bn,self.dropout,nn.ReLU())
        else:
        
            self.out=nn.Sequential(self.Tconv,self.bn,nn.ReLU())

    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)

  
    def forward(self,x):
    
  
        return self.out(x)

class GeneratorBlock(nn.Module):


    def __init__(self,in_channels,out_channels):
    
        super(GeneratorBlock,self).__init__()
        self.encoders=[
            DownSampleBlock(in_channels,64,batch_norm=False).cuda(),
            DownSampleBlock(64,128).cuda(),
            DownSampleBlock(128,256).cuda(),
            DownSampleBlock(256,512).cuda(),
            DownSampleBlock(512,512).cuda(),
            DownSampleBlock(512,512).cuda(),
            DownSampleBlock(512,512).cuda(),
            DownSampleBlock(512,512,batch_norm=False).cuda()
        ]
    
        self.decoders=[
            UpSampleBlock(512,512,drop_out=True).cuda(),
            UpSampleBlock(1024,512,drop_out=True).cuda(),
            UpSampleBlock(1024,512,drop_out=True).cuda(),
            UpSampleBlock(1024,512).cuda(),
            UpSampleBlock(1024,256).cuda(),
            UpSampleBlock(512,128).cuda(),
            UpSampleBlock(256,64).cuda()               
        ]
    
        self.conv_final=nn.ConvTranspose2d(128,out_channels,kernel_size=4,stride=2,padding=1)
        self.init_weight(self.conv_final)
    
    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)

  
    def forward(self,x):
    
        skips=[]
        for encoder in self.encoders:
            x=encoder(x)
            skips.append(x)
        skips=list(reversed(skips[:-1]))
    
        for decoder,skip in zip(self.decoders,skips):     
           # print(x.shape)   
            x=decoder(x)
            #print(x.shape, skip.shape)
            x=torch.cat((x,skip),axis=1)  # concat along channel axis,  that is axis=1
        x=self.conv_final(x)
        tanh=nn.Tanh()
       # print(x.shape)
    
        return tanh(x)

class CascadeGenerator(nn.Module): # check architecture of preprocessing convnet

  
    def __init__(self,in_channels,out_channels):
    
        super(CascadeGenerator,self).__init__()
    
        self.Stage1conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same') 
        self.Stage1conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding='same')
        self.Stage1bn1=nn.BatchNorm2d(out_channels)
        self.Stage1bn2=nn.BatchNorm2d(out_channels)
        self.init_weight(self.Stage1conv1)
        self.init_weight(self.Stage1conv2)
        self.Generator1=nn.Sequential(
            self.Stage1conv1,
            self.Stage1bn1,
            nn.ReLU(),
            self.Stage1conv2,
            self.Stage1bn2,
            nn.ReLU(),
            GeneratorBlock(out_channels,out_channels)
        )
    
        self.Stage2conv1=nn.Conv2d(2*out_channels,2*out_channels,kernel_size=3,stride=1,padding='same') 
        self.Stage2conv2=nn.Conv2d(2*out_channels,2*out_channels,kernel_size=3,stride=1,padding='same')
        self.Stage2bn1=nn.BatchNorm2d(2*out_channels)
        self.Stage2bn2=nn.BatchNorm2d(2*out_channels)
        self.init_weight(self.Stage2conv1)
        self.init_weight(self.Stage2conv2)

        self.Generator2=nn.Sequential(
            self.Stage2conv1,
            self.Stage2bn1,
            nn.ReLU(),
            self.Stage2conv2,
            self.Stage2bn2,
            nn.ReLU(),
            GeneratorBlock(2*out_channels,out_channels) 
        )
    
        self.Stage3conv1=nn.Conv2d(3*out_channels,3*out_channels,kernel_size=3,stride=1,padding='same') 
        self.Stage3conv2=nn.Conv2d(3*out_channels,3*out_channels,kernel_size=3,stride=1,padding='same')
        self.Stage3bn1=nn.BatchNorm2d(3*out_channels)
        self.Stage3bn2=nn.BatchNorm2d(3*out_channels)
        self.init_weight(self.Stage3conv1)
        self.init_weight(self.Stage3conv2)

        self.Generator3=nn.Sequential(
            self.Stage3conv1,
            self.Stage3bn1,
            nn.ReLU(),
            self.Stage3conv2,
            self.Stage3bn2,
            nn.ReLU(),
            GeneratorBlock(out_channels*3,out_channels)        
        )

    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)

  
    def forward(self,x):
    
      
        y1=self.Generator1(x)
        #print(y1.shape,x.shape)
        x1=torch.cat((x,y1),axis=1)
        y2=self.Generator2(x1)
        x2=torch.cat((x1,y2),axis=1)
        y3=self.Generator3(x2)
        y3=(y3+1)/2
    
        return y3   #   map values between 0 and 1

class GlobalDiscriminatorBlock(nn.Module):

    

    def __init__(self,in_channels,kernel_size=5,stride=2):
    
        super(GlobalDiscriminatorBlock,self).__init__()
        conv1=nn.Conv2d(in_channels,64,kernel_size=kernel_size,stride=stride)
        conv2=nn.Conv2d(64,128,kernel_size=kernel_size,stride=stride)
        conv3=nn.Conv2d(128,256,kernel_size=kernel_size,stride=stride)
        conv4=nn.Conv2d(256,512,kernel_size=kernel_size,stride=stride)
        conv5=nn.Conv2d(512,512,kernel_size=kernel_size,stride=stride)
        conv6=nn.Conv2d(512,512,kernel_size=kernel_size,stride=stride)
        bn1=nn.BatchNorm2d(64)
        bn2=nn.BatchNorm2d(128)
        bn3=nn.BatchNorm2d(256)
        bn4=nn.BatchNorm2d(512)
        bn5=nn.BatchNorm2d(512)
        self.init_weight(conv1)
        self.init_weight(conv2)
        self.init_weight(conv3)
        self.init_weight(conv4)
        self.init_weight(conv5)
        self.init_weight(conv6)
        self.net=nn.Sequential(
            conv1,
            bn1,
            nn.LeakyReLU(0.2),
            conv2,
            bn2,
            nn.LeakyReLU(0.2),
            conv3,
            bn3,
            nn.LeakyReLU(0.2),
            conv4,
            bn4,
            nn.LeakyReLU(0.2),
            conv5,
            bn5,
            nn.LeakyReLU(0.2),
            conv6,
            nn.LeakyReLU(0.2)
        )

    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)
  
    def forward(self,x):
    
  
        return self.net(x)

class LocalDiscriminatorBlock(nn.Module):


    def __init__(self,in_channels,kernel_size=5,stride=2):
    
        super(LocalDiscriminatorBlock,self).__init__()
        conv1=nn.Conv2d(in_channels,64,kernel_size=kernel_size,stride=stride)
        conv2=nn.Conv2d(64,128,kernel_size=kernel_size,stride=stride)
        conv3=nn.Conv2d(128,256,kernel_size=kernel_size,stride=stride)
        conv4=nn.Conv2d(256,512,kernel_size=kernel_size,stride=stride)
        conv5=nn.Conv2d(512,512,kernel_size=kernel_size,stride=stride)
        bn1=nn.BatchNorm2d(64)
        bn2=nn.BatchNorm2d(128)
        bn3=nn.BatchNorm2d(256)
        bn4=nn.BatchNorm2d(512)
        self.init_weight(conv1)
        self.init_weight(conv2)
        self.init_weight(conv3)
        self.init_weight(conv4)
        self.init_weight(conv5)        
        self.net=nn.Sequential(
            conv1,
            bn1,
            nn.LeakyReLU(0.2),
            conv2,
            bn2,
            nn.LeakyReLU(0.2),
            conv3,
            bn3,
            nn.LeakyReLU(0.2),
            conv4,
            bn4,
            nn.LeakyReLU(0.2),
            conv5,
            nn.LeakyReLU(0.2)
        )
    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)
      
    def forward(self,x):
    
  
        return self.net(x)

def removal(img_gen,img_in,mask):

    PILtransform=transforms.ToPILImage()    
    img_gen_transform=PILtransform(img_gen)
    img_gen_masked=Image.new(mode='1',size=img_gen_transform.size,color=1)
    img_gen_crop=img_gen_transform.crop((mask[0].item(),mask[1].item(),mask[2].item(),mask[3].item()))
    img_gen_masked.paste(img_gen_crop,(mask[0].item(),mask[1].item()))
    img_in_transform=PILtransform(img_in)
    img_in_masked=Image.new(mode='1',size=img_in_transform.size,color=1)
    img_in_crop=img_in_transform.crop((mask[0].item(),mask[1].item(),mask[2].item(),mask[3].item()))
    img_in_masked.paste(img_in_crop,(mask[0].item(),mask[1].item()))
    toTensor=transforms.ToTensor()
    return toTensor(img_gen_masked).cuda(),toTensor(img_in_masked).cuda()



class Discriminator(nn.Module):


    def __init__(self,in_channels,kernel_size=5,stride=2):
    
        super(Discriminator,self).__init__()
        # assume mask is of appropriate channel dimension
        self.LocalD=LocalDiscriminatorBlock(2*in_channels,kernel_size,stride)
        self.GlobalD=GlobalDiscriminatorBlock(2*in_channels,kernel_size,stride)
        self.linear_global=nn.Linear(512,1024)
        self.linear_local=nn.Linear(12800,1024)
        self.linear_final=nn.Linear(2048,1)
        self.init_weight(self.linear_global)
        self.init_weight(self.linear_local)
        self.init_weight(self.linear_final)

    def init_weight(self,layer):
        torch.nn.init.kaiming_normal_(layer.weight)
  
    def forward(self,x,inputs,mask):
    
  
        x_split=torch.split(x,1,dim=0) #tuple
        mask_split=torch.split(mask,1,dim=0)
        inputs_split=torch.split(inputs,1,dim=0)
        x_list=[]
        input_list=[]
        for x_,input_,mask_ in zip(x_split,inputs_split,mask_split):
            x__,input__=removal(x_.squeeze(),input_.squeeze(), mask_.squeeze())
            #print(x__.shape,input__.shape)
            x_list.append(torch.unsqueeze(x__,0))
            input_list.append(torch.unsqueeze(input__,0))
        x_masked=torch.stack(tuple(x_list),0)
        input_masked=torch.stack(tuple(input_list),0)
        #dummy=torch.cat((inputs,x),dim=1)
        #print(x.shape, inputs.shape)
        y_global=self.GlobalD(torch.cat((inputs,x),dim=1))
        x_conditioned=torch.cat((input_masked,x_masked),dim=1).squeeze()
        #print("X shape ",len(x_conditioned.shape))
        if len(x_conditioned.shape)==3:
            x_conditioned=torch.unsqueeze(x_conditioned,0)

        y_local=self.LocalD(x_conditioned)  # WHOLE IMAGE OR ONLY PATCH INPUT???
        y_global=y_global.view(y_global.shape[0],-1)
        y_local=y_local.view(y_local.shape[0],-1)
        y_global=self.linear_global(y_global)
        y_global=nn.ReLU()(y_global)
        y_local=self.linear_local(y_local)
        y_local=nn.ReLU()(y_local)
        y=torch.cat((y_local,y_global),dim=1)      
        y=self.linear_final(y)     # only logits since using nn.BCEWithLogitsLoss  for numerical stablity
                       
        return y

class SketchClassifier(nn.Module):

  
    def __init__(self,in_channels,num_classes):
    
    
        super(SketchClassifier,self).__init__()
        self.num_classes = num_classes
        conv1 =nn.Conv2d(in_channels, 64,kernel_size=15, stride=3)
        conv2 =nn.Conv2d(64, 128,kernel_size=5, stride=1)
        conv3 =nn.Conv2d(128, 256,kernel_size=3 , stride=1, padding=1)
        conv4 =nn.Conv2d(256, 256,kernel_size=3, stride=1, padding=1)
        conv5 =nn.Conv2d(256, 256,kernel_size=3 , stride=1, padding=1)
        conv6 =nn.Conv2d(256, 512,kernel_size=7 , stride=1, padding=0)
        conv7 =nn.Conv2d(512, 512,kernel_size=1, stride=1, padding=0)
        self.linear=nn.Linear(2048, self.num_classes)
        self.init_weight(conv1)
        self.init_weight(conv2)
        self.init_weight(conv3)
        self.init_weight(conv4)
        self.init_weight(conv5)
        self.init_weight(conv6)
        self.init_weight(conv7)
        self.init_weight(self.linear)
        self.fcn=nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            conv2,
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            conv3,
            nn.ReLU(),
            conv4,
            nn.ReLU(),
            conv5,
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            conv6,
            nn.ReLU(),
            nn.Dropout(0.5),
            conv7,
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    def init_weight(self,layer):

        torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
    

        x=self.fcn(x)
        x=x.view(x.shape[0],-1)
  
        return self.linear(x)