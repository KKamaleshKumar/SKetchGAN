from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import glob

class CreateDataset(Dataset):

    def __init__(self,dataset_path):
        super(Dataset, self).__init__()
        self.ratio1=0.05
        self.ratio2=0.75   # ratio2 greater than ratio 1
        self.clearance=64
        #self.center_crop=900    omit for sketchy
        self.dataset_path=dataset_path
        class_paths=glob.glob(self.dataset_path+'*')
        self.classes={}
        self.img_data=[]
        for i,path in enumerate(class_paths):
            class_name=path.split('/')[-1]
            self.classes.update({class_name:i})
            img_list=[[_,i] for _ in glob.glob(path+'/*')]
            self.img_data.extend(img_list)

    def __len__(self):

        return len(self.img_data)
    
    ''' def random_mask(self,low,high):
        #print(low,' ',high)
        coord_0=np.random.randint(int(low),int(high-self.clearance))
        print(coord_0)
        coord_1=np.random.randint(coord_0+self.clearance+1,int(high))
        print(coord_0,' ',coord_1)
         

        return coord_0,coord_1'''
    
    def __getitem__(self, idx):

        img_path,class_idx=self.img_data[idx]
        img=Image.open(img_path)
        img=ImageOps.grayscale(img)
        ''' img_target=img
        img_mask=ImageDraw.Draw(img)
        count_unique=0
        while(count_unique<=1):            
            x0,x1=self.random_mask(self.ratio1*img.size[0],self.ratio2*img.size[0])
            y0,y1=self.random_mask(self.ratio1*img.size[1],self.ratio2*img.size[1])
            img_cropped=img.crop((x0,y0,x1,y1))            
            count_unique=np.unique(np.asarray(img_cropped)).shape[0]
        img_mask.rectangle([x0,y0,x1,y1],fill='white')
        #img.save('/home/user/Documents/SketchGAN_KAMALESH/masked/Image{}.jpg'.format(idx))'''
        transforms_=transforms.Compose([
            #transforms.CenterCrop(self.center_crop),   # comment for sketchy
            transforms.ToTensor()
        ])
        class_idx=torch.tensor([class_idx])
       # mask=torch.tensor([x0,y0,x1,y1])

        return transforms_(img),class_idx

        

if __name__=='__main__':
    dataset=CreateDataset('/home/user/Documents/SketchGAN_KAMALESH/dataset/')
    data_loader=DataLoader(dataset, batch_size=4, shuffle=True)
    dummy=list(data_loader)
    





