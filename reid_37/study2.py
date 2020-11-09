#%%
from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model
import cv2

#torchreid.utils.load_pretrained_weights(model, 'log/resnet50/model.pth.tar-60')

class FeatureExtractor(object):
    
    def __init__(
        self,
        model_name='',
        # model_path='',
        image_size=(128, 256), # (width, height)
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=751,
            pretrained=True,
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        num_params, flops = compute_model_complexity(
            model, (1, 3, image_size[0], image_size[1])
        )

        if verbose:
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        # if model_path and check_isfile(model_path):
        #     load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [Resize(image_size)]
        transforms += [ToTensor()]
        if pixel_norm:
            transforms += [Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_cv = ToCVImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_cv = to_cv
        self.device = device
        
        
    """
    input : image list
    """
    def __call__(self, input):
        
        if isinstance(input, list):
            images = []
            #images, pids, camids = [], [], []
            
            for element in input:
                if isinstance(element, str):
                    image = cv2.imread(element)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                elif isinstance(element, np.ndarray):
                    image = self.to_cv(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image) # (C, H, W) // torch.Size([3, 256, 128])
                images.append(image) 
            
            images = torch.stack(images, dim=0) # (N, C, H, W) // torch.Size([14, 3, 256, 128])
            images = images.to(self.device)

        elif isinstance(input, str):
            image = cv2.imread(input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_cv(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images) # torch.Size([14, 2048])

        return features


import cv2
import collections
from torchvision.transforms import functional as F
from torch.nn import functional as tF

Iterable = collections.abc.Iterable

class Resize(object):
    """
    Args:
        size : (width, height)
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self, img):
        return cv2.resize(img, self.size, self.interpolation)

class ToTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic)
        img = img.permute(2,0,1)
        img = img.float().div(255)
        return img

class Normalize(object):
    def __init__(self, mean, std, inplace = True):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)


def _is_cv_image(img):
    return isinstance(img, cv2.cv2)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(pic):
    if not(_is_cv_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be CV Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = pic[:, :, None]
            
        img = torch.from_numpy(pic.transpose((2,0,1)))
        
        if isinstance(img, torch.ByteTensor): # unsinged 8bit
            return img.float().div(255)
        else:
            return img

def to_cv_image(pic):
    """
    Convert a tensor or an ndarray to CV Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))
            
    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))
            
        elif pic.ndimension() == 2:
            pic = pic.unsqueeze(0)
    
    elif isinstance(pic, np.ndarray):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))
            
        elif pic.ndimension() == 2:
            pic = np.expand_dims(pic, 2)
        
            
    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        # floatTensor → uint8 (0 ~ 255)
        ##pic = pic.mul(255).byte()
        pic = (np.clip(pic.numpy(),0.,1.) * 255.).astype(np.uint8)
    if isinstance(pic, torch.Tensor):
        npimg = np.transpose(pic.numpy(), (1, 2, 0)) # CHW(torch) → HWC(numpy)
        
    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                    'not {}'.format(type(npimg)))    
    
    if npimg.shape[2] == 3: # npimg = (H,W,C), 채널이 3 이면
        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))
        
    return npimg


    
class ToCVImage(object):
    """
    CHW → HWC
    """
    def __init__(self, mode = None):
        self.mode = mode
        
    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to CV Image.

        Returns:
            CV Image: Image converted to CV Image.
        """
        return to_cv_image(pic, self.mode)
    
def compute_distance_matrix(input1,input2,metric='euclidean'):
    """
    input dim : 2-D
    """
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim()==2,'input : 2D, but got {}D'.format(input1.dim())
    assert input2.dim()==2,'input : 2D, but got {}D'.format(input2.dim())
    assert input1.size(1)==input2.size(1)
    
    if metric=='euclidean':
        distmat = euclidean_squared_distance(input1,input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1,input2)
    else:
        raise ValueError(
            'Unknown distance metric : {}'.format(metric)
            )
    return distmat

def euclidean_squared_distance(input1, input2):
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m,n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n,m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1,-2,input1,input2.t())
    return distmat

def cosine_distance(input1, input2):
    input1_normed = tF.normalize(input1, p=2, dim=1)
    input2_normed = tF.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

def parse_data_for_eval(data):
    for i in range(len(data)):
        imgs = data[i][0]
        pids = data[i][1]
        camids = data[i][2]
        return imgs, pids, camids
#%%
import os
import os.path as osp
import torchreid
import cv2
from torchreid.utils import mkdir_if_missing, visualize_ranked_results

path = 'C:/Users/ojkk3/Documents/Dropbox/person-searching/Person-Tracking-and-Re-ID/deep-reid/reid-data/Test_folder/'
q = 'query/'
g = 'gallery/'

#%%
class Dataloader(object):
    def __init__(self, qdir, gdir):
        self.qdir = qdir
        self.gdir = gdir
        
    def __call__(self):        
        q_list = os.listdir(self.q_dir)
        g_list = os.listdir(self.g_dir)
        query = []
        for file in q_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                query.append((self.q_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2])))
                
        gallery = []
        for file in g_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                gallery.append((self.g_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2])))
        return query, gallery



class Feature_extractor(object):
    def __init__(self, query, gallery):
        self.query = query
        self.gallery = gallery
        
        self.extractor = FeatureExtractor(
            model_name='resnet50',
            model_path='log/resnet50/model.pth.tar-60',
            device='cpu'
            )
        
    def __call__(self):
        fs, pids, camids = [], [], []
        for i in range(len(self.query)):
            img, pid, camid = parse_data_for_eval(self.query)
            f = self.extractor(img)
            fs.append(f)
            pids.append(pid)
            camids.append(camid)
        fs = torch.cat(fs, 0)
        pids = np.asarray(pids)
        camids = np.asarray(camids)
        return fs, pids, camids

print(qfs.size()) # torch.Size([1, 2048])
print(gfs.size()) # torch.Size([25, 2048])

distmat = compute_distance_matrix(qfs, gfs, 'euclidean')
distmat = distmat.numpy()

print("query_feature :", qfs.shape)
print("gallery_feature :", gfs.shape)

#nq, ng = distmat.shape # (1, 25)
#indices = np.argsort(distmat, axis=1)

dataset = (query, gallery)
save_dir='C:/Users/ojkk3/Documents/Dropbox/person-searching/Person-Tracking-and-Re-ID/deep-reid/reid-data/visrank_Test_dataset/'

visualize_ranked_results(
    distmat,
    dataset,
    data_type='image',
    save_dir=save_dir
    )