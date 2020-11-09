# #%%
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

# import sys
# import os
# import os.path as osp
# import torchreid
# from torchreid.data import ImageDataset


# class NewDataset(ImageDataset):
#     dataset_dir = 'Test_folder'

#     def __init__(self, root='', **kwargs):
#         self.root = osp.abspath(osp.expanduser(root))
#         self.dataset_dir = osp.join(self.root, self.dataset_dir)

#         q_dir = 'query/'
#         g_dir = 'gallery/'
        
#         q_dir = osp.join(self.dataset_dir, q_dir)
#         g_dir = osp.join(self.dataset_dir, g_dir)
        
#         train = [['temp.jpg',0,0]]
        
#         query = []
#         q_files = os.listdir(q_dir)
#         for file in q_files:
#             if file.endswith(".jpg"):
#                 filename = file.split('.')[0]
#                 query.append([q_dir + file, int(filename.split('_')[1]), filename.split('_')[2]])
                
#         gallery = []
#         g_files = os.listdir(g_dir)
#         for file in g_files:
#             if file.endswith(".jpg"):
#                 filename = file.split('.')[0]
#                 gallery.append([g_dir + file, int(filename.split('_')[1]), filename.split('_')[2]])

#         super(NewDataset, self).__init__(train, query, gallery, **kwargs)
        
# torchreid.data.register_image_dataset('Test_dataset', NewDataset)
# #%%
# # query feature
# qfs, q_pids, q_camids = [], [], []
# for i in range(len(query)):
#     qimg, q_pid, q_camid = parse_data_for_eval(query[i])
#     qimg = qimg.cuda()
#     qf = extractor(qimg)
#     qf = qf.cpu()
#     qfs.append(qf)
#     q_pids.append(q_pid)
#     q_camids.append(q_camid)
# qfs = torch.cat(qfs, 0)
# q_pids = np.asarray(q_pids)
# q_camids = np.asarray(q_camids)

# # gallery feature
# gfs, g_pids, g_camids = [], [], []
# for j in range(len(gallery)):
#     gimg, g_pid, g_camid = parse_data_for_eval(gallery[j])
#     gimg = gimg.cuda()
#     gf = extractor(gimg)
#     gf = gf.cpu()
#     gfs.append(gf)
#     g_pids.append(g_pid)
#     g_camids.append(g_camid)
# gfs = torch.cat(gfs, 0)
# g_pids = np.asarray(g_pids)
# g_camids = np.asarray(g_camids)

# print("query_feature :", qfs.shape) # torch.Size([3, 2048])
# print("gallery_feature :", gfs.shape) # torch.Size([25, 2048])
# #import pdb;pdb.set_trace()
# # result(distance)
# distmat = compute_distance_matrix(qfs, gfs, 'euclidean')
# distmat = distmat.numpy()

# #nq, ng = distmat.shape # (3, 25)
# #indices = np.argsort(distmat, axis=1)

# dataset = (query, gallery)
# save_dir='./reid-data/visrank_Test_dataset/'

# visualize_ranked_results(
#     distmat,
#     dataset,
#     data_type='image',
#     save_dir=save_dir
#     )
# #%%
# import os
# import os.path as osp
# import torchreid
# import cv2

# path = '/home/cresprit/venv/reid_37/deep-person-reid/reid-data/Test_folder/'
# q = 'query/'
# g = 'gallery/'

# q_dir = osp.join(path + q)
# g_dir = osp.join(path + g)

# q_list = os.listdir(q_dir)
# g_list = os.listdir(g_dir)

# query = []
# for file in q_list:
#     if file.endswith(".jpg"):
#         filename = file.split('.')[0]
#         query.append((q_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2])))
        
# gallery = []
# for file in g_list:
#     if file.endswith(".jpg"):
#         filename = file.split('.')[0]
#         gallery.append((g_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2])))

# print(gallery)
# #%%
# #feature extractor build
# extractor = FeatureExtractor(
#     model_name='resnet50',
#     model_path='/home/cresprit/log/resnet50/model/model.pth.tar-60',
#     device='cuda'
#     )

# impath, pid, camid = [], [], []
# for i in range(len(query)):
#     qimg, qpid, qcamid = parse_data_for_eval(query[i])
#     impath.append(qimg)
#     pid.append(qpid)
#     camid.append(qcamid)
# pid = torch.as_tensor(pid)
# camid = torch.as_tensor(camid)
# print("impath :", impath)
# print("pid :", pid)
# print("camid :", camid)
# #%%
# import os
# import os.path as osp
# dataset_dirr = 'Test_folder'
# root='reid-data'
# root = osp.abspath(osp.expanduser(root))
# dataset_dirr = osp.join(root, dataset_dirr)
# qd = 'query/'
# gd = 'gallery'
# qd = osp.join(dataset_dirr, qd)
# gd = osp.join(dataset_dirr, gd)

# print(gd)
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
        model_path='',
        image_size=(128, 256), # (width, height)
        device='cuda',
        verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=14,
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

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.device = device
        
        
    """
    input : image list
    """
    def __call__(self, input):
        
        with torch.no_grad():
            features = self.model(input) # torch.Size([14, 2048])

        return features


import collections
from torchvision.transforms import functional as F
from torch.nn import functional as tF

Iterable = collections.abc.Iterable

class Resize(object):
    """
    Args:
        size : (width, height)
    """
    def __init__(self, size, interpolation = cv2.INTER_LINEAR):
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

def parse_data(data):
    img = data[0]
    pid = data[1]
    camid = data[2]
    dsetid = data[3]
    return img, pid, camid, dsetid

def parse_data_for_eval(data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

import cv2
import os
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm
import os
import os.path as osp
import time
from torchreid.utils import AverageMeter, visualize_ranked_results


class DataLoader: #ImagePreprocessor:
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    dataset_dir = 'Test_folder'
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
                
    def preprocessor(self, root): #__call__(self, root):
        #import pdb;pdb.set_trace()        
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        query_dir = 'query/'
        gallery_dir = 'gallery/'
        
        self.query_dir = osp.join(self.dataset_dir, query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, gallery_dir)
        
        self.query_file_list = os.listdir(self.query_dir) # path 안에 있는 파일목록
        query = []
        for k in tqdm(range(len(self.query_file_list))):
            
            # BGR to RGB
            img = cv2.imread(self.query_dir + '/' + self.query_file_list[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            img = img.transpose(2,0,1)
            img = img.astype('float32') / 255.0
            
            img = torch.from_numpy(img)
            img = F.normalize(img, DataLoader.norm_mean, DataLoader.norm_std)

            query.append(img)
        self.query_tensor = torch.stack(query)
        
        self.gallery_file_list = os.listdir(self.gallery_dir) # path 안에 있는 파일목록
        gallery = []
        for k in tqdm(range(len(self.gallery_file_list))):
            
            # BGR to RGB
            img = cv2.imread(self.gallery_dir + '/' + self.gallery_file_list[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            img = img.transpose(2,0,1)
            img = img.astype('float32') / 255.0
            
            img = torch.from_numpy(img)
            img = F.normalize(img, DataLoader.norm_mean, DataLoader.norm_std)

            gallery.append(img)
        self.gallery_tensor = torch.stack(gallery)
        
        #return (self.query_tensor, self.gallery_tensor)

        """ 
        [0] : tensor(impaths)
        [1] : int(pids) 
        [2] : str(camids) 
        [3] : int(dsetids)
        """    
        query_parse = []
        for file in self.query_file_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                query_parse.append((self.query_dir + file, int(filename.split('_')[1]), filename.split('_')[2],int(filename.split('_')[3])))
        
        
        qimpaths, qpids, qcamids, qdsetids = [], [], [], []
        for i in range(len(query_parse)):
            qimg, qpid, qcamid, qdsetid = parse_data(query_parse[i])
            qimpaths.append(qimg)
            qpids.append(qpid)
            qcamids.append(qcamid)
            qdsetids.append(qdsetid)
        qpids = torch.as_tensor(qpids)
        qdsetids = torch.as_tensor(qdsetids)
        query_list = [self.query_tensor, qpids, qcamids, qimpaths, qdsetids]
                
                
        gallery_parse = []
        for file in self.gallery_file_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                gallery_parse.append((self.gallery_dir + file, int(filename.split('_')[1]), filename.split('_')[2], int(filename.split('_')[3])))
        
        
        gimpaths, gpids, gcamids, gdsetids = [], [], [], []
        for i in range(len(gallery_parse)):
            gimg, gpid, gcamid, gdsetid = parse_data(gallery_parse[i])
            gimpaths.append(gimg)
            gpids.append(gpid)
            gcamids.append(gcamid)
            gdsetids.append(gdsetid)
        #import pdb;pdb.set_trace()
        gpids = torch.as_tensor(gpids)
        gdsetids = torch.as_tensor(gdsetids)
        gallery_list = [self.gallery_tensor, gpids, gcamids, gimpaths, gdsetids]
        
        
        key_list = ['img','pid','camid','impath','dsetid']
        
        query_loader = dict(zip(key_list, query_list))
        gallery_loader = dict(zip(key_list, gallery_list))
        data_loader = (query_loader, gallery_loader)
        
        return data_loader
        
        
    
    def feature_extraction(self, model, dataloader):
        self.extractor = FeatureExtractor(
            model_name=model,
            model_path='/home/cresprit/log/resnet50/model/model.pth.tar-60',
            device='cuda'
            )
        
        self.query_loader, self.gallery_loader = dataloader
        
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            #import pdb;pdb.set_trace()
            f_ = [] #, pids_, camids_ = [], [], []
        #for batch_idx, data in enumerate(data_loader):
            imgs, pids, camids = parse_data_for_eval(data_loader)
            imgs = imgs.cuda()
            end = time.time()
            features = self.extractor(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            f_.append(features)
            #pids_.extend(pids)             # CMC, mAP 계산할때 씀
            #camids_.extend(camids)         # CMC, mAP 계산할때 씀
            f_ = torch.cat(f_, 0)
            #pids_ = np.asarray(pids_)      # CMC, mAP 계산할때 씀
            #camids_ = np.asarray(camids_)  # CMC, mAP 계산할때 씀
            return f_ #, pids_, camids_     # CMC, mAP 계산할때 씀
        
        self.qf = _feature_extraction(self.query_loader)  # ,q_pids, q_camids
        self.gf = _feature_extraction(self.gallery_loader) # , g_pids, g_camids 
        feature = [self.qf, self.gf]
        
        return feature
    
    
    def compute_distance(self, feature):
        self.qf, self.gf = feature
        distmat = compute_distance_matrix(self.qf, self.gf, 'euclidean')
        distmat = distmat.numpy()
        
        return distmat
    
    
    def dataset(self):
        query_d = []
        for file in self.query_file_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                query_d.append((self.query_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2]), int(filename.split('_')[3])))
                
        gallery_d = []
        for file in self.gallery_file_list:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                gallery_d.append((self.gallery_dir + file, int(filename.split('_')[1]), int(filename.split('_')[2]), int(filename.split('_')[3])))
        
        return (query_d, gallery_d)       
     
        
"""-------------------------------------------------------------------"""
datamanager = DataLoader(256, 128)
data_loader = datamanager.preprocessor('reid-data')
feature = datamanager.feature_extraction('resnet50', data_loader)

dataset = datamanager.dataset()
distmat = datamanager.compute_distance(feature)

print("\n")

print("distmat :", distmat)
#print("dataset :", dataset)
print("--------------------------------")

save_dir='./reid-data/visrank_Test_dataset/'
visualize_ranked_results(
    distmat,
    dataset,
    data_type='image',
    save_dir=save_dir,
    topk=5
    )