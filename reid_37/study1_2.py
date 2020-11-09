### Feature_extractor
from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model
import cv2



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
"""###################################################################################################################################"""
### utils
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
    """
    input1 = query_feature / tensor([[, .., ..]])
    input2 = gallery_feature / tensor([[, .., ..]])
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m,n)            # torch.Size([7, 55])
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n,m).t()        # torch.Size([7, 55])
    distmat = mat1 + mat2
    distmat.addmm_(1,-2,input1,input2.t())                                      # addmm_ how?
    return distmat                                                              # torch.Size([7, 55])
    # Warning
    # addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)  → addmm_(Tensor mat1, Tensor mat2, *, Number beta, Number alpha)
    # distmat.addmm_(input1, input2.t(), *, 1, -2)
    

def cosine_distance(input1, input2):
    input1_normed = tF.normalize(input1, p=2, dim=1)
    input2_normed = tF.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

"""###################################################################################################################################"""
### Viusalizer
import numpy as np
import shutil
import os.path as osp
import cv2

from torchreid.utils.tools import mkdir_if_missing

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


def visualize_ranked_result(
    distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_path = query[q_idx][:]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(
                save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
            )
            mkdir_if_missing(qdir)


        rank_idx = 1
        #import pdb;pdb.set_trace()
        for g_idx in indices[q_idx, :]:
            gimg_path = gallery[g_idx][:]

            if data_type == 'image':
                border_color = BLUE
                gimg = cv2.imread(gimg_path)
                gimg = cv2.resize(gimg, (width, height))
                gimg = cv2.copyMakeBorder(
                    gimg,
                    BW,
                    BW,
                    BW,
                    BW,
                    cv2.BORDER_CONSTANT,
                    value=border_color
                )
                gimg = cv2.resize(gimg, (width, height))
                start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                end = (
                    rank_idx+1
                ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                grid_img[:, start:end, :] = gimg

            rank_idx += 1
            if rank_idx > topk:
                break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))

"""###################################################################################################################################"""
### reid
import cv2
import os
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm
import os
import os.path as osp
import time
from torchreid.utils import AverageMeter, visualize_ranked_results


class Image_reid:
    """
    OpenCV based
    """
    def dataset_register(self, root, dataset_dir):
        
        """
        ex) root = 'reid-data'
        ex) dataset_dir = 'Test_folder'
        """
        self.root = osp.abspath(osp.expanduser(root))           # '/home/cresprit/reid-data'
        self.dataset_dir = osp.join(self.root, dataset_dir)     # '/home/cresprit/reid-data/Test_folder2'
        
        query_dir = 'query/'
        gallery_dir = 'gallery/'
        
        
        self.query_dir = osp.join(self.dataset_dir, query_dir)      # '/home/cresprit/reid-data/Test_folder2/query/'
        self.gallery_dir = osp.join(self.dataset_dir, gallery_dir)  # '/home/cresprit/reid-data/Test_folder2/gallery/'
        
        self.query_file_list = os.listdir(self.query_dir) # path 안에 있는 파일목록 / ex. ['0004_c5s3_066212_00.jpg', ...
        self.gallery_file_list = os.listdir(self.gallery_dir) # path 안에 있는 파일목록 / ex. ['0004_c2s3_059152_00.jpg', ...

    
        self.query = []
        for file in self.query_file_list:   # file = '0004_c5s3_066212_00.jpg' ...
            if file.endswith((".jpg",".png")):
                self.query.append(self.query_dir + file)
                
                # query
                # ['/home/cresprit/reid-data/Test_folder2/query/0004_c5s3_066212_00.jpg',
                #  '/home/cresprit/reid-data/Test_folder2/query/0001_c3s1_000551_00.jpg', ...]

               
        self.gallery = []
        for file in self.gallery_file_list:
            if file.endswith((".jpg",".png")):
                self.gallery.append(self.gallery_dir + file)  
                
                # gallery
                # ['/home/cresprit/reid-data/Test_folder2/gallery/0004_c2s3_059152_00.jpg',
                #  '/home/cresprit/reid-data/Test_folder2/gallery/0005_c6s1_004576_00.jpg', ...]
        
        
        dataset = (self.query, self.gallery)  # tuple(list, list)
        return dataset
        
                
    def preprocessor(self, width, height): #__call__(self, root):
        self.width = width
        self.height = height
        
        norm_mean = [0.485, 0.456, 0.406]   # from ImageNet
        norm_std = [0.229, 0.224, 0.225]    # from ImageNet
        
        def _preprocessor(file_list):
            #self.file_list = file_list
            
            data = []
            for k in tqdm(range(len(file_list))):
                
                img = cv2.imread(file_list[k])
                #import pdb;pdb.set_trace()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB__why?
                img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
                
                img = img.transpose(2,0,1)                  # numpy_array HWC → torch_tensor CHW__why?
                img = img.astype('float32') / 255.0         # __why?
                
                img = torch.from_numpy(img)                 # torch_tensor
                img = F.normalize(img, norm_mean, norm_std) # normalize__why?
    
                data.append(img)
            preprocess_tensor = torch.stack(data)           # batch, GPU parallel process
            return preprocess_tensor                        # torch.Tensor([N, C, H, W]) / N = batch
        
        pq = _preprocessor(self.query)          # torch.Tensor[N, C, H, W]
        pg = _preprocessor(self.gallery)        # torch.Tensor[N, C, H, W]
        
        preprocess_data = (pq, pg)              # (torch.Tensor, torch.Tensor)
        return preprocess_data
        
        
    
    def feature_extraction(self, model, preprocess_data):
        """
        model = 'resnet50'
        preprocess_data : tuple(torch.Tensor(), torch.Tensor())
        """
        # build
        extractor = FeatureExtractor(
            model_name=model,
            model_path='/home/cresprit/log/resnet50/model/model.pth.tar-60',
            device='cuda'
            )
        
        self.query_tensor, self.gallery_tensor = preprocess_data
        
        batch_time = AverageMeter()

        def _feature_extraction(preprocessed_tensor):
            f_ = []
            imgs = preprocessed_tensor
            imgs = imgs.cuda()
            end = time.time()
            features = extractor(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            f_.append(features)
            f_ = torch.cat(f_, 0)
            return f_
        
        self.qf = _feature_extraction(self.query_tensor)  # ,q_pids, q_camids
        self.gf = _feature_extraction(self.gallery_tensor) # , g_pids, g_camids 
        feature = [self.qf, self.gf]
        
        return feature
    
    
    def compute_distance(self, feature):
        """
        qf : query_feature
        gf : gallery_feature
        """
        self.qf, self.gf = feature
        distmat = compute_distance_matrix(self.qf, self.gf, 'euclidean')
        distmat = distmat.numpy()
        
        return distmat           
"""-------------------------------------------------------------------"""
test2 = Image_reid()

# ([query], [gallery])
dataset = test2.dataset_register(
    root = 'reid-data', 
    dataset_dir = 'Test_folder4'
    )

# (torch.Tensor(), torch.Tensor())
preprocessed_data = test2.preprocessor(128, 256)

# [torch.Tensor([N, 2048]), torch.Tensor([N, 2048])]
feature = test2.feature_extraction('resnet50', preprocessed_data)

# numpy_array distance
distmat = test2.compute_distance(feature)

print("\n")
#print("distmat :", distmat)
print("-------------------------------------------------------------------")

save_dir='./reid-data/visrank_Test_dataset/'
visualize_ranked_result(
    distmat,
    dataset,
    data_type='image',
    save_dir=save_dir,
    topk=4
    )