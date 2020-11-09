#%% utils
import cv2
import collections
import torchreid
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.nn import functional as tF
import numpy as np

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
#%% build_transforms
import torch

def build_transforms(
    height,
    width,
    transforms= None, #'random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)
    
    #print('Building train transforms ...')
    transform_tr = []

    #print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]


    #print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    #print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    transform_tr = T.Compose(transform_tr)
    
    print('Building test transforms ...')
    print('+ resize to (width x height = {}x{})'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = T.Compose([
        Resize((width, height)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te
#%% DataManager
from __future__ import division, print_function, absolute_import

class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader
#%% ImageDataManager
from torchreid.data.datasets import init_image_dataset

class ImageDataManager(DataManager):
    
    data_type = 'image'

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        #train_sampler='RandomSampler',
        #train_sampler_t='RandomSampler',
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False
    ):

        super(ImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )
        
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)
        
        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams
        
        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
                
        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source images   : {}'.format(len(queryset)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        print('  *****************************************')
        print('\n')
#%% ComputeDistanceMatrix

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

def parse_data_for_eval(self, data):
    imgs = data['img']
    pids = data['pid']
    camids = data['camid']
    return imgs, pids, camids
# def parse_data_for_eval(data):
#     imgs = data[0]
#     pids = data[1]
#     camids = data[2]
#     return imgs, pids, camids

#%% Engine.run
from __future__ import division, print_function, absolute_import
import time
import os.path as osp
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.utils import (
    AverageMeter, re_ranking, visualize_ranked_results
)
#from torchreid.losses import DeepSupervision


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.epoch = 0

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()
                
                
    def run(
        self,
        save_dir='log',
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20]
    ):  

        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            return
        
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)
#%% test_feature-extraction
    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20]
    ):
        
        self.set_model_mode('eval')
        
        targets = list(self.test_loader.keys())
    
        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            
            batch_time = AverageMeter()
            
            def _feature_extraction(data_loader):
                f_, pids_, camids_ = [], [], []
                for batch_idx, data in enumerate(data_loader):
                    imgs, pids, camids = self.parse_data_for_eval(data)
                    if self.use_gpu:
                        imgs = imgs.cuda()
                    end = time.time()
                    features = self.extract_features(imgs)
                    batch_time.update(time.time() - end)
                    features = features.data.cpu()
                    f_.append(features)
                    pids_.extend(pids)
                    camids_.extend(camids)
                f_ = torch.cat(f_, 0)
                pids_ = np.asarray(pids_)
                camids_ = np.asarray(camids_)
                return f_, pids_, camids_
            
            print('Extracting features from query set ...')
            qf, q_pids, q_camids = _feature_extraction(query_loader)
            print('Done, obtained {}-by-{} mtrix'.format(qf.size(0), qf.size(1)))
    
            print('Extracting features from gallery set ...')
            gf, g_pids, g_camids = _feature_extraction(gallery_loader)
            print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
    
            print('Speed: {:.4f} sec/batch'.format(batch_time.avg))
    
            if normalize_feature:
                print('Normalzing features with L2 norm ...')
                qf = tF.normalize(qf, p=2, dim=1)
                gf = tF.normalize(gf, p=2, dim=1)
    
            print(
                'Computing distance matrix with metric={} ...'.format(dist_metric)
            )
            
            
            distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
            distmat = distmat.numpy()
    
            # if rerank:
            #     print('Applying person re-ranking ...')
            #     distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            #     distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            #     distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    
            print('** Results **')
    
            if visrank:
                visualize_ranked_results(
                    distmat,
                    self.datamanager.fetch_test_loaders(name),
                    self.datamanager.data_type,
                    width=self.datamanager.width,
                    height=self.datamanager.height,
                    save_dir=osp.join(save_dir, 'visrank_' + name),
                    topk=visrank_topk
                )
                
    def extract_features(self, input):
        return self.model(input)
#%% NewDataset
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import torchreid
from torchreid.data import ImageDataset


class NewDataset(ImageDataset):
    dataset_dir = 'Test_folder'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        q_dir = 'query/'
        g_dir = 'gallery/'
        
        q_dir = osp.join(self.dataset_dir, q_dir)
        g_dir = osp.join(self.dataset_dir, g_dir)
        
        train = [['temp.jpg',0,0]]
        
        query = []
        q_files = os.listdir(q_dir)
        for file in q_files:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                query.append([q_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])
                
        gallery = []
        g_files = os.listdir(g_dir)
        for file in g_files:
            if file.endswith(".jpg"):
                filename = file.split('.')[0]
                gallery.append([g_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

torchreid.data.register_image_dataset('Test_dataset3', NewDataset)
#%% test
import pdb; pdb.set_trace()
datamanager = ImageDataManager(
    root='reid-data',
    sources='Test_dataset3',
    targets='Test_dataset3',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=751,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/resnet50',
    test_only=True,
    dist_metric='euclidean',
    visrank=True,
    visrank_topk=5
)