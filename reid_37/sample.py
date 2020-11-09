# import os
# import os.path as osp
# dataset_dir = 'Test_folder'
# root = 'reid-data'
# root = osp.abspath(osp.expanduser(root))
# dataset_dir = osp.join(root, dataset_dir)
# #import pdb;pdb.set_trace()
# q_dir = 'query/'
# g_dir = 'gallery/'

# q_dir = osp.join(dataset_dir, q_dir)
# g_dir = osp.join(dataset_dir, g_dir)
# print(q_dir)
#%%
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
#         import pdb;pdb.set_trace()
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
#                 query.append([q_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])
                
#         gallery = []
#         g_files = os.listdir(g_dir)
#         for file in g_files:
#             if file.endswith(".jpg"):
#                 filename = file.split('.')[0]
#                 gallery.append([g_dir + file, int(filename.split('_')[1]), filename.split('_')[0]])

#         super(NewDataset, self).__init__(train, query, gallery, **kwargs)
        
# torchreid.data.register_image_dataset('Test2', NewDataset)
#%%
import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    #workers=0,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
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

#torchreid.utils.load_pretrained_weights(model, '/home/cresprit/log/resnet50/model/model.pth.tar-60')

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
    #visrank=True,
    #visrank_topk=5
)