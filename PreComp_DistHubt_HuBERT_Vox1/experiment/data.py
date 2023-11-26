import random
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from exp_lib import augment, util, dataset

class DataManager():
    '''Initialize and control all datasets & dataloaders used in the experiment. 
    '''
    def __init__(self, args):
        self.total_epoch = args['epoch']
        self.libri100 = dataset.Libri_100h(args['path_libri'])
        self.vox1 = dataset.VoxCeleb1(args['path_train'], args['path_test'], args['path_trials'])
        
        # train loader
        self.train_set_kd = TrainSetKD(
            self.libri100.train_set, 
            self.vox1.train_set, 
            args['crop_size'],
        )
        self.train_sampler_kd = DistributedSampler(self.train_set_kd, shuffle=True)
        self.train_loader_kd = DataLoader(
            self.train_set_kd, 
            num_workers=args['num_workers'],
            batch_size=args['batch_size'],
            pin_memory=True,
            sampler=self.train_sampler_kd,
            drop_last=True
        )
        
        self.train_set_ft = TrainSetFT( 
            self.vox1.train_set, 
            args['crop_size'],
        )
        self.train_sampler_ft = DistributedSampler(self.train_set_ft, shuffle=True)
        self.train_loader_ft = DataLoader(
            self.train_set_ft, 
            num_workers=args['num_workers'],
            batch_size=args['batch_size'],
            pin_memory=True,
            sampler=self.train_sampler_ft,
            drop_last=True
        )
        
        # test loader
        self.test_set_O = EnrollmentSet(self.vox1.test_set)
        self.test_loader_O = DataLoader(
            self.test_set_O,
            num_workers=args['num_workers'] * 2,
            batch_size=1,
            sampler=DistributedSampler(self.test_set_O, shuffle=False),
            pin_memory=True,
        )
        
        self.test_set_O_TTA = TTAEnrollmentSet(self.vox1.test_set, args['num_seg'], args['seg_size'])
        self.test_loader_O_TTA = DataLoader(
            self.test_set_O_TTA,
            num_workers=args['num_workers'] * 2,
            batch_size=args['batch_size'] // args['num_seg'],
            sampler=DistributedSampler(self.test_set_O_TTA, shuffle=False),
            pin_memory=True,
        )

    def set_epoch(self, epoch):
        self.train_sampler_kd.set_epoch(epoch)
        self.train_sampler_ft.set_epoch(epoch)
        self.train_set_kd.shuffle_kd_items()

class TrainSetKD(Dataset):
    def __init__(self, items_kd, items_ft, size):
        self.items_kd = items_kd
        self.items_ft = items_ft
        self.crop_size = size

    def __len__(self):
        return len(self.items_ft)

    def __getitem__(self, index):
        # sample
        item_kd = self.items_kd[index % len(self.items_kd)]

        # read wav
        wav_kd = util.rand_crop_read(item_kd.path, self.crop_size)

        return wav_kd

    def shuffle_kd_items(self):
        random.shuffle(self.items_kd)
            
class TrainSetFT(Dataset):
    def __init__(self, items_ft, size):
        self.items_ft = items_ft
        self.crop_size = size

    def __len__(self):
        return len(self.items_ft)

    def __getitem__(self, index):
        # sample 
        item_ft = self.items_ft[index]
        
        # read wav
        wav_ft = util.rand_crop_read(item_ft.path, self.crop_size)
        
        return wav_ft, item_ft.label

class EnrollmentSet(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        wav, _ = sf.read(item.path)
        
        return wav, item.key
    
class TTAEnrollmentSet(Dataset):
    def __init__(self, items, num_seg, seg_size):
        self.items = items
        self.num_seg = num_seg
        self.seg_size = seg_size
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        wav_TTA = util.linspace_crop_read(item.path, self.num_seg, self.seg_size)
        
        return wav_TTA, item.key