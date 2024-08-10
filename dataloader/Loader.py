import torch
import torchvision.transforms as tfs
from torch.utils.data.distributed import DistributedSampler


class read_data(torch.utils.data.Dataset):
    def __init__(self, rhos_pathes, bathy_patches, info_, transform=None):
        super(read_data, self).__init__()

        self.data = rhos_pathes
        self.gt = bathy_patches
        self.infos = info_
        self.transform = transform

    def __getitem__(self, item):
        data = torch.tensor(self.data[item], dtype=torch.float32)
        gt = torch.tensor(self.gt[item], dtype=torch.float32)

        data_gt_c = torch.cat([data, gt.unsqueeze(-1)], dim=-1)
        # transfer data from HxWx(C+1) to (C+1)xHxW
        data_gt_c = torch.permute(data_gt_c, [2, 0, 1])

        data_gt_c = self.transform(data_gt_c)

        data = data_gt_c[:-1, :, :]
        gt = data_gt_c[-1, :, :]
        info_ = self.infos[item]

        return data, gt, info_

    def __del__(self):
        del self.data, self.gt, self.infos
        return

    def __len__(self):
        return len(self.data)


'''
def dataset_creator(data, batch_size, radius, is_training=True, is_distributed=False):
    if is_training:
        transform = tfs.Compose([
            tfs.RandomVerticalFlip(p=0.5),
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.RandomApply([tfs.RandomRotation(45)], p=0.5),
            tfs.CenterCrop(2 * radius + 1)
        ])
    else:
        transform = tfs.Compose([
            tfs.CenterCrop(2 * radius + 1)
        ])

    # 此处仅读取data, bathy and info
    dataset = read_data(data.data, data.bathy, data.info_, transform=transform)
    # get the geo-information
    geo_info = data.info_list

    # 构建数据集
    if is_distributed:
        sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                                                 pin_memory=True)

        return dataloader, sampler, geo_info

    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=0,
                                                 pin_memory=True)

        return dataloader, geo_info
'''


class dataset_creator():
    def __init__(self, data, batch_size, radius, is_training, is_distributed=False):
        super(dataset_creator, self).__init__()
        if is_training:
            self.transform = tfs.Compose([
                tfs.RandomVerticalFlip(p=0.5),
                tfs.RandomHorizontalFlip(p=0.5),
                tfs.RandomApply([tfs.RandomRotation(45)], p=0.5),
                tfs.CenterCrop(2 * radius + 1)
                ])
        else:
            self.transform = tfs.Compose([tfs.CenterCrop(2 * radius + 1)])
        self.data = data
        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.radius = radius
        self.is_training = is_training
        
        # 此处仅读取data, bathy and info
        self.dataset = read_data(self.data.data, data.bathy, data.info_, transform=self.transform)
        self.geo_info = self.data.info_list
        
        dataloader = None
        sampler = None

        # 构建数据集
        if self.is_distributed:
            sampler = DistributedSampler(self.dataset)
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                                                     pin_memory=True)
        else:
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=is_training, num_workers=0,
                                                     pin_memory=True)
                                                 
        self.dataloader = dataloader
        self.sampler = sampler
        
    
    def update(self, data):
        self.dataset = None
        self.dataset = read_data(self.data.data, data.bathy, data.info_, transform=self.transform)
        self.geo_info = self.data.info_list
        if self.is_distributed:
            self.sampler = DistributedSampler(self.dataset)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=0,
                                                          pin_memory=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.is_training, num_workers=0,
                                                          pin_memory=True)
    
    def __del__(self):
        
        return
        
    
        