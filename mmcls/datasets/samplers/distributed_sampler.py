import torch
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)

class DistributedSIDSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        data_infos = dataset.data_infos
        self.sub_dirs = [data_info['img_info']['filename'] for data_info in data_infos]   
        self.sids = ['/'.join(sub_dir.split('/')[:2]) for sub_dir in self.sub_dirs]
        self.sids = list(set(self.sids))
        print(len(self.sids), self.num_samples * self.num_replicas)
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.sids)

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        exist_sid = []
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            total_indices = torch.randperm(len(self.sub_dirs), generator=g).tolist()
        else:
            total_indices = torch.arange(len(self.sub_dirs)).tolist()

        for idx in total_indices:
            sub_dir = self.sub_dirs[idx]
            sid = '/'.join(sub_dir.split('/')[:2])
            if sid not in exist_sid:
                indices.append(idx)
                exist_sid.append(sid)
        print(len(indices), self.total_size, len(total_indices))
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)
