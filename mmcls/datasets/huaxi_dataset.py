from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import random
import os
import numpy as np
import pandas as pd
from .builder import DATASETS
import csv
import pdb
import pickle

def load_feats(result_path):
    result_dict = pickle.load(open(result_path, 'rb'))
    file_list = result_dict['filenames']
    file_dict = {}
    for idx, filename in enumerate(file_list):
        file_dict[filename] = idx
    result_dict['filenames'] = file_dict
    return result_dict

@DATASETS.register_module()
class huaxiCTDataset(BaseDataset):
    CLASSES = ['慢阻肺', '支扩','气胸','肺炎','间质性肺炎','肺结核','积液','肺癌']
 
    def __init__(self, data_prefix, pipeline, feat_path=None, ann_file=None, sub_set=None, test_mode=False, use_sid_sampler=False):
        self.feat_path = feat_path
        super(huaxiCTDataset, self).__init__(data_prefix, pipeline, ann_file, sub_set, test_mode, use_sid_sampler)

    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = self.ann_file
        with open(ann_file,'r') as f:
            tmp=csv.reader(f)
            lines = []
            for i in tmp:
                lines.append(i)
        lines=lines[1:]
        data_infos = []
        for index in range(len(lines)):
            ID, path, label = lines[index]
            label = label[1:-1]
            labels = []
            for i in label.split(','):
                labels.append(int(i))
            #filename = os.path.join(self.data_prefix,'data_outputsize_64_256_256', path,'norm_image.npz')
            filename = os.path.join(path,'norm_image.npz')
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array((labels), dtype=np.int64)
            data_infos.append(data_info)
        # load feats
        if self.feat_path is None:
            return data_infos
        feat_dict = load_feats(self.feat_path)
        for data_info in data_infos:
            feat_idx = feat_dict['filenames'][data_info['img_info']['filename']]
            feat = feat_dict['feats'][feat_idx, :]
            data_info['aux_feat'] = feat
        return data_infos

    def __len__(self):
        if self.use_sid_sampler:
            sub_dirs = [data_info['img_info']['filename'] for data_info in self.data_infos]
            sids = ['/'.join(sub_dir.split('/')[:2]) for sub_dir in sub_dirs]
            sids = list(set(sids))
            length = len(sids)
        else:
            length = len(self.data_infos)

        return length 

@DATASETS.register_module()
class huaxiDRDataset(BaseDataset):
    CLASSES = ['慢阻肺', '支扩','气胸','肺炎','间质性肺炎','肺结核','积液','肺癌']

    def __init__(self, data_prefix, pipeline, feat_path=None, ann_file=None, sub_set=None, test_mode=False, use_sid_sampler=False):
        self.feat_path = feat_path
        super(huaxiDRDataset, self).__init__(data_prefix, pipeline, ann_file, sub_set, test_mode, use_sid_sampler)

    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = self.ann_file
        with open(ann_file,'r') as f:
            tmp=csv.reader(f)
            lines = []
            for i in tmp:
                lines.append(i)
        lines=lines[1:]
        data_infos = []
        for index in range(len(lines)):
            ID, path, label = lines[index]
            label = label[1:-1]
            labels = []
            for i in label.split(','):
                labels.append(int(i))
            #filename = os.path.join(self.data_prefix,'data_outputsize_64_256_256', path,'norm_image.npz')
            #print(self.data_prefix, path)
            filename = path
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array((labels), dtype=np.int64)
            data_infos.append(data_info)
        if self.feat_path is None:
            return data_infos
        feat_dict = load_feats(self.feat_path)
        for data_info in data_infos:
            feat_idx = feat_dict['filenames'][data_info['img_info']['filename']]
            feat = feat_dict['feats'][feat_idx, :]
            data_info['aux_feat'] = feat
        return data_infos


@DATASETS.register_module()
class huaxiCT8Dataset(huaxiCTDataset):
    CLASSES = ['慢性阻塞性肺疾病', '支气管扩张', '气胸', '肺恶性肿瘤', '肺炎', '肺结核', '胸腔积液', '间质性肺疾病']

@DATASETS.register_module()
class huaxiCTlesion20Dataset(huaxiCTDataset):
    CLASSES = [ '气胸', '液气胸', '肺不张', '肺含气支气管征', '肺大疱', '肺实变影', '气道病变', '肺斑片影', '肺条索影', '肺气肿', '肺磨玻璃密度影', '肺空洞', '肺网格影', '肺蜂窝影', '胸腔积液', '胸膜增厚', '淋巴结肿大', '钙化',  '肿块', '结节']

@DATASETS.register_module()
class huaxiDR8Dataset(huaxiDRDataset):
    CLASSES = ['慢性阻塞性肺疾病', '支气管扩张', '气胸', '肺恶性肿瘤', '肺炎', '肺结核', '胸腔积液', '间质性肺疾病']

@DATASETS.register_module()
class huaxiDRlesion18Dataset(huaxiDRDataset):
    CLASSES = ['气胸', '液气胸', '肺不张', '肺含气支气管征', '肺大疱', '肺实变影', '肺斑片影', '肺条索影', '肺气肿', '肺磨玻璃密度影', '肺空洞', '肺网格影', '肺蜂窝影', '胸腔积液', '胸膜增厚', '钙化',  '肿块', '结节']
