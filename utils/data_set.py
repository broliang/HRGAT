from torch.utils.data import Dataset
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        # , ent2textvector, rel2textvector, ent2attr, ent2attrlabel
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent
        # self.ent2textvector = ent2textvector
        # self.rel2textvector = rel2textvector
        # self.ent2attr = ent2attr
        # self.ent2attrlabel = ent2attrlabel
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        # triple_reverse, label_reverse = torch.tensor(ele['triple_reverse'], dtype=torch.long), np.int32(ele['label_reverse'])
        head, rel, tail = triple
        # head_reverse, rel_reverse, tail_reverse = triple_reverse

        label = self.get_label(label)
        # label_reverse = self.get_label(label_reverse)
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)
            # label_reverse = (1.0 - self.label_smooth) * label_reverse + (1.0 / self.num_ent)
        return triple, label
            # , triple_reverse, label_reverse
            # , (self.ent2textvector[head],self.ent2textvector[rel],self.ent2textvector[tail]), \
            #    (self.ent2attr[head], self.ent2attr[tail]), (self.ent2attrlabel[head],self.ent2attrlabel[tail])

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_ent = num_ent
        self.num_ent = num_ent
        # self.ent2textvector = ent2textvector
        # self.rel2textvector = rel2textvector
        # self.ent2attr = ent2attr
        # self.ent2attrlabel = ent2attrlabel
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label
            # , (self.ent2textvector[head],self.ent2textvector[rel],self.ent2textvector[tail]), \
            #    (self.ent2attr[head], self.ent2attr[tail]), (self.ent2attrlabel[head],self.ent2attrlabel[tail])

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)