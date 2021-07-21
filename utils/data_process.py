from collections import defaultdict as ddict
import csv
from itertools import  islice
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import h5py
def _read_dictionary(filename):
    d = {}
    if 'id' in filename:
        with open(filename, 'r+') as f:
            num = 0
            for line in islice(f,1,None):
                line = line.strip().split('\t')
                d[line[0]] = num
                num += 1
    else:
        with open(filename, 'r+') as f:
            for id, line in enumerate(f):
                line = line.strip()
                d[line] = int(id)
    d = {k: v for k, v in sorted(d.items(), key=lambda kv: (kv[1], kv[0]))}
    return d

def to_unicode(input):
    # FIXME (lingfan): not sure about python 2 and 3 str compatibility
    return str(input)
    """ lingfan: comment out for now
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')
    """

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

def _read_same_link(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split(' ')
            d[line[2]] = line[0]
    return d


class RGCNLinkDataset(object):
    """RGCN link prediction dataset
    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18
    The original knowledge base is stored as an RDF file, and this class will
    download and parse the RDF file, and performs preprocessing.
    An object of this class has 5 member attributes needed for link
    prediction:
    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing
    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).
    Examples
    --------
    Load FB15k-237 dataset
    >>> from dgl.contrib.data import load_data
    >>> data = load_data(dataset='FB15k-237')
    """

    def __init__(self, name):
        self.name = name
        self.dir = './data'
        self.dir = os.path.join(self.dir, self.name)

    def _read_text(self):
        self.ent2text = {}
        self.ent2text_short = {}
        with open(os.path.join(self.dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    self.ent2text_short[temp[0]] = temp[1]  # [:end]

        if self.dir.find("FB15") != -1:
            with open(os.path.join(self.dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    # self.ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
                    self.ent2text[temp[0]] = temp[1].split('.')[0]  # [:first_sent]

        self.entities_short = list(self.ent2text_short.keys())
        self.entities = list(self.ent2text.keys())
        self.rel2text = {}
        with open(os.path.join(self.dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                self.rel2text[temp[0]] = temp[1]

    def _laod_vgg(self):
        self.vgg_feature = h5py.File('/home/liangshuang/NewWork/data/FB15k-237/FB15K_ImageData.h5')
        self.img_index = {}
        with open(os.path.join('./data/FB15k-237/FB15K_ImageIndex.txt'), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                self.img_index[temp[0]] = temp[1]  # [:end]
        self.ent2imgvector = []
        for k,v in self.entity_dict.items():
            if k in self.img_index.keys():
                self.ent2imgvector.append(self.vgg_feature[self.img_index[k]])
            else:
                self.ent2imgvector.append(np.zeros((1,4096)))

    def _read_numerical(self):
        self.ent2num = {}
        self.attributes = []
        with open(os.path.join(self.dir, "FB15K_NumericalTriples.txt"), 'r') as f:
            num_lines = f.readlines()
            for line in num_lines:
                temp = line.strip().split('\t')
                attr = temp[1].split('/')[-1]
                self.attributes.append(attr)
        self.attributes = set(self.attributes)
        self.attributes_dict = {k: v+1 for v,k in enumerate(self.attributes)}
        self.ent2num = {k: {j: 0.0 for j in self.attributes} for k,v in self.entity_dict.items()} #every ent's all real numerical value(if don't have, its 0)
        with open(os.path.join(self.dir, "FB15K_NumericalTriples.txt"), 'r') as f:
            num_lines = f.readlines()
            for line in num_lines:
                temp = line.strip().split('\t')
                ent = temp[0]
                attr = temp[1].split('/')[-1]
                value = temp[2]
                self.ent2num[ent][attr] = 1.0
        self.ent2numpd = pd.DataFrame(self.ent2num).T  # row is ent, column is attr_name
        self.ent2attrlabel = self.ent2numpd.copy()
        self.ent2attrlabel[self.ent2attrlabel != 0] = 1.0  # which attr ent has
        # self.ent2numpd = self.ent2numpd / self.ent2numpd.max(axis = 0) #normalization num_feature to 0-1
        self.ent2attrlabel = self.ent2attrlabel.T
        if os.path.exists(os.path.join(self.dir,'attr2textvector.npz')):
            print('***********load bert attr vector successfully*************')
            self.attr2textvector = np.load(os.path.join(self.dir,'attr2textvector.npz'))
            self.attr2textvector = {k:v for k,v in zip(self.attr2textvector['x'],self.attr2textvector['y'])}
        else:
            self.bert = SentenceTransformer(
                '/home/liangshuang/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1_part')
            self.attr2textvector = {k: self.bert.encode(k.replace('.', ' ')) for k in self.attributes}
            np.savez(os.path.join(self.dir, 'attr2textvector.npz'), x=np.array(list(self.attr2textvector.keys())),
                     y=np.array(list(self.attr2textvector.values())))
        # self.ent2numpd = (self.ent2numpd - self.ent2numpd.mean(axis = 0)) / self.ent2numpd.std(axis = 0) #normalization num_feature
    def _load_bert(self):
        if os.path.exists(os.path.join(self.dir,'ent2textvector.npz')):
            print('***********load bert ent vector successfully*************')
            self.ent2textvector = np.load(os.path.join(self.dir,'ent2textvector.npz'))
            self.ent2textvector = {k:v for k,v in zip(self.ent2textvector['x'],self.ent2textvector['y'])}

            self.ent2textvector_short = np.load(os.path.join(self.dir,'ent2textvector_short.npz'))
            self.ent2textvector_short = {k:v for k,v in zip(self.ent2textvector_short['x'],self.ent2textvector_short['y'])}

        if os.path.exists(os.path.join(self.dir,'rel2textvector.npz')):
            print('***********load bert rel vector successfully*************')
            self.rel2textvector = np.load(os.path.join(self.dir,'rel2textvector.npz'))
            self.rel2textvector = {k:v for k,v in zip(self.rel2textvector['x'],self.rel2textvector['y'])}
        else:
            print('*********there is no bert vector, begin to encode text*************')
            self.bert = SentenceTransformer('/home/liangshuang/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1_part')
            self.ent2textvector = {k: self.bert.encode(v) for k,v in self.ent2text.items()}
            self.ent2textvector_short = {k: self.bert.encode(v) for k,v in self.ent2text_short.items()}
            self.rel2textvector = {k: self.bert.encode(v) for k,v in self.rel2text.items()}
            np.savez(os.path.join(self.dir,'ent2textvector.npz'), x = np.array(list(self.ent2textvector.keys())),
                     y =  np.array(list(self.ent2textvector.values())))
            np.savez(os.path.join(self.dir,'ent2textvector_short.npz'), x = np.array(list(self.ent2textvector_short.keys())),
                     y =  np.array(list(self.ent2textvector_short.values())))
            np.savez(os.path.join(self.dir,'rel2textvector.npz'), x = np.array(list(self.rel2textvector.keys())),
                     y =  np.array(list(self.rel2textvector.values())))


    def load(self):
        if os.path.exists(os.path.join(self.dir, 'entity2id.txt')) and os.path.exists(os.path.join(self.dir, 'relation2id.txt')):
            entity_path = os.path.join(self.dir, 'entity2id.txt')
            relation_path = os.path.join(self.dir, 'relation2id.txt')
        else:
            entity_path = os.path.join(self.dir, 'entities.txt')
            relation_path = os.path.join(self.dir, 'relations.txt')
        train_path = os.path.join(self.dir, 'train.tsv')
        valid_path = os.path.join(self.dir, 'dev.tsv')
        test_path = os.path.join(self.dir, 'test.tsv')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self._read_text()
        self._load_bert()
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))

        self._laod_vgg()
        self._read_numerical()
        self.ent2num = self.ent2numpd.to_numpy()
        self.attr2vector = np.array(
            [self.attr2textvector[i] for i in self.attributes])  # attribute bert vector (116, 768)
        self.attr2vector = np.matmul(self.ent2num , self.attr2vector)
        self.attrname = self.ent2numpd.columns  # attribute name
        self.ent2value = None
        self.ent2attrlabel = None

        self.ent2textvector = [self.ent2textvector[k] for k,v in self.entity_dict.items()]
        self.ent2textvector = np.array(self.ent2textvector)

        self.rel2textvector_ = []
        for k,v in self.relation_dict.items():
            if k in self.rel2textvector.keys():
                self.rel2textvector_.append(self.rel2textvector[k])
            else:
                self.rel2textvector_.append(np.zeros(768))
        self.rel2textvector = np.array(self.rel2textvector_)

        # self.ent2textvector = {self.entity_dict[k]: v for k,v in self.ent2textvector.items()}
        # self.rel2textvector = {self.relation_dict[k]: v for k,v in self.rel2textvector.items()}
        # self.attr2textvector = {self.attributes_dict[k]: v for k,v in self.attr2textvector.items()}

        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))

        # for ent,v in self.ent2num.items():
        #     temp = {}
        #     attr_label = []
        #     for attr_name, value in v.items():
        #         if value != 0:
        #             temp[attr_name] = (self.attr2textvector[attr_name], value, self.ent2numpd[attr_name][ent])
        #             attr_label.append(1)
        #         else:
        #             attr_label.append(0)

            # if len(temp) == 0:
            #     self.ent2numreal[ent] = np.zeros(768)
            # self.ent2numreal[ent] = temp
            # self.ent2attr[ent] = attr_label




class RGCNLinkDataset_DB(object):
    """RGCN link prediction dataset
    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18
    The original knowledge base is stored as an RDF file, and this class will
    download and parse the RDF file, and performs preprocessing.
    An object of this class has 5 member attributes needed for link
    prediction:
    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing
    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).
    Examples
    --------
    Load FB15k-237 dataset
    >>> from dgl.contrib.data import load_data
    >>> data = load_data(dataset='FB15k-237')
    """

    def __init__(self, name):
        self.name = name
        self.dir = '/home/liangshuang/MultiGCN/data'
        self.dir = os.path.join(self.dir, self.name)

    def _read_text(self):
        self.ent2text = {}
        self.ent2text_short = {}
        with open(os.path.join(self.dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    self.ent2text_short[temp[0]] = temp[1]  # [:end]

        with open(os.path.join(self.dir, "entity2textlong.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                # first_sent_end_position = temp[1].find(".")
                # self.ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
                self.ent2text[temp[0]] = temp[1].split('.')[0]  # [:first_sent]

        self.entities_short = list(self.ent2text_short.keys())
        self.entities = list(self.ent2text.keys())
        self.rel2text = {}
        with open(os.path.join(self.dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                self.rel2text[temp[0]] = temp[1]

    def _laod_vgg(self):
        self.vgg_feature = h5py.File(self.dir + '/' + self.name + '_ImageData.h5')
        self.img_index = {}
        with open(self.dir + '/' + self.name + '_ImageIndex.txt', 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                self.img_index[temp[0]] = temp[1]  # [:end]
        self.ent2imgvector = []
        for k,v in self.entity_dict.items():
            if k in self.img_index.keys():
                self.ent2imgvector.append(self.vgg_feature[self.img_index[k]])
            else:
                self.ent2imgvector.append(np.zeros((1,4096)))

    def _read_numerical(self):
        self.ent2num = {}
        self.attributes = []


        with open(os.path.join(self.dir + '/' + self.name + '_NumericalTriples.txt'), 'r') as f:
            num_lines = f.readlines()
            for line in num_lines:
                if '\t' in line:
                    temp = line.strip().split('\t')
                else:
                    temp = line.strip().split(' ')
                attr = temp[1].split('/')[-1]
                self.attributes.append(attr)
        self.attributes = set(self.attributes)
        self.attributes_dict = {k: v+1 for v,k in enumerate(self.attributes)}

        if os.path.exists(os.path.join(self.dir,'attr2textvector.npz')):
            print('***********load bert attr vector successfully*************')
            self.attr2textvector = np.load(os.path.join(self.dir,'attr2textvector.npz'))
            self.attr2textvector = {k:v for k,v in zip(self.attr2textvector['x'],self.attr2textvector['y'])}
        else:
            self.bert = SentenceTransformer(
                '/home/liangshuang/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1_part')
            self.attr2textvector = {k: self.bert.encode(k.replace('.', ' ')) for k in self.attributes}
            np.savez(os.path.join(self.dir, 'attr2textvector.npz'), x=np.array(list(self.attr2textvector.keys())),
                     y=np.array(list(self.attr2textvector.values())))

        self.ent2num = {k: {j: 0.0 for j in self.attributes} for k in self.entity_dict.keys()} #every ent's all real numerical value(if don't have, its 0)
        with open(os.path.join(self.dir + '/' + self.name + '_NumericalTriples.txt'), 'r') as f:
            num_lines = f.readlines()
            for line in num_lines:
                if '\t' in line:
                    temp = line.strip().split('\t')
                else:
                    temp = line.strip().split(' ')
                ent = temp[0]
                attr = temp[1].split('/')[-1]
                value = self.attr2textvector[attr]
                # if '"' in temp[2]:
                #     value = temp[2].split('"')[1]
                # elif '-' in temp[2]:
                #     value = temp[2].split('-')[1]
                self.ent2num[ent][attr] = 1.0
        self.ent2numpd = pd.DataFrame(self.ent2num).T  # row is ent, column is attr_name
        self.ent2attrlabel = self.ent2numpd.copy()
        self.ent2attrlabel[self.ent2attrlabel != 0] = 1.0 # which attr ent has
        # self.ent2numpd = self.ent2numpd / self.ent2numpd.max(axis = 0) #normalization num_feature to 0-1
        self.ent2attrlabel = self.ent2attrlabel.T

        # self.ent2numpd = (self.ent2numpd - self.ent2numpd.mean(axis = 0)) / self.ent2numpd.std(axis = 0) #normalization num_feature
    def _load_bert(self):
        if os.path.exists(os.path.join(self.dir,'ent2textvector_short.npz')):
            print('***********load bert ent vector successfully*************')

            self.ent2textvector_short = np.load(os.path.join(self.dir,'ent2textvector_short.npz'))
            self.ent2textvector_short = {k:v for k,v in zip(self.ent2textvector_short['x'],self.ent2textvector_short['y'])}

        if os.path.exists(os.path.join(self.dir,'rel2textvector.npz')):
            print('***********load bert rel vector successfully*************')
            self.rel2textvector = np.load(os.path.join(self.dir,'rel2textvector.npz'))
            self.rel2textvector = {k:v for k,v in zip(self.rel2textvector['x'],self.rel2textvector['y'])}
        else:
            print('*********there is no bert vector, begin to encode text*************')
            self.bert = SentenceTransformer('/home/liangshuang/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1_part')
            self.ent2textvector = {k: self.bert.encode(v) for k,v in self.ent2text.items()}
            self.ent2textvector_short = {k: self.bert.encode(v) for k,v in self.ent2text_short.items()}
            self.rel2textvector = {k: self.bert.encode(v) for k,v in self.rel2text.items()}
            np.savez(os.path.join(self.dir,'ent2textvector.npz'), x = np.array(list(self.ent2textvector.keys())),
                     y =  np.array(list(self.ent2textvector.values())))
            np.savez(os.path.join(self.dir,'ent2textvector_short.npz'), x = np.array(list(self.ent2textvector_short.keys())),
                     y =  np.array(list(self.ent2textvector_short.values())))
            np.savez(os.path.join(self.dir,'rel2textvector.npz'), x = np.array(list(self.rel2textvector.keys())),
                     y =  np.array(list(self.rel2textvector.values())))


    def load(self):
        if os.path.exists(os.path.join(self.dir, 'entity2id.txt')) and os.path.exists(os.path.join(self.dir, 'relation2id.txt')):
            entity_path = os.path.join(self.dir, 'entity2id.txt')
            relation_path = os.path.join(self.dir, 'relation2id.txt')
        else:
            entity_path = os.path.join(self.dir, 'entities.txt')
            relation_path = os.path.join(self.dir, 'relations.txt')
        train_path = os.path.join(self.dir, 'train.tsv')
        valid_path = os.path.join(self.dir, 'dev.tsv')
        test_path = os.path.join(self.dir, 'test.tsv')
        entity_dict = _read_dictionary(entity_path)

        relation_dict = _read_dictionary(relation_path)

        same_link = os.path.join(self.dir, self.name + '_SameAsLink.txt')
        self.same_link = _read_same_link(same_link)


        self._read_text()

        self._load_bert()

        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        self._laod_vgg()
        self._read_numerical()
        self.ent2num = self.ent2numpd.to_numpy()
        self.attr2vector = np.array(
            [self.attr2textvector[i] for i in self.attributes])  # attribute bert vector (116, 768)
        self.attr2vector = np.matmul(self.ent2num , self.attr2vector)
        self.attrname = self.ent2numpd.columns  # attribute name

        # self.ent2value = [np.array(self.ent2numpd[k]) for k, v in
        #                   self.entity_dict.items()]  # ent num value (ent_num, 116)
        # self.ent2attrlabel = [np.array(self.ent2attrlabel[k]) for k, v in
        #                       self.entity_dict.items()]  # ent num attr label (ent_num, 116)
        self.ent2value = None
        self.ent2attrlabel = None
        self.ent2textvector = []
        for k,v in self.entity_dict.items():
            if k in self.same_link.keys():
                self.ent2textvector.append(self.ent2textvector_short[self.same_link[k]])
            else:
                self.ent2textvector.append(np.zeros(768))
        # self.ent2textvector = [self.ent2textvector_short[self.same_link[k]] for k,v in self.entity_dict.items()]
        # self.rel2textvector = [self.rel2textvector[k] for k,v in self.relation_dict.items()]
        self.rel2textvector = None
        # self.ent2textvector = {self.entity_dict[k]: v for k,v in self.ent2textvector.items()}
        # self.rel2textvector = {self.relation_dict[k]: v for k,v in self.rel2textvector.items()}
        # self.attr2textvector = {self.attributes_dict[k]: v for k,v in self.attr2textvector.items()}

        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))



class RGCNLinkDataset_WN(object):
    """RGCN link prediction dataset
    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18
    The original knowledge base is stored as an RDF file, and this class will
    download and parse the RDF file, and performs preprocessing.
    An object of this class has 5 member attributes needed for link
    prediction:
    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing
    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).
    Examples
    --------
    Load FB15k-237 dataset
    >>> from dgl.contrib.data import load_data
    >>> data = load_data(dataset='FB15k-237')
    """

    def __init__(self, name):
        self.name = name
        self.dir = '/home/liangshuang/MultiGCN/data'
        self.dir = os.path.join(self.dir, self.name)

    def _read_text(self):
        self.ent2text = {}
        with open(os.path.join(self.dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                self.ent2text[temp[0]] = temp[1]  # [:end]

        self.entities = list(self.ent2text.keys())
        self.rel2text = {}
        with open(os.path.join(self.dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                self.rel2text[temp[0]] = temp[1]

    def _load_bert(self):
        if os.path.exists(os.path.join(self.dir,'ent2textvector.npz')):
            print('***********load bert ent vector successfully*************')
            self.ent2textvector = np.load(os.path.join(self.dir,'ent2textvector.npz'))
            self.ent2textvector = {k:v for k,v in zip(self.ent2textvector['x'],self.ent2textvector['y'])}

        if os.path.exists(os.path.join(self.dir,'rel2textvector.npz')):
            print('***********load bert rel vector successfully*************')
            self.rel2textvector = np.load(os.path.join(self.dir,'rel2textvector.npz'))
            self.rel2textvector = {k:v for k,v in zip(self.rel2textvector['x'],self.rel2textvector['y'])}
        else:
            print('*********there is no bert vector, begin to encode text*************')
            self.bert = SentenceTransformer('/home/liangshuang/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1_part')
            self.ent2textvector = {k: self.bert.encode(v) for k,v in self.ent2text.items()}
            self.rel2textvector = {k: self.bert.encode(v) for k,v in self.rel2text.items()}
            np.savez(os.path.join(self.dir,'ent2textvector.npz'), x = np.array(list(self.ent2textvector.keys())),
                     y =  np.array(list(self.ent2textvector.values())))
            np.savez(os.path.join(self.dir,'rel2textvector.npz'), x = np.array(list(self.rel2textvector.keys())),
                     y =  np.array(list(self.rel2textvector.values())))


    def load(self):
        entity_path = os.path.join(self.dir, 'entities.txt')
        relation_path = os.path.join(self.dir, 'relations.txt')
        train_path = os.path.join(self.dir, 'train.tsv')
        valid_path = os.path.join(self.dir, 'dev.tsv')
        test_path = os.path.join(self.dir, 'test.tsv')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self._read_text()
        self._load_bert()
        self.train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))


        self.attr2vector = None
        self.attrname = None  # attribute name
        self.ent2value = None
        self.ent2attrlabel = None

        self.ent2textvector = [self.ent2textvector[k] for k,v in self.entity_dict.items()]
        self.ent2textvector = np.array(self.ent2textvector)
        self.ent2imgvector = None
        self.rel2textvector_ = []
        for k,v in self.relation_dict.items():
            if k in self.rel2textvector.keys():
                self.rel2textvector_.append(self.rel2textvector[k])
            else:
                self.rel2textvector_.append(np.zeros(768))
        self.rel2textvector = np.array(self.rel2textvector_)

        # self.ent2textvector = {self.entity_dict[k]: v for k,v in self.ent2textvector.items()}
        # self.rel2textvector = {self.relation_dict[k]: v for k,v in self.rel2textvector.items()}
        # self.attr2textvector = {self.attributes_dict[k]: v for k,v in self.attr2textvector.items()}

        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))


def load_link(dataset):
    if 'DB' in dataset or 'YAGO' in dataset:
        data = RGCNLinkDataset_DB(dataset)
    elif 'WN' in dataset:
        data = RGCNLinkDataset_WN(dataset)
    else:
        data = RGCNLinkDataset(dataset)
    data.load()
    return data

def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
        sr2o[(obj, rel + num_rel)].add(subj)

    # for subj, rel, obj in dataset['train']:
    #     sr2o[(subj, rel)].add(obj)
    #     sr2o[(obj, rel)].add(subj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}        # sr2o[(obj, rel + num_rel)].add(subj)
    triplets = ddict(list)

    # for subj, rel, obj in dataset['train']:
    #     triplets['train'].append({'triple':(subj, rel, -1), 'label': sr2o[(subj, rel)]})
    #     triplets['train'].append({'triple_reverse':(obj, rel + num_rel, -1), 'label_reverse': sr2o[(obj, rel + num_rel)]})

    # for (subj, rel), obj in sr2o_train.items():
    #     triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})


    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})
    for (obj, rel), subj in sr2o_train.items():
        triplets['train'].append({'triple': (obj, rel, -1), 'label': sr2o_train[(obj, rel)]})


    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            triplets[f"{split}_head"].append(
                {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets




