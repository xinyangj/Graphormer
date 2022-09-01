import torch
import os
import numpy as np
import pickle
from xml.dom.minidom import parse, Node, parseString

from torch_geometric.data import Data
from .svg_parser import SVGParser
from .svg_parser import SVGGraphBuilderBezier2 as SVGGraphBuilderBezier
from sklearn.metrics.pairwise import euclidean_distances

#import networkx as nx
import math
import cv2
import random

from graphormer.utils.det_util import bbox_iou_ios_cpu, intersect_bb_idx

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .. import algos
#from a2c import a2c

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    #print(x.size(), feature_num, feature_offset, feature_offset.size())
    x = x + feature_offset
    return x

class idxTree:
    def __init__(self):
        self.children = []
        self.value = {}

class SESYDDiagramWhole(torch.utils.data.Dataset):
    def __init__(self, root,# opt, 
        partition = 'train', 
        data_aug = False, 
        do_mixup = True, 
        drop_edge = 0, 
        bbox_file_postfix = '_bb.pkl', 
        bbox_sampling_step = 5):
        super(SESYDDiagramWhole, self).__init__() 

        svg_list = open(os.path.join(root, partition + '_list.txt')).readlines()
        svg_list = [os.path.join(root, line.strip()) for line in svg_list]
        self.graph_builder = SVGGraphBuilderBezier()
        #print(svg_list)

        #self.pos_edge_th = opt.pos_edge_th
        self.data_aug = data_aug
        self.svg_list = svg_list
        self.bbox_sampling_step = bbox_sampling_step
        self.bbox_file_postfix = bbox_file_postfix
        
        stats = pickle.load(open(os.path.join(root, 'stats.pkl'), 'rb'))
        self.attr_mean = np.array([stats['angles']['mean'], stats['distances']['mean']])
        self.attr_std = np.array([stats['angles']['std'], stats['distances']['std']])

        self.normalize_bbox = True
        self.do_mixup = do_mixup
        self.class_dict = {
            'armchair':0, 
            'bed':1, 
            'door1':2, 
            'door2':3, 
            'sink1':4, 
            'sink2':5, 
            'sink3':6, 
            'sink4':7, 
            'sofa1':8, 
            'sofa2':9, 
            'table1':10, 
            'table2':11, 
            'table3':12, 
            'tub':13, 
            'window1':14, 
            'window2':15, 
            'None': 16
        }

        self.class_dict = {
            'diode2':0, 
            'capacitor2': 1, 
            'diode3': 2, 
            'earth': 3, 
            'battery1': 4, 
            'battery2': 5, 
            'core-iron': 6, 
            'outlet': 7, 
            'transistor-npn': 8, 
            'capacitor1': 9, 
            'resistor': 10, 
            'relay': 11, 
            'core-air': 12, 
            'transistor-mosfetn': 13, 
            'transistor-mosfetp': 14, 
            'core-hiron': 15, 
            'transistor-pnp': 16, 
            'diode1': 17, 
            'diodephoto': 18, 
            'gate-ampli':19, 
            'unspecified': 20, 
            'None': 21
          }

        self.n_classes = len(list(self.class_dict.keys()))
        
        '''
        self.class_dict = {
            'armchair':0, 
            'bed':1, 
            'door1':2, 
            'door2':2, 
            'sink1':3, 
            'sink2':3, 
            'sink3':3, 
            'sink4':3, 
            'sofa1':4, 
            'sofa2':4, 
            'table1':5, 
            'table2':5, 
            'table3':5, 
            'tub':6, 
            'window1':7, 
            'window2':7
        }
        '''
        #self.anchors = self.get_anchor()
        '''
        self.n_objects = 0
        for idx in range(len(self.svg_list)):
            filepath = self.svg_list[idx]
            print(filepath)
            p = SVGParser(filepath)
            width, height = p.get_image_size()
            #graph_dict = self.graph_builder.buildGraph(p.get_all_shape())

            gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
            self.n_objects += gt_bbox.shape[0]
        print(self.n_objects)
        '''
        self.n_objects = 13238

    def __len__(self):
        return len(self.svg_list)
        
    def _get_bbox(self, path, width, height):
        dom = parse(path.replace('.svg', '.xml'))
        root = dom.documentElement

        nodes = []
        for tagname in ['a', 'o']:
            nodes += root.getElementsByTagName(tagname)
        
        bbox = []
        labels = []
        for node in nodes:
            for n in node.childNodes:
                if n.nodeType != Node.ELEMENT_NODE:
                    continue
                x0 = float(n.getAttribute('x0')) / width
                y0 = float(n.getAttribute('y0')) / height
                x1 = float(n.getAttribute('x1')) / width 
                y1 = float(n.getAttribute('y1')) / height
                label = n.getAttribute('label')
                bbox.append((x0, y0, x1, y1))
                labels.append(self.class_dict[label])

        return np.array(bbox), np.array(labels)


    def __transform__(self, pos, scale, angle, translate):
        scale_m = np.eye(2)
        scale_m[0, 0] = scale
        scale_m[1, 1] = scale

        rot_m = np.eye(2)
        rot_m[0, 0:2] = [np.cos(angle), np.sin(angle)]
        rot_m[1, 0:2] = [-np.sin(angle), np.cos(angle)]

        #print(pos.shape, scale_m[0:2].shape)
        
        #print(pos.shape)
        center = np.array((0.5, 0.5))[None, :]
        pos -= center
        if random.choice([True, False]):
            pos[:, 0] = -pos[:, 0]
        if random.choice([True, False]):
            pos[:, 1] = -pos[:, 1]
        pos = np.matmul(pos, rot_m[0:2])
        pos += center
        pos += np.array(translate)[None, :]
        pos = np.matmul(pos, scale_m[0:2])
        return pos

    def __transform_bbox__(self, bbox, scale, angle, translate):
        p0 = bbox[:, 0:2]
        p2 = bbox[:, 2:]
        p1 = np.concatenate([p2[:, 0][:, None], p0[:, 1][:, None]], axis = 1)
        p3 = np.concatenate([p0[:, 0][:, None], p2[:, 1][:, None]], axis = 1)
        
        p0 = self.__transform__(p0, scale, angle, translate)
        p1 = self.__transform__(p1, scale, angle, translate)
        p2 = self.__transform__(p2, scale, angle, translate)
        p3 = self.__transform__(p3, scale, angle, translate)

        
        def bound_rect(p0, p1, p2, p3):
            x = np.concatenate((p0[:, 0][:, None], p1[:, 0][:, None], p2[:, 0][:, None], p3[:, 0][:, None]), axis = 1)
            y = np.concatenate((p0[:, 1][:, None], p1[:, 1][:, None], p2[:, 1][:, None], p3[:, 1][:, None]), axis = 1)
            x_min = x.min(1, keepdims = True)
            x_max = x.max(1, keepdims = True)
            y_min = y.min(1, keepdims = True)
            y_max = y.max(1, keepdims = True)

            return np.concatenate([x_min, y_min, x_max, y_max], axis = 1)
        return bound_rect(p0, p1, p2, p3)

    def random_transfer(self, pos, bbox, gt_bbox, bbox_targets):
        scale_ratio = 0.6
        scale = (np.random.random() * 2 - 1) * scale_ratio + 1 #np.random.random() * 0.2 + 0.9
        angle = np.random.random() * np.pi * 2

        translate_ratio = 0.1
        translate = [0, 0]
        translate[0] = (np.random.random() * 2 - 1) * translate_ratio #np.random.random() * 0.2 - 0.1
        translate[1] = (np.random.random() * 2 - 1) * translate_ratio #np.random.random() * 0.2 - 0.1

        pos = self.__transform__(pos, scale, angle, translate)
        #bbox = self.__transform_bbox__(bbox, scale, angle, translate)
        gt_bbox = self.__transform_bbox__(gt_bbox, scale, angle, translate)
        bbox_targets = self.__transform_bbox__(bbox_targets, scale, angle, translate)

        return pos, bbox, gt_bbox, bbox_targets

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filepath = self.svg_list[idx]
        #filepath = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-06/file_97.svg'
        #print(filepath)
        
        
        graph_dict = pickle.load(open(filepath.replace('.svg', '.pkl'), 'rb'))
        width, height = graph_dict['img_width'], graph_dict['img_height']
        
        gt_bbox, gt_labels = self._get_bbox(filepath, width, height)
        filename_bbox = filepath.replace('.svg', self.bbox_file_postfix)
        
        feats = np.concatenate((
            graph_dict['attr']['color'], 
            #graph_dict['attr']['stroke_width'], 
            graph_dict['pos']['spatial']), 
            axis = 1)
        #feats = graph_dict['pos']['spatial']
        pos = graph_dict['pos']['spatial']
        is_control = graph_dict['attr']['is_control']
        is_super = graph_dict['attr']['is_super']

        edge = graph_dict['edge']['shape']
        edge_super = graph_dict['edge']['super']
        e_attr = graph_dict['edge_attr']['shape']
        e_attr_super = graph_dict['edge_attr']['super']
        #e_attr[:, -2:] = (e_attr[:, -2:] - self.attr_mean) / self.attr_std
        #e_attr_super[:, -2:] = (e_attr_super[:, -2:] - self.attr_mean) / self.attr_std
        e_attr = e_attr[:, 0:4]
        e_attr_super = e_attr_super[:, 0:4]
        
        def update_bbox(pos, bbox_idx):
            #print(pos.shape, bbox_idx.shape)
            idx = [0]
            bbox = []
            for i in range(1, len(bbox_idx)):
                if bbox_idx[i] != bbox_idx[i - 1]:
                    pos_bbox = pos[idx, :]
                    max_x = pos_bbox[:, 0].max(0)
                    min_x = pos_bbox[:, 0].min(0)
                    max_y = pos_bbox[:, 1].max(0)
                    min_y = pos_bbox[:, 1].min(0)
                    bbox.append([min_x, min_y, max_x, max_y])
                    idx = [i]
                else:
                    idx.append(i)
            pos_bbox = pos[idx, :]
            max_x = pos_bbox[:, 0].max(0)
            min_x = pos_bbox[:, 0].min(0)
            max_y = pos_bbox[:, 1].max(0)
            min_y = pos_bbox[:, 1].min(0)
            bbox.append([min_x, min_y, max_x, max_y])
            return np.array(bbox)

        if self.data_aug:
            pos, bbox, gt_bbox, bbox_targets = self.random_transfer(pos, bbox, gt_bbox, bbox_targets)
            #bbox = update_bbox(pos, bbox_idx)
            #print(bbox)
            #print(bbox_targets)
            #print()
            #raise SystemExit
    
        #print(pos.shape, edge.shape, e_attr.shape, gt_bbox.shape, gt_labels.shape)

        '''
        feats = np.concatenate((
            np.zeros((pos.shape[0], 3)),
            pos), 
            axis = 1)
        '''
        feats = pos


        feats = torch.tensor(feats, dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        edge = torch.tensor(edge, dtype=torch.long)
        edge_super = torch.tensor(edge_super, dtype=torch.long)
        #is_control = torch.tensor(is_control, dtype=torch.bool)
        #is_super = torch.tensor(is_super, dtype=torch.bool)
        #bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32)
        #bbox = torch.tensor(bbox, dtype=torch.float32)
        #labels = torch.tensor(labels, dtype=torch.long)
        #has_obj = torch.tensor(has_obj, dtype=torch.long)
        gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)
        #e_weight = torch.tensor(e_weight, dtype=torch.float32)
        #e_weight_super = torch.tensor(e_weight_super, dtype=torch.float32)
        #bbox_idx = torch.tensor(bbox_idx, dtype=torch.long)
        #stat_feats = torch.tensor(stat_feats, dtype=torch.float32)
        e_attr = torch.tensor(e_attr, dtype=torch.float32)
        e_attr_super = torch.tensor(e_attr_super, dtype=torch.float32)

        #e_attr_super = torch.zeros((edge_super.size(0), 4), dtype=torch.float32)

        data = Data(x = feats, pos = pos)
        data.edge = edge
        data.edge_super = edge_super
        #data.is_control = is_control
        #data.is_super = is_super
        #data.bbox = bbox
        #data.bbox_targets = bbox_targets
        #data.labels = labels
        data.gt_bbox = gt_bbox
        data.gt_labels = gt_labels
        #data.filepath = filepath
        data.width = width
        data.height = height
        data.e_attr = e_attr
        data.e_attr_super = e_attr_super
        #data.bbox_idx = bbox_idx
        #data.stat_feats = stat_feats
        #data.has_obj = has_obj
        #data.roots = roots
        #data.e_weight = e_weight
        #data.e_weight_super = e_weight_super
        #print(data.pos)
    
        return data

    def preprocess_item(self, item):
        #print(item)
        edge_attr, edge_index, x = item.e_attr, item.edge, item.x
        edge_index = edge_index.T
        
        N = x.size(0)
        #x = convert_to_single_emb(x)

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)]) #, dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
           edge_attr #convert_to_single_emb(edge_attr) + 1
        )
        
        
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        # combine
        item.x = x
        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree  # for undirected graph
        item.edge_input = torch.from_numpy(edge_input)#.long()
 
        return item



if __name__ == '__main__':
    svg_list = open('/home/xinyangjiang/Datasets/SESYD/FloorPlans/train_list.txt').readlines()
    svg_list = ['/home/xinyangjiang/Datasets/SESYD/FloorPlans/' + line.strip() for line in svg_list]
    builder = SVGGraphBuilderBezier()
    for line in svg_list:
        print(line)
        #line = '/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-01/file_56.svg'
        p = SVGParser(line)
        builder.buildGraph(p.get_all_shape())

    #train_dataset = SESYDFloorPlan(opt.data_dir, pre_transform=T.NormalizeScale())
    #train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    #for batch in train_loader:
    #    pass

    #paths, attributes, svg_attributes = svg2paths2('/home/xinyangjiang/Datasets/SESYD/FloorPlans/floorplans16-05/file_47.svg')
    #print(paths, attributes, svg_attributes)
