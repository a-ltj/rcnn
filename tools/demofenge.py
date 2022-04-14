#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from tensorflow.python.framework import graph_util
from PIL import Image
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'Disconnector', 'GroundDisconnector', 'CBreaker','Bus','ACLineEnd',
           'Transformer3','Transformer2','Reactor','Capacitor','SVG','Generator')
NETS = {'vgg16': ('vgg16_50000.ckpt',), 'res101': ('res101_faster_rcnn_iter_50000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}

#NETS = {'res101': ('res101_faster_rcnn_iter_70000.ckpt',)}	#修改这
#DATASETS= {'pascal_voc': ('voc_2007_trainval',)}


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image   载入demo图像
    im_file = os.path.join(cfg.DATA_DIR, 'demo/f', image_name)
    img = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    mulpicplus = "3"  # 1 for normal,2 for 4pic plus,3 for 9pic plus and so on
    assert (int(mulpicplus) >= 1)
    if mulpicplus == "1":

        scores, boxes = im_detect(sess, net, img)


    else:
        xsz = img.shape[0]
        ysz = img.shape[1]
        mulpicplus = int(mulpicplus)
        x_smalloccur = int(xsz / mulpicplus * 1.2)
        y_smalloccur = int(ysz / mulpicplus * 1.2)
        for i in range(mulpicplus):
            x_startpoint = int(i * (xsz / mulpicplus))
            for j in range(mulpicplus):
                y_startpoint = int(j * (ysz / mulpicplus))
                x_real = min(x_startpoint + x_smalloccur, xsz)
                y_real = min(y_startpoint + y_smalloccur, ysz)
                if (x_real - x_startpoint) % 64 != 0:
                    x_real = x_real - (x_real - x_startpoint) % 64
                if (y_real - y_startpoint) % 64 != 0:
                    y_real = y_real - (y_real - y_startpoint) % 64
                dicsrc = img[x_startpoint:x_real,y_startpoint:y_real]
                print(dicsrc.shape)
                scores_temp0, boxes_temp0 = im_detect(sess, net, dicsrc)
                print(boxes_temp0.shape)
                print(boxes_temp0)
                boxes_temp = np.split(boxes_temp0,12,axis = 1)
                for k in boxes_temp:   
                    k[...,0] = k[...,0] + y_startpoint
                    k[...,1] = k[...,1] + x_startpoint
                    k[...,2] = k[...,2] + y_startpoint
                    k[...,3] = k[...,3] + x_startpoint
                boxes_temp = np.hstack((boxes_temp[0],boxes_temp[1],boxes_temp[2],boxes_temp[3],
                                        boxes_temp[4],boxes_temp[5],boxes_temp[6],boxes_temp[7],
                                        boxes_temp[8],boxes_temp[9],boxes_temp[10],boxes_temp[11]))
                print(boxes_temp.shape)
                print(boxes_temp)
                if i == 0 and j == 0:
                    scores = scores_temp0
                    boxes = boxes_temp

                else:
                    boxes= np.concatenate((boxes, boxes_temp),axis=0)
                    scores= np.concatenate((scores, scores_temp0),axis=0)

                    

    timer.toc()
    CONF_THRESH = 0.7	#这啥
    NMS_THRESH = 0.3	#nms阈值吗  
    thresh = 0.7


    # 对每一类的每一个目标，在图片上生成框
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:,np.newaxis]))
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            c1 = (int(bbox[0]), int(bbox[1]))
            c2 = (int(bbox[2]), int(bbox[3]))
            bbox_mess = '%s: %.3f' % (cls, score)
            cv2.rectangle(img, c1, c2, (0,255,0), 1)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1 )[0]
            cv2.rectangle(img, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (0,255,0), -1)
            cv2.putText(img, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\f\\'+ image_name
            cv2.imwrite(save_path, img)
            
            f.writelines(im_name+'\t'+cls+'\t'+str(bbox[0])+'\t'+str(bbox[1])+'\t'+str(bbox[2])+'\t'+str(bbox[3]))
            f.write('\n')



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',		
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',		
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  			# Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join(r'C:\\Users\\admin\\Desktop\\RCNNv3\\output', demonet, DATASETS[dataset][0], 'default2', NETS[demonet][0])
    print(tfmodel)	


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError


    net.create_architecture("TEST", 12,
                          tag='default', anchor_scales=[8, 16, 32]) 	#少了个sess
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    for root, dirs, files in os.walk(r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\demo\\f"):
            im_names = files
    f=open('result.txt','w',encoding='utf-8')
    for im_name in im_names:
        print(im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        
        demo(sess, net, im_name)      
    f.close()



