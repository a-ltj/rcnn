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

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'Disconnector', 'GroundDisconnector', 'CBreaker','Bus','ACLineEnd',
           'Transformer3','Transformer2','Reactor','Capacitor','SVG','Generator')
NETS = {'vgg16': ('vgg16_50000.ckpt',), 'res101': ('res101_faster_rcnn_iter_50000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}

#NETS = {'res101': ('res101_faster_rcnn_iter_70000.ckpt',)}	#修改这
#DATASETS= {'pascal_voc': ('voc_2007_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)		#框线的颜色和粗细
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')			#字号大小和字体颜色

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
     #                                             thresh),
     #             fontsize=14)
   # plt.axis('off')			
   # plt.tight_layout()			
   # plt.draw()  

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image   载入demo图像
    #f=open('result.txt','w',encoding='utf-8')
    im_file = os.path.join(cfg.DATA_DIR, 'demo/d', image_name)  #可以换个   和v1不一样
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.7	#这啥
    NMS_THRESH = 0.3	#nms阈值吗  
    thresh = 0.7
#thresh 又重新赋值了

    #这块少了一段 打开图片的
    # 打开图片
    #im = im[:, :, (2, 1, 0)]
    
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal', alpha=1)

    # 对每一类的每一个目标，在图片上生成框
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)     #v1注释掉了  这块先别注释掉 看看是啥函数

       #少了一大段
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            #print(bbox)
            c1 = (int(bbox[0]), int(bbox[1]))
            c2 = (int(bbox[2]), int(bbox[3]))
            bbox_mess = '%s: %.3f' % (cls, score)
            cv2.rectangle(im, c1, c2, (0,255,0), 1)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1 )[0]
            cv2.rectangle(im, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (0,255,0), -1) 
            cv2.putText(im, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\f\\'+ image_name
            cv2.imwrite(save_path, im)
            
            f.writelines(im_name+'\t'+cls+'\t'+str(bbox[0])+'\t'+str(bbox[1])+'\t'+str(bbox[2])+'\t'+str(bbox[3]))
            f.write('\n')
        
            # ax.add_patch(
                # plt.Rectangle((bbox[0], bbox[1]),
                              # bbox[2] - bbox[0],
                              # bbox[3] - bbox[1], fill = False,
                              # edgecolor = 'red', linewidth = 1))
            # ax.text(bbox[0], bbox[1] - 2,
                    # '{:s} {:.3f}'.format(cls, score),
                    # bbox = dict(facecolor='blue', alpha=0.4),
                    # fontsize = 10, color = 'white')
            #print (bbox)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',		
                        choices=NETS.keys(), default='res101')			#这块是写带训练次数的吗
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',		
                        choices=DATASETS.keys(), default='pascal_voc')			#这块改不
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  			# Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset			#下边路径不完整！！！！！和v1不一样
    tfmodel = os.path.join(r'C:\\Users\\admin\\Desktop\\RCNNv3\\output', demonet, DATASETS[dataset][0], 'default2', NETS[demonet][0])	


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
    #ckpt to pb
    # graph = tf.get_default_graph()
    # input_graph_def = graph.as_graph_def()
    # output_graph = r"C:\Users\admin\Desktop\RCNNv3\pb\tf_faster_rcnn_resnet_demo.pb"
    # output_node_names = "resnet_v1_101_5/cls_prob,add,resnet_v1_101_3/rois/concat,resnet_v1_101_5/cls_score/BiasAdd"
    # output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
    # with tf.gfile.GFile(output_graph, "wb") as f:
        # f.write(output_graph_def.SerializeToString())

    print('Loaded network {:s}'.format(tfmodel))

    for root, dirs, files in os.walk(r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\demo\\d"):
            im_names = files
    f=open('result.txt','w',encoding='utf-8')
    for im_name in im_names:
        print(im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        
        demo(sess, net, im_name)      
    f.close()
        #save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\'+ im_name
        #cv2.imwrite(save_path, im)
        #plt.savefig('C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\'+ im_name )
        #plt.clf()


#    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
#                '001763.jpg', '004545.jpg']
#    for im_name in im_names:
#        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print('Demo for data/demo/{}'.format(im_name))
#        demo(sess, net, im_name)

#    plt.show()
