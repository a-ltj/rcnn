#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
pb预测
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from model.bbox_transform import bbox_transform_inv
from model.test import _clip_boxes,_get_blobs
from utils.timer import Timer
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import glob
#import gdal

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from tensorflow.python.framework import graph_util

CLASSES = ('__background__',
           'Disconnector', 'GroundDisconnector', 'CBreaker','Bus','ACLineEnd',
           'Transformer3','Transformer2','Reactor','Capacitor','SVG','Generator')
NETS = {'vgg16': ('vgg16_50000.ckpt',), 'res101': ('res101_faster_rcnn_iter_50000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}

#NETS = {'res101': ('res101_faster_rcnn_iter_70000.ckpt',)}	#修改这
#DATASETS= {'pascal_voc': ('voc_2007_trainval',)}

def vis_detections(im,class_name, dets,outfile, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        #print(bbox)
        score = dets[i, -1]

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),0]=0
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),0] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),0] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),0] = 0

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),1]=0
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),1] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),1] = 0
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),1] = 0

        im[int(bbox[1]),int(bbox[0]):int(bbox[2]),2]=255
        im[int(bbox[3]),int(bbox[0]):int(bbox[2]),2] = 255
        im[int(bbox[1]):int(bbox[3]),int(bbox[0]),2] = 255
        im[int(bbox[1]):int(bbox[3]),int(bbox[2]),2] = 255

# def vis_detections(im, class_name, dets, thresh=0.5):
    # """Draw detected bounding boxes."""
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # if len(inds) == 0:
        # return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # for i in inds:
        # bbox = dets[i, :4]
        # score = dets[i, -1]

        # ax.add_patch(
            # plt.Rectangle((bbox[0], bbox[1]),
                          # bbox[2] - bbox[0],
                          # bbox[3] - bbox[1], fill=False,
                          # edgecolor='red', linewidth=3.5)		#框线的颜色和粗细
            # )
        # ax.text(bbox[0], bbox[1] - 2,
                # '{:s} {:.3f}'.format(class_name, score),
                # bbox=dict(facecolor='blue', alpha=0.5),
                # fontsize=14, color='white')			#字号大小和字体颜色

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
     #                                             thresh),
     #             fontsize=14)
   # plt.axis('off')			
   # plt.tight_layout()			
   # plt.draw()  
def demo(image_name,out_file,sess):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im=cv2.imread(image_name)

    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    print(blobs["im_info"])

    #Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = freeze_graph_test(sess, blobs,im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    #

    # im = im[:, :, (2, 1, 0)]

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im,cls, dets,out_file, thresh=CONF_THRESH)

    cv2.imencode('.jpg',im)[1].tofile(out_file)

# def demo(sess, net, image_name):
    # """Detect object classes in an image using pre-computed object proposals."""

    # # Load the demo image   载入demo图像
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)  #可以换个   和v1不一样
    # im = cv2.imread(im_file)

    # # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    # scores, boxes = im_detect(sess, net, im)
    # timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # # Visualize detections for each class
    # CONF_THRESH = 0.7	#这啥
    # NMS_THRESH = 0.3	#nms阈值吗  
    # thresh = 0.7
# #thresh 又重新赋值了

    # #这块少了一段 打开图片的
    # # 打开图片
    # # im = im[:, :, (2, 1, 0)]
    # # fig, ax = plt.subplots(figsize=(12, 12))
    # # ax.imshow(im, aspect='equal', alpha=1)

    # # 对每一类的每一个目标，在图片上生成框
    # for cls_ind, cls in enumerate(CLASSES[1:]):
        # cls_ind += 1 # because we skipped background
        # cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        # cls_scores = scores[:, cls_ind]
        # dets = np.hstack((cls_boxes,
                          # cls_scores[:, np.newaxis])).astype(np.float32)
        # keep = nms(dets, NMS_THRESH)
        # dets = dets[keep, :]
        # #vis_detections(im, cls, dets, thresh=CONF_THRESH)     #v1注释掉了  这块先别注释掉 看看是啥函数

       # #少了一大段
        # inds = np.where(dets[:, -1] >= thresh)[0]
        # if len(inds) == 0:
            # continue
        # for i in inds:
            # bbox = dets[i, :4]
            # score = dets[i, -1]
            # c1 = (int(bbox[0]), int(bbox[1]))
            # c2 = (int(bbox[2]), int(bbox[3]))
            # bbox_mess = '%s: %.3f' % (cls, score)
            # cv2.rectangle(im, c1, c2, (0,255,0), 1)
            # t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1 )[0]
            # cv2.rectangle(im, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (0,255,0), -1) 
            # cv2.putText(im, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            # save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\'+ image_name
            # cv2.imwrite(save_path, im)

            # ax.add_patch(
                # plt.Rectangle((bbox[0], bbox[1]),
                              # bbox[2] - bbox[0],
                              # bbox[3] - bbox[1], fill = False,
                              # edgecolor = 'red', linewidth = 1.5))
            # ax.text(bbox[0], bbox[1] - 2,
                    # '{:s} {:.3f}'.format(cls, score),
                    # bbox = dict(facecolor='blue', alpha=0.5),
                    # fontsize = 14, color = 'white')
            # print (bbox)
#plt.axis('off')
#        plt.tight_layout()
#        plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',		
                        choices=NETS.keys(), default='res101')			#这块是写带训练次数的吗
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',		
                        choices=DATASETS.keys(), default='pascal_voc')			#这块改不
    args = parser.parse_args()

    return args

def freeze_graph_test(sess, blobs,im):
    '''
	:param pb_path:pb文件的路径
	:param image_path:测试图片的路径
	:return:
	'''
    # 定义输入的张量名称,对应网络结构的输入张量
    # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
    # 定义输出的张量名称
    input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
    tensor_info = sess.graph.get_tensor_by_name("Placeholder_1:0")

    biasadd = sess.graph.get_tensor_by_name("resnet_v1_101_5/cls_score/BiasAdd:0")
    score = sess.graph.get_tensor_by_name("resnet_v1_101_5/cls_prob:0")
    bbox = sess.graph.get_tensor_by_name("add:0")
    rois = sess.graph.get_tensor_by_name("resnet_v1_101_3/rois/concat:0")

    _, scores, bbox_pred, rois = sess.run([biasadd, score, bbox, rois],
                                          feed_dict={input_image_tensor: blobs['data'], tensor_info: blobs['im_info']})

    im_scales=blobs['im_info'][2]
    boxes = rois[:, 1:5] / im_scales
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

if __name__ == '__main__':
    # cfg.TEST.HAS_RPN = True  			# Use RPN for proposals
    # args = parse_args()

    # # model path
    # demonet = args.demo_net
    # dataset = args.dataset			#下边路径不完整！！！！！和v1不一样
    # tfmodel = os.path.join(r'C:\Users\admin\Desktop\RCNNv3\output', demonet, DATASETS[dataset][0], 'default2', NETS[demonet][0])	


    # if not os.path.isfile(tfmodel + '.meta'):
        # raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       # 'our server and place them properly?').format(tfmodel + '.meta'))

    # # set config
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth=True

    # # init session
    # sess = tf.Session(config=tfconfig)
    # # load network
    # if demonet == 'vgg16':
        # net = vgg16()
    # elif demonet == 'res101':
        # net = resnetv1(num_layers=101)
    # else:
        # raise NotImplementedError


    # net.create_architecture("TEST", 12,
                          # tag='default', anchor_scales=[8, 16, 32]) 	#少了个sess
    # saver = tf.train.Saver()
    # saver.restore(sess, tfmodel)

    # print('Loaded network {:s}'.format(tfmodel))


    # for root, dirs, files in os.walk(r"C:\Users\admin\Desktop\RCNNv3\data\demo"):
            # im_names = files
    
    # for im_name in im_names:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(im_name))
        # demo(sess, net, im_name)
    input_dir = r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\demo\\d"
    output_dir = r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\"
    pb_path = r"C:\Users\admin\Desktop\RCNNv3\pb\tf_faster_rcnn_resnet_demo.pb"
    im_names = glob.glob(os.path.join(input_dir,"*.jpg"))
    print(im_names)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for im_name in im_names:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                img_basename = os.path.basename(im_name)
                out_file = os.path.join(output_dir,img_basename)
                print(im_name)
                demo(im_name,out_file,sess)