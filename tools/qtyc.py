#切图预测
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

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'Disconnector', 'GroundDisconnector', 'CBreaker','Bus','ACLineEnd',
           'Transformer3','Transformer2','Reactor','Capacitor','SVG','Generator')
NETS = {'vgg16': ('vgg16_50000.ckpt',), 'res101': ('res101_faster_rcnn_iter_50000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image   载入demo图像
    im_file = os.path.join(cfg.DATA_DIR, 'result/qifen/', image_name)  #可以换个   和v1不一样
    im = cv2.imread(image_name)

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
            save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\qifen\\result\\'+ image_name
            cv2.imwrite(save_path, im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',		
                        choices=NETS.keys(), default='res101')			#这块是写带训练次数的吗
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',		
                        choices=DATASETS.keys(), default='pascal_voc')			#这块改不
    args = parser.parse_args()

    return args
#图像裁剪
def divide_method2(img,m,n):#分割成m行n列
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h, w = img.shape[0],img.shape[1]
    grid_h=int(h*1.0/(m-1)+0.5)#每个网格的高
    grid_w=int(w*1.0/(n-1)+0.5)#每个网格的宽
    
    #满足整除关系时的高、宽
    h=grid_h*(m-1)
    w=grid_w*(n-1)
    
    #图像缩放
    img_re=cv2.resize(img,(w,h),cv2.INTER_LINEAR)# 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m))
    gx=gx.astype(np.int)
    gy=gy.astype(np.int)

    divide_image = np.zeros([m-1, n-1, grid_h, grid_w,3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,...]=img_re[
            gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]


    return divide_image
    
#显示分块
def display_blocks(divide_image):
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            #divide_image[i,j,:]
            qiefen = divide_image[i,j,:]
            a = str(i*n+j+1)
            save_path = 'C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\qifen\\'+a+'.jpg'
            #demo(sess, net, qiefen)
            # print(type(qiefen))
            # print(qiefen.shape)
            # a = str(i*n+j+1)
            # save_path = 'C:\\Users\\ltj20\\Desktop\\a\\'+a+'.jpg'
            # cv2.imwrite(save_path, qiefen)
        
#复原
def image_concat(divide_image):
    #divide_image = cv2.cvtColor(divide_image,cv2.COLOR_BGR2RGB)
    m,n,grid_h, grid_w=[divide_image.shape[0],divide_image.shape[1],#每行，每列的图像块数
                       divide_image.shape[2],divide_image.shape[3]]#每个图像块的尺寸

    restore_image = np.zeros([m*grid_h, n*grid_w, 3], np.uint8)
    restore_image[0:grid_h,0:]
    for i in range(m):
        for j in range(n):
            restore_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w]=divide_image[i,j,:]
    return restore_image

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

    print('Loaded network {:s}'.format(tfmodel))
    for root, dirs, files in os.walk(r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\demo\\d"):
            im_names = files
    
    for im_name in im_names:
        print(im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        m=3
        n=5
        divide_image1=divide_method2(im_name,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块
        display_blocks(divide_image1)  
        for root, dirs, files in os.walk(r"C:\\Users\\admin\\Desktop\\RCNNv3\\data\\result\\qifen"):
            im_names = files
    
        for im_name in im_names:        
            demo(sess, net, im_name)
        # restore_image2=image_concat(divide_image1)#图像缩放法分块还原
        # save_path = 'C:\\Users\\ltj20\\Desktop\\a\\'+'ww.jpg'
        # cv2.imwrite(save_path, restore_image2)


        
        


