import os
import sys
import math
import random

import numpy as np
import tensorflow as tf

from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

from PIL import Image

slim = tf.contrib.slim

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

net_shape = (512, 512)
data_format = 'NHWC'
img_path = 'demo/'
classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
ssd_anchors = ssd_net.anchors(net_shape)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    
# Main image processing routine.
def process_image(img, sess, select_threshold=0.5, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})
                
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors, select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
                        
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def main(_):
    ckpt_path = '/data/ssd_models/vgg_512/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path) 
        
        img_names = sorted(os.listdir(img_path))
        #img = mpimg.imread(path + image_names[-5])
        img = Image.open(img_path + img_names[-5])
        rclasses, rscores, rbboxes = process_image(img, sess)
    
        found_classes = []
        for c in rclasses:
           found_classes.append(classes[c-1])
        
        print("class_indexes:{}\nclasses: {}\nscores:{}\nbbxs: {}\n".format(rclasses, found_classes, rscores, rbboxes))

if __name__ == '__main__':
    tf.app.run()
