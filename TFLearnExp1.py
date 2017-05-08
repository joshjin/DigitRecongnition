''''

This program is a CNN that identifies an integer from 0 to 9.
By the given example, this model gets an accuracy of 0.8154, but depending on the test, value may vary
I acknowledge that many segments of codes in this program come from online tutorials and github.

@Author: Zexi Josh Jin
'''''

# training set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import argparse
import sys
import tensorflow as tf
from glob import glob
import os
import ntpath
from PIL import Image
from os import listdir
import numpy
import numpy as np
import skimage.io
import skimage.util
import skimage.transform
from skimage import data, img_as_float
from skimage import exposure
from sys import argv
#from skimage import io

# setting up CNN, by tutorial of Tensorflow online

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
results = tf.argmax(y_conv, 1)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

train_flag = 1

# testing print statement
''''
print("Image detected, name: " + f)
print(arr.shape)
print(resized_arr.shape)
print('processing file: ' + f)
print(files)
print(anno_list)
'''''

# process the testinf images to get a maximum result
def parse_image():
    path = argv[1]
    output = open("prediction.txt", "w")
    files = [f for f in listdir(path)]
    name_list = []
    for f in files:
        if f.endswith(".png") and (not f.startswith("copy")):
            im = make_square(skimage.io.imread(path+"/"+f, as_grey=True))
            # comment section that included some code didn't use in the final version of PA4
            ''''
            #data = np.asarray(im, dtype="int32")
                        for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] != 1:
                        data[i][j] = 0
            print(data)
            #print(data)
            '''''
            logarithmic_corrected = exposure.adjust_gamma(im, 0.15)
            f_new = "copy"+f
            resized_im = skimage.transform.resize(logarithmic_corrected, (28, 28))
            skimage.io.imsave(f_new, resized_im)
            sps = f.split("/")
            image_name = sps[-1]
            name_list.append(image_name)
            input_imgs = numpy.reshape(resized_im, (1, 784))
            feed_dict = {x: input_imgs, keep_prob: 1.0}
            # Eval the final result
            p = results.eval(feed_dict=feed_dict)
            output.write("%s\t%s\n" % (ntpath.basename(image_name),p[0]))
    output.close()
    # calculate the final result, file manipulation etc
    test = {}
    total = 0
    right = 0
    anno = open("annotation.txt","r")
    anno_list = []
    for line in anno:
        image,label=line.split()
        test[image]=label
        anno_list.append(line)
    opt = open("prediction.txt", "r")
    for line in opt:
        image,label=line.split()
        total += 1
        if label==test[image]:
            right += 1
    result=right/total
    anno.close()
    opt.close()
    # result
    print("%.4f" % result)

# scrap, tests not going to be used
''''
import Image
import numpy as np
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data
def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
    print(image_list)
    print(name_list)
'''''

# convert a image to a array and try to obtain the maximum result through manipulating the array
def make_square(img_arr):
    tmp = img_arr.max()
    (vertical_pixel, horizontal_pixel) = img_arr.shape
    for i in range(len(img_arr)):
        for j in range(len(img_arr[i])):
            # amplifiying the signal strength by 2
            if img_arr[i][j] * 2 > tmp:
                img_arr[i][j] = tmp
            else:
                img_arr[i][j] *= 2
            # to make the image bolder, not working yet, might be useful in the future
            ''''
            #if i != 0 and i != len(img_arr) - 1 and j != 0 and j != len(img_arr[i]) - 1:
                #if img_arr[i][j] + img_arr[i - 1][j] + img_arr[i + 1][j] + img_arr[i][j - 1] + img_arr[i][j + 1] > tmp:
                #    img_arr[i][j] = tmp
                #else:
                #    img_arr[i][j] = img_arr[i][j] + img_arr[i - 1][j] + img_arr[i + 1][j] + img_arr[i][j - 1] + img_arr[i][j + 1]
            #if img_arr[i][j] > tmp:
            #    img_arr[i][j] = tmp
            '''''
            # not used, absolution
            ''''
                    #if img_arr[i][j] <= 0.8:
                    #    img_arr[i][j] = 0
                    #if img_arr[i][j] > 0.8:
                    #    img_arr[i][j] = 1
                    #if
            #print(img_arr)
            '''''
            # padding the image
            # padding 1.1 gets 0.71 on model 1
            # padding 1.2 gets 0.76 on model 1
            # padding 1.3 gets 0.81 on model 1
    if vertical_pixel > horizontal_pixel:
        vertical_padding = int(round(vertical_pixel*0.15))
        horizontal_padding = int(round((vertical_pixel*1.3 - horizontal_pixel) / 2))
    else:
        horizontal_padding = int(round(horizontal_pixel*0.15))
        vertical_padding = int(round((horizontal_pixel*1.3 - vertical_pixel) / 2))
    padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
    return skimage.util.pad(img_arr, padding, 'constant', constant_values=0)

# code for regression from TF official web
# not used
''''
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''''

# the main: train_flag == 1 means training the model, train_flag != 1 means restoring a model and test on testing set
if train_flag == 1:
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(200):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        save_path = saver.save(sess, "model2.ckpt")
        print("Model saved in file: %s" % save_path)
else:
    with tf.Session() as sess:
        saver.restore(sess, "./model1.ckpt")
        print("Model restored.")
        parse_image()
        #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


