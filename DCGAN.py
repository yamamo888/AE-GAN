# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 18:08:33 2019

@author: yu
"""

import os

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

import pdb

# Load mnist dataset
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# -------------- path ------------------- #
# saved images
images = "images"
# ---------------------------------------- #

# ----------- paramters ----------------- #
NUM_EPOCKS = 200
NUM_STEPS = 100
BATCH_SIZE = 10
lr = 1e-4
TRAIN_PERIOD = 500

DIM_IMG = 784
HIDDEN0 = 256
HIDDEN1 = 128
HIDDEN2 = 64
DIM_CH = 1
DIM_NOISE = 200

# ---------------------------------------- #

# ----------- placeholder ---------------- #
# input noise of Generator
Z = tf.placeholder(tf.float32, shape=[None,DIM_NOISE])
# images
img_real = tf.placeholder(tf.float32,shape=[None,28,28,1])
# target for Discriminator
d_t = tf.placeholder(tf.float32,shape=[None,1])
# target for Generator
g_t = tf.placeholder(tf.float32,shape=[None,1])
# train or not-train for Discriminator
#d_train = tf.placeholder(tf.bool)
# train or not-train for Generator
#g_train = tf.placeholder(tf.bool)
# train of not-train
is_train = tf.placeholder(tf.bool)
# ---------------------------------------- #


# --------------------------------------------------------------------------- #
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
# --------------------------------------------------------------------------- #    
def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
# --------------------------------------------------------------------------- #
def scale_variable(name,shape):
    return tf.get_variable(name, shape, initializer=tf.ones_initializer())
# --------------------------------------------------------------------------- #
def offset_variable(name,shape):
    return tf.get_variable(name, shape, initializer=tf.zeros_initializer())
# --------------------------------------------------------------------------- #
def conv2d(inputs, w, b, stride):
    return tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b
# --------------------------------------------------------------------------- #
def conv2d_t(inputs, w, b ,stride, output_shape):
    return tf.nn.conv2d_transpose(inputs, w, strides=stride,output_shape=output_shape,padding='SAME') + b
# --------------------------------------------------------------------------- #
def fc(inputs, w, b, keepProb):
    fc = tf.matmul(inputs,w) + b
    fc = tf.nn.dropout(fc, keepProb)
    return fc 
# --------------------------------------------------------------------------- #
def fc_sigmoid(inputs, w, b, keepProb):
    sigmoid = tf.matmul(inputs,w) + b
    sigmoid = tf.nn.dropout(sigmoid, keepProb)
    sigmoid = tf.nn.sigmoid(sigmoid)
    return sigmoid
# --------------------------------------------------------------------------- #
def fc_relu(inputs,w,b,keepProb):
    relu = tf.matmul(inputs,w) + b
    relu = tf.nn.dropout(relu, keepProb)
    relu = tf.nn.relu(relu)
    return relu
# --------------------------------------------------------------------------- #
def leaky_relu(inputs):
    leaky_relu = tf.nn.leaky_relu(inputs,alpha=0.2)
    return leaky_relu
# --------------------------------------------------------------------------- #
def tanh(inputs):
    tanh = tf.nn.tanh(inputs)
    return tanh
# --------------------------------------------------------------------------- #
def tensor2vector(inputs):
    shape = inputs.get_shape()[1:].as_list()
    dim = np.prod(shape)
    return tf.reshape(inputs,[-1,dim])
# --------------------------------------------------------------------------- #
def batch_normalization(inputs,scale,offset,axes,is_train=False):
    # 1.return unchanced value, isTrain == True
    if not is_train:
        return inputs
    # 2. return batch normalization
    else:
        eps = 1e-5
        mean, variance = tf.nn.moments(inputs,axes=[0])
        #bn = tf.nn.batch_normalization(inputs,mean,variance,None,None,eps)
        bn = tf.nn.batch_normalization(inputs,mean,variance,offset,scale,eps)
        return bn
# --------------------------------------------------------------------------- #
def Generator(z,reuse=False):
    """
    Generator Networks. Generate data like train data.
    IN: noise, OUT: image
    Artictures: 4 layers
    Activation: leaky relu -> leaky relu -> leaky relu -> tanh
    [argument]
    x: noise
    """
    with tf.variable_scope("Generator") as scope:
        keepProb = 1.0
        if reuse:
           keepProb = 1.0
           scope.reuse_variables()
        
        w0 = weight_variable("g_w0",[DIM_NOISE,4*4*HIDDEN0])
        b0 = bias_variable("g_b0",[4*4*HIDDEN0])
        fc0 = fc_relu(z,w0,b0,keepProb)
        h0 = tf.reshape(fc0,[-1,4,4,HIDDEN0])
        
        # 1 layer
        w1 = weight_variable("g_w1",[4,4,HIDDEN1,HIDDEN0])
        b1 = bias_variable("g_b1",[HIDDEN1])
        g_convt1 = conv2d_t(h0,w1,b1,stride=[1,2,2,1],output_shape=[BATCH_SIZE,7,7,HIDDEN1])
        s1 = scale_variable("g_s1",[HIDDEN1])
        o1 = offset_variable("g_o1",[HIDDEN1])
        bn1 = batch_normalization(g_convt1,s1,o1,[0,1,2])
        h1 = leaky_relu(bn1)
       
        # 2 layer
        w2 = weight_variable("g_w2",[4,4,HIDDEN2,HIDDEN1])
        b2 = bias_variable("g_b2",[HIDDEN2])
        g_convt2 = conv2d_t(h1,w2,b2,stride=[1,2,2,1],output_shape=[BATCH_SIZE,14,14,HIDDEN2])
        s2 = scale_variable("g_s2",[HIDDEN2])
        o2 = offset_variable("g_o2",[HIDDEN2])
        bn2 = batch_normalization(g_convt2,s2,o2,[0,1,2])
        h2 = leaky_relu(bn2)
       
        # 3 layer
        w3 = weight_variable("g_w3",[4,4,DIM_CH,HIDDEN2])
        b3 = bias_variable("g_b3",[DIM_CH])
        conv_t3 = conv2d_t(h2,w3,b3,stride=[1,2,2,1],output_shape=[BATCH_SIZE,28,28,DIM_CH])
        y = tanh(conv_t3)
       
        return y
# --------------------------------------------------------------------------- #    
def Discriminator(x,reuse=False):
    """
    Activation: leaky relu -> leaky relu -> leaky relu -> leaky relu 
    """
    with tf.variable_scope("Discriminator") as scope:
        keepProb = 1.0
        if reuse:
            keepProb = 1.0
            scope.reuse_variables()
        
        # 1 layer
        w1 = weight_variable("d_w1",[4,4,DIM_CH,HIDDEN2])
        b1 = bias_variable("d_b1",[HIDDEN2])
        d_conv1 = conv2d(x,w1,b1,stride=[1,2,2,1])
        h1 = leaky_relu(d_conv1)
        
        # 2 layer
        w2 = weight_variable("d_w2",[4,4,HIDDEN2,HIDDEN1])
        b2 = bias_variable("d_b2",[HIDDEN1])
        d_conv2 = conv2d(h1,w2,b2,stride=[1,2,2,1])
        h2 = leaky_relu(d_conv2)
        
        # 3 layer
        w3 = weight_variable("d_w3",[4,4,HIDDEN1,HIDDEN0])
        b3 = bias_variable("d_b3",[HIDDEN0])
        d_conv3 = conv2d(h2,w3,b3,stride=[1,2,2,1])
        h3 = leaky_relu(d_conv3)
        
        h3_vec = tensor2vector(h3)
        
        # 4 layer
        w4 = weight_variable("g_w4",[4*4*HIDDEN0,DIM_CH])
        b4 = bias_variable("g_b4",[DIM_CH])        
        y = fc_sigmoid(h3_vec,w4,b4,keepProb)
        
        return y
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    # Build Generator Network 
    g_sample = Generator(Z)

    # Build Discriminator Networks
    d_real = Discriminator(img_real) # from noise input
    d_fake = Discriminator(g_sample,reuse=True) # from generated samples
    d_concat = tf.concat([d_real,d_fake],0)
    
    # Build 
    g_stack = Discriminator(g_sample,reuse=True)
    
    # Loss for Discriminator
    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_concat,labels=d_t))
    # Loss for Generator
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_stack,labels=g_t))
    
    # Generator Network variables
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    # Discriminator Network variables
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")
    
    # Optimizer for Generator
    g_trainer = tf.train.AdamOptimizer(lr).minimize(g_loss,var_list=g_vars)
    # Optimizer for Discriminator
    d_trainer = tf.train.AdamOptimizer(lr).minimize(d_loss,var_list=d_vars)
    
    # ======================================================================= #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # ======================================================================= #
    # start training
    for epock in range(NUM_EPOCKS):
        for itr in range(NUM_STEPS):
            
            # noise for Generator
            z = np.random.uniform(-1.,1.,size=[BATCH_SIZE,DIM_NOISE])
            
            # Get the next batch of MNIST data (only images are needed(=batchX))
            batchX,batchY = mnist.train.next_batch(BATCH_SIZE)
            batchX = np.reshape(batchX,[-1,28,28,1])
            
            # real image: 1, fake image: 0 
            # half of real images, the ohers of fake images 
            batchYLabelD = np.concatenate([np.ones([BATCH_SIZE]),np.zeros([BATCH_SIZE])],0)[:,np.newaxis]
            #
            batchYLabelG = np.ones([BATCH_SIZE])[:,np.newaxis]
            
            # Train Discriminator & Generator
            _, _, trainDLoss, trainGLoss = sess.run([g_trainer,d_trainer,g_loss,d_loss],feed_dict={img_real:batchX,Z:z,d_t:batchYLabelD,g_t:batchYLabelG,is_train:True})
            
            if itr % TRAIN_PERIOD == 0:
                print("trainDLoss: %f, trainGLoss: %f" % (trainDLoss,trainGLoss))
        # =================================================================== #
        f, a = plt.subplots(5, 10, figsize=(10, 5))
        for i in range(10):
            # noise for Generator
            # BAG?: only size=[BATCH_SIZE,DIM_NOISE], so I must change output_shape in conv2d_transpose()
            z = np.random.uniform(-1.,1.,size=[BATCH_SIZE,DIM_NOISE])
            
            g_img = sess.run(g_sample,feed_dict={Z:z})
            
            for j in range(5):
                # Generate image from noise. Extend to 3 channels for matplot figure.
                img = np.reshape(np.repeat(g_img[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
                
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")
                a[j][i].imshow(img)
                
        plt.savefig(os.path.join(images,"{}_{}.png".format(epock,i)))  
        plt.close()
            