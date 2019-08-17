import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import pdb
import matplotlib.pylab as plt

# Get MNIST datasets
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# --------------------------- parameters ------------------------------------ #
# batch size
BATCH_SIZE = 200
# number of training
NUM_STEPS = 50000
lr = 1e-3
# encoder & decoder layer
HIDDEN1 = 32 
HIDDEN2 = 32
HIDDEN3 = 32
# output channel
DIM_CH = 1
# --------------------------------------------------------------------------- #

# --------------------- placeholder ----------------------------------------- #
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
# --------------------------------------------------------------------------- #    
def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
# --------------------------------------------------------------------------- #    
"""
For 1d convolution
def conv1d_relu(inputs, w, b, stride):
    conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
# --------------------------------------------------------------------------- #
def conv1d_t_relu(inputs, w, b, output_shape, stride):
    conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
"""
# --------------------------------------------------------------------------- #
def max_pool(inputs):
    return tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
# --------------------------------------------------------------------------- #
def conv2d_relu(inputs, w, b, stride):
    conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
# --------------------------------------------------------------------------- #
def conv2d_t_relu(inputs, w, b ,output_shape,stride):
    conv = tf.nn.conv2d_transpose(inputs, w, strides=stride,output_shape=output_shape ,padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
# --------------------------------------------------------------------------- #    
def fc_relu(inputs, w, b):
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.relu(fc)
    return fc
# --------------------------------------------------------------------------- #
def Encoder(x, reuse=False):
    """
    [argument]
    x: images [None,28,28,1]
    """
    with tf.variable_scope('Encoder') as scope:
        if reuse:
            scope.reuse_variables()
        
        # 1 layer
        convW1 = weight_variable("convW1", [5, 5, DIM_CH, HIDDEN1])
        convB1 = bias_variable("convB1", [HIDDEN1])
        conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1]) #[none,14,14,32]
       
        # 2 layer
        convW2 = weight_variable("convW2", [5, 5, HIDDEN1, HIDDEN2])
        convB2 = bias_variable("convB2", [HIDDEN2])
        conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1]) #[none,7,7,32]
                
        # 3 layer
        convW3 = weight_variable("convW3", [5, 5, HIDDEN2, HIDDEN3])
        convB3 = bias_variable("convB3", [HIDDEN3])
        conv3 = conv2d_relu(conv2, convW3, convB3, stride=[1,2,2,1]) #[none,4,4,32]

        # convert to vector
        conv3_vec = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
        
        # fc1
        fcW1 = weight_variable("fcW1", [4*4*HIDDEN3, HIDDEN2])
        fcB1 = bias_variable("fcB1", [HIDDEN2])
        fc1 = fc_relu(conv3_vec, fcW1, fcB1) #[none,32]
       
        # fc2
        fcW2 = weight_variable("fcW2", [HIDDEN2, HIDDEN1])
        fcB2 = bias_variable("fcB2", [HIDDEN1])
        fc2 = fc_relu(fc1, fcW2, fcB2) #[none,32]
        
        return fc2
# --------------------------------------------------------------------------- #
def Decoder(z, reuse=False):
    """
    [argument]
    z: encoder output [none,32]
    """
    
    with tf.variable_scope('decoder') as scope:
        if reuse:
            scope.reuse_variables()
        #pdb.set_trace()
        # fc1
        fcW11 = weight_variable("fcW1", [HIDDEN3, HIDDEN2]) 
        fcB11 = bias_variable("fcB1", [HIDDEN2])
        fc11 = fc_relu(z, fcW11, fcB11) #[none,32]
                
        # fc2
        fcW2 = weight_variable("fcW2", [HIDDEN2, 4*4*HIDDEN1])
        fcB2 = bias_variable("fcB2", [4*4*HIDDEN1])    
        fc2 = fc_relu(fc11, fcW2, fcB2) #[none,512]
        
        fc2 = tf.reshape(fc2, tf.stack([BATCH_SIZE, 4, 4, HIDDEN1])) #[200,4,4,32]
        
        # deconv1
        convW1 = weight_variable("convW1", [5, 5, HIDDEN2, HIDDEN3])
        convB1 = bias_variable("convB1", [HIDDEN2])
        conv1 = conv2d_t_relu(fc2, convW1,convB1,output_shape=[BATCH_SIZE,7,7,HIDDEN2] ,stride=[1,2,2,1]) #[none,7,7,32]
        
        # deconv2
        convW2 = weight_variable("convW2", [5, 5, HIDDEN1, HIDDEN2])
        convB2 = bias_variable("convB2", [HIDDEN1])
        conv2 = conv2d_t_relu(conv1, convW2, convB2,output_shape=[BATCH_SIZE,14,14,HIDDEN1], stride=[1,2,2,1]) #[none,14,14,32]
        
        # deconv3
        convW3 = weight_variable("convW3", [5, 5, DIM_CH, HIDDEN1])
        convB3 = bias_variable("convB3", [DIM_CH])
        output = conv2d_t_relu(conv2, convW3, convB3,output_shape=[BATCH_SIZE,28,28,DIM_CH] ,stride=[1,2,2,1]) #[none,28,28,1]
        
        return output
# --------------------------------------------------------------------------- #
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

# --------------------------------------------------------------------------- #
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
# --------------------------------------------------------------------------- #

# ============================== train ====================================== #
encoder_op = Encoder(x_image) #IN:images[None,28,28,1]
decoder_op = Decoder(encoder_op) #[none,28,28,1]
decoder_img = tf.reshape(decoder_op,tf.stack([-1,784])) #[5600,28]
# ============================== test ======================================= #
encoder_op_test = Encoder(x_image,reuse=True)
decoder_op_test = Decoder(encoder_op_test,reuse=True)
decoder_img_test = tf.reshape(decoder_op_test,tf.stack([-1,784])) #[5600,28]
# --------------------------------------------------------------------------- #
# loss
trloss = tf.reduce_mean(tf.square(decoder_img - x))
teloss = tf.reduce_mean(tf.square(decoder_img_test - x))
# optimizer
trainer = tf.train.AdamOptimizer(lr).minimize(trloss)
# --------------------------------------------------------------------------- #
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# start training
for i in range(NUM_STEPS):
    
    # Get train images
    batchX,batchY = mnist.train.next_batch(BATCH_SIZE)
    trainImg = np.reshape(batchX,[BATCH_SIZE,784])
    
    _, trainLoss = sess.run([trainer, trloss], feed_dict={x:trainImg})
    
    if i % 500 == 0:    
        # Get test images
        testX,testY = mnist.test.next_batch(BATCH_SIZE)
        testImg = np.reshape(testX,[BATCH_SIZE,784])
        
        testLoss = sess.run(teloss, feed_dict={x:testImg})
            
        print("Itr: %d, trianLoss: %f, testLoss: %f " % (i,trainLoss,testLoss))