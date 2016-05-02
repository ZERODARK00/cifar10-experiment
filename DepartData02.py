#coding=utf-8
import tensorflow as tf
import numpy as np
import random as rd

#创建权值和偏置
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

#创建池化和pooling
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# batch 数据的大小
practise_num = 128

# 解压数据集
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
#将list的数字转换为narray对应的向量
def OneHot_Encoding(x):
	labels = np.zeros((len(x),10))
	for i in range(len(x)):
		labels[i][x[i]]=1
	return labels
#将二维narray数组的图片数据进行预处理
def normalization2(x):
	mean = np.mean(x,axis=0)
	var  = np.var(x,axis=0)
	return (x-mean)/var
#将三维narray数组的图片数据进行预处理
def normalization3(x):
	mean = np.mean(np.mean(x,axis=0),axis=0)
	var = np.var(np.var(x,axis=0),axis=0)
	return (x-mean)/var
#测试数据集的预处理
test = unpickle("test_batch")
row = test['data'].shape[0]
column = test['data'].shape[1]
test_datas = normalization2(test['data'])
test_labels = OneHot_Encoding(test['labels'])

#训练数据集预处理
data = [];label=[]
for i in range(5):
	file = "data_batch_%d" % (i+1)
	temp = unpickle(file)
	data.append(temp['data']))
	label.append(OneHot_Encoding(temp['labels']))
train_datas = normalization3(np.array(data));train_labels = np.array(label)
#搭建预测模型softmax
x = tf.placeholder("float",[None,3072])
y_ = tf.placeholder("float",[None,10])
W = tf.Variable(tf.zeros([row,10]))
b = tf.Variable(tf.zeros([10]))

#第一层卷积
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,32,32,3])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#密集连接层
W_fc1 = weight_variable([16*16*32,512])
b_fc1 = bias_variable([512])
h_pool1_flat = tf.reshape(h_pool1,[-1,16*16*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([512,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))) 

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

#初始化所有变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#开始训练，每一次batch数据量进行训练，训练10000次
for i in range(100000):
	data_num = [rd.randint(0,row-1) for m in range(practise_num)]
	j = rd.randint(0,4)
	batch_x = train_datas[j][data_num];batch_y = train_labels[j][data_num]
	if i%100 == 0:
		train_accuracy = sess.run(accuracy,feed_dict={ x:batch_x, y_: batch_y, keep_prob: 1.0})
		print "step %d, training accuracy %g"%(i, train_accuracy)
	sess.run(train_step,feed_dict = {x:batch_x,y_:batch_y,keep_prob:0.5})

#评价训练结果
print "test accuracy %g"%sess.run(accuracy,feed_dict={x: test_datas, y_: test_labels,keep_prob:1.0})


