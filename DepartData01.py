#coding=utf-8
import tensorflow as tf
import numpy as np
import random as rd


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
def normalization(x):
	return (x-128.0)/128.0

#测试数据集的预处理
test = unpickle("test_batch")
row = test['data'].shape[0]
column = test['data'].shape[1]
test_datas = normalization(test['data'])
test_labels = OneHot_Encoding(test['labels'])

#训练数据集预处理
data = [];label=[]
for i in range(5):
	file = "data_batch_%d" % (i+1)
	temp = unpickle(file)
	data.append(normalization(temp['data']))
	label.append(OneHot_Encoding(temp['labels']))
train_datas = np.array(data);train_labels = np.array(label)
#搭建预测模型softmax
x = tf.placeholder("float",[None,column])
W = tf.Variable(tf.zeros([column,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) 

#初始化所有变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#开始训练，每一次batch按照practise_num的数据量进行训练，训练10000次
for i in range(100000):
	data_num = [rd.randint(0,row-1) for m in range(practise_num)]
	j = rd.randint(0,4)
	batch_x = train_datas[j][data_num];batch_y = train_labels[j][data_num]
	if i%100 == 0:
		train_accuracy = sess.run(accuracy,feed_dict={ x:batch_x, y_: batch_y})
		print "step %d, training accuracy %g"%(i, train_accuracy)
	sess.run(train_step,feed_dict = {x:batch_x,y_:batch_y})

#评价训练结果
print sess.run(accuracy,feed_dict={x:test_datas,y_:test_labels})

