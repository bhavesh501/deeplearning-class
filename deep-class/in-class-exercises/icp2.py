from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
rng = np.random


# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
# Training Data
#Created some fake data
dataframe = [[230.1,37.8,69.2,22.1],[2230.1,32.8,61.2,21.1]] #pandas.read_csv("Advertising.csv", delim_whitespace=True, header=None)
dataset = dataframe

X1,X2,X3,y1 = [],[],[],[]
for i in range(0,len(dataset)):
    X = dataset[i][0]
    X1.append(np.float32(dataset[i][0]))
    X2.append(np.float32(dataset[i][1]))
    X3.append(np.float32(dataset[i][2]))
    y1.append(np.float32(dataset[i][3]))
#X=np.array([X1,X2,X3])
X = np.column_stack((X1,X2,X3)) ##MYEDIT: This combines all three values. If you find you need to stack in a different way then you will need to ensure the shapes below match this shape.
#X = np.column_stack((X,X3))

n_samples = len(X1)
#print(n_samples) = 17
# tf Graph Input
X_1 = tf.placeholder(tf.float32, [ None,3])##MYEDIT: Changed order
Y = tf.placeholder(tf.float32, [None])
# Set model weights
initial1 = tf.constant(rng.randn(), dtype=tf.float32, shape=[3,1]) ###MYEDIT: change order and you are only giving 1 sample at a time with your method of calling
initial2 = tf.constant(rng.randn(), dtype=tf.float32, shape=[3,1])
W1 = tf.Variable(initial_value=initial1)
b = tf.Variable(initial_value=initial2)


mul=tf.matmul(W1, X_1)   ##MYEDIT: remove matmul from pred for clarity and shape checking
# Construct a linear model
pred = tf.add(mul, b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x1, y) in zip(X, y1):
            Xformatted=np.array([x1])  #has shape (1,3)  #MYEDIT: separated this to demonstrate shapes
            yformatted=np.array([y])  #shape (1,)  #MYEDIT: separated this to demonstrate shapes
                                                    #NB. X_1 shape is (?,3)   and Y shape is (?,)
            sess.run(optimizer, feed_dict={X_1: Xformatted, Y: yformatted})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X_1: Xformatted, Y: yformatted})   #NB. x1 an y are out of scope here - you will only get the last values. Double check if this is what you meant.
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "Weights=", sess.run(W1),"b=", sess.run(b))
