import pandas as pd
import tensorflow as tf
import xlrd
import numpy as np

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

n_samples = sheet.nrows - 1
dataset = pd.read_csv('50_Startups.csv')

# separate x & y columns
y = dataset.Profit
X = dataset.drop(['Profit','State','R&D Spend'],axis=1)
#X = dataset.drop('R&D Spend',axis=1)
#X = dataset.drop('Administration',axis=1)
#X = dataset.drop('State',axis=1)


#print(y)

#print(X)
d1 = X['Administration']
d2=pd.Series(d1)
x1_data=pd.to_numeric(d2)

x2_data = X['Marketing Spend']
y_data = y

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print("")
#print (step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))