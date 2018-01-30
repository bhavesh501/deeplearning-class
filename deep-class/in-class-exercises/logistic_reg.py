import xlrd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pandas as pd

DATA_FILE = 'data/SUV.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
data = data[:, [2,3,4]]
n_samples = sheet.nrows - 1

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(data)

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X1 = tf.placeholder(tf.float32, name='X1')
X2 = tf.placeholder(tf.float32, name='X2')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w1 = tf.Variable(-1.0, name='weights1')
w2 = tf.Variable(-1.0, name='weights2')
b = tf.Variable(-1.0, name='bias')

# Step 4: build model to predict Y
z = X1 * w1 + X2 * w2 + b
Y_predicted = 1/(1 + pow(math.e, -z))

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(50):  # train the model 50 epochs
        total_loss = 0
        for x1,x2, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([optimizer, loss], feed_dict={X1: x1,X2:x2, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w1,w2, b = sess.run([w1,w2, b])
"""predicted = Y_predicted
predicted = np.reshape(predicted, (400, 1))
predicted = sc.inverse_transform(Y_predicted)"""

# plot the results
X1, X2 = data.T[0], data.T[1]
X1 = np.reshape(X1, (400, 1))
X2 = np.reshape(X2, (400, 1))
X = pd.DataFrame(np.column_stack([X1, X2]))
Y = data.T[2]
Y = np.reshape(Y, (400, 1))
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, 1/(1 + pow(math.e, -(X1 * w1 + X2 * w2 + b))), 'r', label='Predicted data')
plt.legend()
plt.show()