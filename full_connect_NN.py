import numpy as np
import random
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel(r'analysis.xls',sheetname=0)

le = preprocessing.LabelEncoder()
for i in df.columns.values[1:8]:
    le.fit(df[i])
    df[i] = le.transform(df[i])
    df[i] = df[i].astype('int')

#trainX = np.array(df.loc[0:1999,'Water Body Type':'DOM'])
#trainy = np.array(df.loc[0:1999,'Price'])
trainX = np.array(df.loc[0:1999,'Price':'Garage Capacity'])
trainy = np.array(df.loc[0:1999,'DOM'])
trainy = trainy.reshape(2000,1)

#testX = np.array(df.loc[1999:,'Water Body Type':'DOM'])
#testy = np.array(df.loc[1999:,'Price'])
testX = np.array(df.loc[1999:,'Price':'Garage Capacity'])
testy = np.array(df.loc[1999:,'DOM'])
testy = testy.reshape(np.shape(testy)[0],1)


x = tf.placeholder(tf.float32,[None,19])
y = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.truncated_normal([19,200],stddev=0.1))
b1 = tf.Variable(tf.zeros([200])+0.1)
z1 = tf.matmul(x,W1)+b1
h1 = tf.nn.relu(z1)

W2 = tf.Variable(tf.truncated_normal([200,50],stddev=0.1))
b2 = tf.Variable(tf.zeros([50])+0.1)
z2 = tf.matmul(z1,W2)+b2
h2 = tf.nn.tanh(z2)

W3 = tf.Variable(tf.truncated_normal([50,1],stddev=0.1))
b3 = tf.Variable(tf.zeros([1])+0.1)
z3 = tf.matmul(h2,W3)+b3
h3 = tf.nn.tanh(z3)

#W4 = tf.Variable(tf.truncated_normal([100,1],stddev=0.1))
#b4 = tf.Variable(tf.zeros([1])+0.1)
#z4 = tf.matmul(h3,W4)+b4

y_prediction = z3
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prediction), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.cross_entropy_with_logits(labels = y, logits = y_prediction))
loss = tf.reduce_mean((y-y_prediction)**2)

Train = tf.train.GradientDescentOptimizer(0.000001).minimize(loss) #DOM:0.00001
#Train = tf.train.MomentumOptimizer(learning_rate=0.5,momentum = 0.9).minimize(loss)

init = tf.global_variables_initializer()


#correct_p = tf.less_equal(abs(y-y_prediction), y*0.3)
correct_p = tf.less_equal(abs(y-y_prediction), 7)
accuracy = tf.reduce_mean(tf.cast(correct_p,tf.float32))

# using validation set to determine hyperparameters.
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1500): #DOM:650 #price:850
        sess.run(Train,feed_dict={x:trainX,y:trainy})
        acc = sess.run(accuracy,feed_dict={x:testX, y:testy})
        cost = sess.run(loss,feed_dict={x:trainX,y:trainy})
        #print('batch: '+ str(batch)+',node: '+str(node)+', learning rate: '+str(lr))
        print('Iter'+str(epoch+1)+',Training loss: '+str(cost)+', Accuracy: '+str(acc))