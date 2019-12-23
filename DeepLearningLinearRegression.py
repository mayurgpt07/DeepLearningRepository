import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

plt.rcParams['figure.figsize'] = (10, 6)
plt.show()

X = np.arange(0.0, 5.0, 0.1)
a = 1
b = 0

Y= a * X + b 

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

df = pd.read_csv('FuelConsumption.csv', sep = ',', header = 0)
print(df.head())

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b

loss = tf.reduce_mean(tf.square(y - train_y))


optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

initialize = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
	sess.run(initialize)
	loss_values = []
	train_data = []
	#print(sess.run(coorelation))
	for step in range(100):
		i, loss_val, a_val, b_val = sess.run([train, loss, a, b])
		loss_values.append(loss_val)
		if step%5 == 0:
			print(step,loss_val, a_val, b_val)
			train_data.append([a_val, b_val])
	plt.plot(loss_values)
	plt.show()
