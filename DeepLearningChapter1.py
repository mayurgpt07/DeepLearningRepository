import tensorflow as tf
import warnings

warnings.simplefilter('ignore')

graph1 = tf.Graph()

with graph1.as_default():
	constant = tf.constant([2])
	array = tf.constant([1,2,3])
	matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]]) #3 X 3
	tensor = tf.constant([[[1,2,3],[4,5,6],[7,8,9], [10,11,12]], [[1,2,3],[4,5,6],[7,8,9], [10,11,12]]]) #2 X 4 X 3

with tf.compat.v1.Session(graph = graph1) as sess:
	#constant 
	resultConstant = sess.run(constant)
	print('The shape and value of constant', resultConstant.shape, resultConstant)
	#array
	resultArray = sess.run(array)
	print('The shape and value of constant', resultArray.shape, resultArray)	
	#matrix
	resultMatrix = sess.run(matrix)
	print('The shape and value of constant', resultMatrix.shape, resultMatrix)	
	#tensor
	resultTensor = sess.run(tensor)
	print('The shape and value of constant', resultTensor.shape, resultTensor)

#Variable are used for manipulation in the graph it cannot recieve from outside
variable = tf.Variable([0])

counter = tf.compat.v1.assign(variable, variable + 1)

initialize = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
	sess.run(initialize)
	print(sess.run(variable))
	for i in range(3):
		sess.run(counter)
		print(sess.run(variable))