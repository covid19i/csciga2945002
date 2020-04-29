
import tensorflow as tf 
import numpy as np 
import time



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
x_train, x_test = x_train/255., x_test/255.

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
x_train = tf.reshape(x_train, shape=(-1, 784))
x_test  = tf.reshape(x_test, shape=(-1, 784))

weights = tf.Variable(tf.random.normal(shape=(784, 10), dtype=tf.float64))
biases  = tf.Variable(tf.random.normal(shape=(10,), dtype=tf.float64))

def logistic_regression(x):
    lr = tf.add(tf.matmul(x, weights), biases)
    #return tf.nn.sigmoid(lr)
    return lr


def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, 10)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    preds = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
    preds = tf.equal(y_true, preds)
    return tf.reduce_mean(tf.cast(preds, dtype=tf.float32))

def grad(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss_val = cross_entropy(y, y_pred)
    return tape.gradient(loss_val, [weights, biases])

batch_size = 1
n_batches = 10000
n_batches = int(input("Enter the number of batches (batch_size = %i): " % batch_size))
learning_rate = 0.001

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(x_train.shape[0]).batch(batch_size)

optimizer = tf.optimizers.SGD(learning_rate)

start = time.time()
for batch_numb, (batch_xs, batch_ys) in enumerate(dataset.take(n_batches), 1):
    gradients = grad(batch_xs, batch_ys)
    optimizer.apply_gradients(zip(gradients, [weights, biases]))
    if(batch_numb < 4):
        for j in range(490, 493):
            print("w[%i] = %f" % (j, weights[j][0].numpy()))
    if(batch_numb % ((int)(n_batches/5)) == 0 or batch_numb == 1):
        y_pred = logistic_regression(batch_xs)
        loss = cross_entropy(batch_ys, y_pred)
        acc = accuracy(batch_ys, y_pred)
        print("Batch number: %i, Training loss: %f, Training accuracy: %f" % (batch_numb, loss, acc))
        y_pred_test = logistic_regression(x_test)
        loss = cross_entropy(y_test, y_pred_test)
        acc = accuracy(y_test, y_pred_test)
        print("Batch number: %i, Testing loss: %f, Testing accuracy: %f" % (batch_numb, loss, acc))
t = time.time() - start


print("No of iterations: %i" %n_batches)
print("Lambda (Regularization Parameter): 0" )
print("Eta (Learning Rate): %f" %learning_rate)
y_pred_test = logistic_regression(x_test)
loss = cross_entropy(y_test, y_pred_test)
acc = accuracy(y_test, y_pred_test)
print("After Batch number: %i, loss: %f, Testing accuracy: %f" % (n_batches, loss, acc))
print("Time elapsed in  training: %f" % t)
time_per_iter = t/(batch_size * n_batches)
print("Time elapsed in training per data point= %f" % time_per_iter )

