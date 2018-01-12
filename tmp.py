import numpy as np
import tensorflow as tf


def customModel(features, labels, mode):
    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    model = W * features['x'] + b

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model
        )
    else:
        loss = tf.reduce_sum(tf.square(model - labels))
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model,
            loss=loss,
            train_op=train
        )


x_train = np.array([n / 100 for n in range(35, 65)])
y_train = np.array([n * 2 / 100 for n in range(35, 65)])
x_eval = np.array([n / 100 for n in range(20, 35)] + [n / 100 for n in range(65, 100)])
y_eval = np.array([n * 2 / 100 for n in range(20, 35)] + [n * 2 / 100 for n in range(65, 100)])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},
    y_train,
    batch_size=x_train.size,
    num_epochs=None,
    shuffle=True
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},
    y_train,
    batch_size=x_train.size,
    num_epochs=1000,
    shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval},
    y_eval,
    batch_size=x_train.size,
    num_epochs=1000,
    shuffle=False
)

estimator = tf.estimator.Estimator(model_fn=customModel)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)

value = np.array([15.])

check_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': value},
    num_epochs=1,
    shuffle=False
)

predictions = list(estimator.predict(input_fn=check_fn))

print('Predictions for %r is %r' % (value[0], predictions[0]))
