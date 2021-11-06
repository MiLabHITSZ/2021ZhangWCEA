from tensorflow import keras
from test_process import *
from attack import *
from load_data import *


# 执行自定义训练过程
def fashion_mnist_fnn_baseline(model, optimizer):
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    # 初始化模型
    model.build(input_shape=[128, 784])

    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(100):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                out = tf.squeeze(out, axis=1)

                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False))
                loss_print += float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(float(acc))
        print('epoch:', epoch, 'loss:', loss_print / (50000 / 128), 'Evaluate Acc:', float(acc))

    np.save('fashion_mnist_linear//linear//fashion_mnist_test_acc_baseline_100', np.array(acc_list))