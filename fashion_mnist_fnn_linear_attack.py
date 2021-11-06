from tensorflow import keras
from test_process import *
from attack import *
from load_data import *


# 执行自定义训练过程
def fashion_mnist_fnn_linear_attack_train(model, optimizer, num_weight, gama, total_epoch):

    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    x_train_process = tf.convert_to_tensor(x_train.flatten(), dtype=tf.float32) / 2550  # 数据缩放至0-0.1
    # 初始化模型
    model.build(input_shape=[128, 784])

    loss_list = []
    acc_list = []
    mape_list = []

    # 执行训练过程
    for epoch in range(total_epoch):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                out = tf.squeeze(out, axis=1)
                regular = linear_attack(model, x_train_process, num_weight)
                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + gama * regular
                loss_print += float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        mape = calculate_linear_attack_mape(model, x_train, num_weight)
        mape_list.append(mape)
        # 获得对测试集的准确率
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(acc)
        print('epoch:', epoch, 'loss:', loss_print / (50000 / 128), 'Evaluate Acc:', float(acc), 'mape:', mape)
    test_accuracy_name = 'fashion_mnist_linear//linear//fashion_mnist_fnn_linear_attack' + str(
        num_weight) + '_test_accuracy_' + str(gama) + '_' + str(total_epoch)
    mape_name = 'fashion_mnist_linear//linear//fashion_mnist_fnn_linear_attack' + str(num_weight) + '_mape_' + str(
        gama) + '_' + str(total_epoch)
    data_name = 'fashion_mnist_linear//linear//fashion_mnist_fnn_linear_attack' + str(num_weight) + '_stolen_data_' + str(
        gama) + '_' + str(total_epoch)
    data = show_linear_attack_data(model, x_test, num_weight)
    np.save(test_accuracy_name, np.array(acc_list))
    np.save(mape_name, np.array(mape_list))
    np.save(data_name, data)