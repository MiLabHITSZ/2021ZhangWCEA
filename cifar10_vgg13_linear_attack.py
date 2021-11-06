from tensorflow import keras
from test_process import *
from attack import *
from load_data import *


def cifar10_vgg13_linear_attack_train(conv_net, fc_net, optimizer, num_weight, total_epoch, gama):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()
    x_train, y_train, x_test, y_test = load_cifar10()

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    # 将训练集、测试集转成灰度图片
    x_train = rbg_to_grayscale(x_train)
    x_test = rbg_to_grayscale(x_test)
    # 将训练集图片转成tensor并展平
    x_train_process = tf.convert_to_tensor(x_train.flatten(), dtype=tf.float32) / 2550

    acc_list = []
    mape_list = []

    for epoch in range(total_epoch):
        loss = tf.constant(0, dtype=tf.float32)
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out1 = conv_net(x_batch, training=True)
                out = fc_net(out1, training=True)
                out = tf.squeeze(out, axis=[1, 2])
                # 计算恶意正则项
                regular = linear_attack(conv_net, x_train_process, num_weight)
                loss_batch = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(y_batch, out, from_logits=True)) + gama * regular
            # 列表合并，合并2个自网络的参数
            variables = conv_net.trainable_variables + fc_net.trainable_variables
            # 对所有参数求梯度
            grads = tape.gradient(loss_batch, variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, variables))
            loss += loss_batch
        # 计算MAPE
        mape = calculate_linear_attack_mape(conv_net, x_train, num_weight)
        acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
        acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
        acc_list.append(float(acc_test))
        mape_list.append(mape)
        print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_train:', float(acc_train),
              'Evaluate Acc_test', float(
                acc_test), 'mape:', mape)
    data = show_linear_attack_data(conv_net, x_test, num_weight)
    test_accuracy_name = 'cifar10_linear//linear//cifar10_vgg13_linear_new_attack' + str(
        num_weight) + '_test_accuracy_' + str(gama) + '_' + str(total_epoch)
    mape_name = 'cifar10_linear//linear//cifar10_vgg13_linear_new_attack' + str(num_weight) + '_mape_' + str(
        gama) + '_' + str(total_epoch)
    data_name = 'cifar10_linear//linear//cifar10_vgg13_linear_new_attack' + str(num_weight) + '_stolen_data_' + str(
        gama) + '_' + str(total_epoch)
    np.save(test_accuracy_name, np.array(acc_list))
    np.save(mape_name, np.array(mape_list))
    np.save(data_name, data)
