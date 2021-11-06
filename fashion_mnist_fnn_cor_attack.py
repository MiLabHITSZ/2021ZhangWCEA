from tensorflow import keras
from test_process import *
from attack import *
from load_data import *


# 执行自定义训练过程
def fashion_mnist_fnn_cor_attack_train(model, optimizer, gama):
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    # 初始化模型
    model.build(input_shape=[128, 784])

    loss_list = []
    acc_list = []
    mape_list = []
    # 将训练集数据转成tensor并展平
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_train_tensor = tf.reshape(x_train_tensor, [-1, ]) / 255

    # 将模型权重列表中的权重参数连接并展平
    params = tf.concat([tf.reshape(p, (-1, )) for p in model.trainable_variables if len(p.shape) > 1], axis=0)

    # 根据模型参数的数量决定要窃取数据的大小
    stolenData = x_train_tensor[0:params.shape[0]]/255
    d_mean = tf.reduce_mean(stolenData)
    d_m = stolenData - d_mean

    # 执行训练过程
    for epoch in range(100):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                out = tf.squeeze(out, axis=1)
                regular = cor_attack(model, d_m)
                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) - gama*regular
                loss_print += float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        mape = calculate_cor_mape(model, x_train)
        mape_list.append(mape)
        print('mape', mape)
        # 获得对测试集的准确率
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(float(acc))
        print('epoch:', epoch, 'Evaluate Acc:', float(acc))
    stolen_data = recover_cor_stolen_data(model, x_train)
    test_accuracy_name = 'fashion_mnist_linear//cor//fashion_mnist_fnn_cor_attack' + '_test_accuracy_' + str(gama) + '_' + str(100)
    mape_name = 'fashion_mnist_linear//cor//fashion_mnist_fnn_cor_attack' + '_mape_' + str(gama) + '_' + str(100)
    data_name = 'fashion_mnist_linear//cor//fashion_mnist_fnn_cor_attack' + '_stolen_data_' + str(gama) + '_' + str(100)
    np.save(test_accuracy_name, acc_list)
    np.save(mape_name, mape_list)
    np.save(data_name, stolen_data)