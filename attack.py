import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
from PIL import ImageOps, Image


def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def normalize(x):
    x = x.flatten()
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


def cal_error(img1, img2):
    # print(type(img1))
    # print(type(img2))
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    return np.mean(np.abs(img1 - img2))


# 线性权重窃取方法-添加与窃取数据相关联的正则项

def linear_attack(model, x_train, num_weight):
    params = tf.concat([tf.reshape(p, (-1,)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    print(len(params))
    num_pixel = int(len(params) / num_weight)
    data = x_train[:num_pixel]
    params = params[:num_pixel * num_weight]
    params_sum = tf.convert_to_tensor(np.zeros(num_pixel), dtype=tf.float32)
    for i in range(num_weight):
        params_sum += params[i * num_pixel:(i + 1) * num_pixel]
    assert params_sum.shape[0] == data.shape[0]
    regular = tf.reduce_mean(tf.abs(data - tf.abs(params_sum)))
    return regular


def calculate_linear_attack_mape(model, x_train, num_weight):
    # 将模型参数连接到一起并展平
    params = tf.concat([tf.reshape(p, (-1,)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 计算窃取的像素点并截取模型参数
    num_pixel = int(len(params) / num_weight)
    params = params[:num_pixel * num_weight]
    # 根据窃取每个像素点需要的模型参数数量将模型参数分段相加
    params_sum = tf.convert_to_tensor(np.zeros(num_pixel), dtype=tf.float32)
    for i in range(num_weight):
        params_sum += params[i * num_pixel:(i + 1) * num_pixel]
    # 将溢出的数据据转成0.1
    stolen_data = params_sum.numpy() * 2550
    stolen_data[stolen_data > 255] = 255
    # 计算可以窃取的图片数量并截取模型参数
    pixel = np.prod(x_train.shape[1:])
    num_image = int(len(stolen_data) / pixel)
    stolen_data = stolen_data[:num_image * pixel]
    # 将模型参数的shape转成图片shape
    stolen_data = stolen_data.reshape(num_image, x_train.shape[1], x_train.shape[2])
    data = x_train[:num_image]
    # 计算MAPE
    assert stolen_data.shape == data.shape
    mape = np.mean(abs(abs(stolen_data) - data))
    return mape


def show_linear_attack_data(model, x_train, num_weight):
    params = tf.concat([tf.reshape(p, (-1,)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    num_pixel = int(len(params) / num_weight)
    params = params[:num_pixel * num_weight]
    params_sum = tf.convert_to_tensor(np.zeros(num_pixel), dtype=tf.float32)
    for i in range(num_weight):
        params_sum += params[i * num_pixel:(i + 1) * num_pixel]
    stolen_data = tf.abs(params_sum).numpy()
    image_num_pixel = np.prod(x_train.shape[1:])
    num_image = int(len(stolen_data) / image_num_pixel)
    stolen_data = stolen_data[:num_image * image_num_pixel]
    data = stolen_data.reshape(num_image, x_train.shape[1], x_train.shape[2]) * 2550
    return data


def show_linear_color_attack_data(model, x_train, num_weight):
    params = tf.concat([tf.reshape(p, (-1,)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    num_pixel = int(len(params) / num_weight)
    params = params[:num_pixel * num_weight]
    params_sum = tf.convert_to_tensor(np.zeros(num_pixel), dtype=tf.float32)
    for i in range(num_weight):
        params_sum += params[i * num_pixel:(i + 1) * num_pixel]
    stolen_data = tf.abs(params_sum).numpy()
    image_num_pixel = np.prod(x_train.shape[1:])
    num_image = int(len(stolen_data) / image_num_pixel)
    stolen_data = stolen_data[:num_image * image_num_pixel]
    data = stolen_data.reshape(num_image, x_train.shape[1], x_train.shape[2], x_train.shape[3]) * 2550
    return data

def calcu_cor_color_mape(model, x_train):
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    params = normalize(params.numpy())
    params = (params * 255).astype(np.uint8)
    num_pixel = int(np.prod(x_train.shape[1:]))
    num_image = int(len(params) / num_pixel)
    params = params[:num_image * num_pixel]
    params = params.reshape(num_image, x_train.shape[1], x_train.shape[2], x_train.shape[3])
    mape = 0
    for i in range(num_image):
        err1 = cal_error(params[i], x_train[i])
        err2 = cal_error(np.asarray(Image.fromarray(params[i])), x_train[i])
        mape += min([err1, err2])
        mape += err1
    return mape / num_image

def recover_cor_color_stolen_data(model, x_train):
    # 将模型权重列表中的权重参数连接并展平
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 将模型参数正则化
    params = normalize(params.numpy())
    total_pix = np.prod(x_train.shape[1:])
    number = int(params.shape[0] / total_pix)
    print("steal number:", number)
    params = params[0:number * total_pix]
    params = params.reshape(number, x_train.shape[1], x_train.shape[2], x_train.shape[3])
    params = (params * 255).astype(np.uint8)
    return params


# 相关值编码攻击窃取方法
def cor_attack(model, d_m):
    # 将模型权重列表中的权重参数连接并展平
    params = tf.concat([tf.reshape(p, (-1,)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 计算相关性函数
    p_mean = tf.reduce_mean(params)
    p_m = params - p_mean
    r_num = tf.reduce_sum(p_m * d_m)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(p_m)) * tf.reduce_sum(tf.square(d_m)))
    r = r_num / r_den
    loss = tf.abs(r)
    return loss


def calculate_cor_mape(model, x_train):
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    params = normalize(params.numpy())
    params = (params * 255).astype(np.uint8)
    num_pixel = int(np.prod(x_train.shape[1:]))
    num_image = int(len(params) / num_pixel)
    params = params[:num_image * num_pixel]
    params = params.reshape(num_image, x_train.shape[1], x_train.shape[2])
    mape = 0
    for i in range(num_image):
        err1 = cal_error(params[i], x_train[i])
        err2 = cal_error(np.asarray(ImageOps.invert(Image.fromarray(params[i]))), x_train[i])
        mape += min([err1, err2])
    return mape / num_image


def recover_cor_stolen_data(model, x_train):
    # 将模型权重列表中的权重参数连接并展平
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 将模型参数正则化
    params = normalize(params.numpy())
    total_pix = np.prod(x_train.shape[1:])
    number = int(params.shape[0] / total_pix)
    print("steal number:", number)
    params = params[0:number * total_pix]
    params = params.reshape(number, x_train.shape[1], x_train.shape[2])
    params = (params * 255).astype(np.uint8)
    return params


# 将训练集数据转为二进制
def get_binary_secret(X):
    # convert 8-bit pixel images to binary with format {-1, +1}
    assert X.dtype == np.uint8
    s = np.unpackbits(X.flatten())
    s = s.astype(np.float32)
    s[s == 0] = -1
    return s


# 符号编码攻击
def mnist_fnn_sign_attack(model, stolen_data):
    # 将模型权重列表中的权重参数连接并展平
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 将要窃取的二进制字符串与模型参数相乘，并把异号的取出作为惩罚项
    constraints = params * stolen_data
    constraints = tf.squeeze(constraints, axis=1)
    index = tf.where(constraints < 0)
    penalty = tf.gather(constraints, index)
    penalty = tf.abs(penalty)
    loss = tf.reduce_mean(penalty)
    # 获得符号编码的准确率
    accuracy = (params.shape[0] - penalty.shape[0]) / params.shape[0]
    return loss, accuracy


def recover_sign_data(model, x_train):
    # 将模型参数展平
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    # 获得模型参数的正负号并转成二进制编码
    sign = (params / tf.abs(params)).numpy()
    sign[sign == -1] = 0
    # 将二进制编码转成图片像素值
    bits = sign.astype(np.uint8)
    # 获得要窃取的图片数量
    number_image = int(len(sign) / np.prod(x_train.shape[1:]) / 8)
    # 根据要窃取的图片数量截断二进制编码串
    bits = bits[:number_image * np.prod(x_train.shape[1:]) * 8]
    # 将截断的二进制编码串
    imgs = np.packbits(bits.reshape(-1, 8)).reshape(number_image, x_train.shape[1], x_train.shape[2])
    np.save('mnist_fnn_sign_stolen_data', imgs)

    for i in range(5):
        plt.imshow(imgs[60 + i], cmap='gray')
        plt.axis('off')
        plt.show()


def calcu_sign_mape(model, x_train):
    params = tf.concat([tf.reshape(p, (-1, 1)) for p in model.trainable_variables if len(p.shape) > 1], axis=0)
    sign = (params / tf.abs(params)).numpy()
    sign[sign == -1] = 0
    bits = sign.astype(np.uint8)
    number_image = int(len(sign) / np.prod(x_train.shape[1:]) / 8)
    bits = bits[:number_image * np.prod(x_train.shape[1:]) * 8]
    imgs = np.packbits(bits.reshape(-1, 8))
    # 计算MAPE
    x_train = x_train.flatten()
    x_train = x_train[:len(imgs)]
    mape = np.mean(abs(x_train - imgs))
    return mape


def recover_label_data(y, name):
    assert isinstance(y, np.ndarray)
    data = np.zeros(int(y.shape[0] / 2))
    for i in range(len(data)):
        data[i] = y[2 * i] + y[2 * i + 1]
        # data[i] = data[i] * (2 ** 4)
        if data[i] > 15:
            data[i] = 15
    if name == 'cifar10':
        data = np.reshape(data, [-1, 32, 32])
    elif name == 'mnist':
        data = np.reshape(data, [-1, 28, 28])
    data = data.astype(int)
    # 显示数据
    for i in range(data.shape[0]):
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')
        plt.show()


def show_data(x_test, num):
    for i in range(num):
        plt.imshow(x_test[i], cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # cifar10 原始图片灰度展示
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_test_in = rbg_to_grayscale(x_train)
    # mal_data_synthesis(x_test_in, 20, 4)
    show_data(x_test_in, 10)

    # recover_label_data(mal_y, 'cifar10')
    # show_data(x_test_in, 9)
    # print(y_test[0])
