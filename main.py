from build_model import *
import os
import tensorflow as tf

from mnist_fnn_linear_attack import *
from mnist_fnn_cor_attack import *
from mnist_fnn_baseline import *
from mnist_fnn_sign_attack import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from fashion_mnist_fnn_baseline import *
from fashion_mnist_fnn_linear_attack import *
from fashion_mnist_fnn_cor_attack import *
from cifar10_cnn_baseline import *
from cifar10_vgg13_linear_attack import *
from cifar10_vgg13_cor_attack import *
from cifar10_vgg13_linear_color_attack import *
from cifar10_vgg13_cor_color_attack import *

if __name__ == '__main__':
    tf.random.set_seed(999)
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    # session = tf.compat.v1.Session(config=config)

    # cifar10 cnn 线性权重惩罚项攻击

    # conv_net5, fc_net5, optimizer = build_vgg13_model(0.0001)
    # cifar10_vgg13_cor_attack_train(conv_net5, fc_net5, optimizer, 10, 200)
    # cifar10_vgg13_linear_color_attack_train(conv_net5, fc_net5, optimizer, 5, 200, 10)
    # cifar10_vgg13_cor_color_attack_train(conv_net5, fc_net5, optimizer, 30, 20)
    # conv_net6, fc_net6, optimizer = build_vgg13_model(0.0001)
    # cifar10_vgg13_linear_attack_train(conv_net6, fc_net6, optimizer, 5, 200, 10)

    # MNIST Fashion-MNIST 线性权值攻击
    gama_list = [120, 150]
    #
    for gama in gama_list:
        model, optimizer = build_mnist_fnn_model()
        mnist_fnn_linear_attack_train(model, optimizer, 3, gama, 100)
    # for gama in gama_list:
    #     model, optimizer = build_mnist_fnn_model()
    #     fashion_mnist_fnn_linear_attack_train(model, optimizer, 3, gama, 100)

    # MNIST cor攻击
    # for gama in [10]:
    #     model, optimizer = build_mnist_fnn_model()
    #     mnist_fnn_cor_attack_train(model, optimizer, gama)
    #
    # for gama in gama_list:
    #     model, optimizer = build_mnist_fnn_model()
    #     fashion_mnist_fnn_cor_attack_train(model, optimizer, gama)
    # gama_list = [30]
    # for gama in gama_list:
    #     conv_net, fc_net, optimizer = build_vgg13_model(0.0001)
    #     cifar10_vgg13_cor_attack_train(conv_net, fc_net, optimizer, gama, 200)