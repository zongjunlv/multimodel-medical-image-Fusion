# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from generate import generate

patch_size = 84
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8

rc = 4


def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period=1):
    from datetime import datetime
    start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

    MODEL_SAVE_PATH = save_path + 'temporary.ckpt'
    num_imgs = source_imgs.shape[0]
    mod = num_imgs % BATCH_SIZE
    n_batches = int(num_imgs // BATCH_SIZE)
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source_imgs = source_imgs[:-mod]

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        SOURCE_VIS = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE_VIS')
        SOURCE_ir = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size / rc, patch_size / rc, 1),
                                   name='SOURCE_ir')
        print('source_vis shape:', SOURCE_VIS.shape)

        # upsampling vis and ir images
        G = Generator('Generator')
        generated_img = G.transform(vis=SOURCE_VIS, ir=SOURCE_ir)
        print('generate:', generated_img.shape)
        g0 = tf.nn.avg_pool(generated_img, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        generated_img_ds = tf.nn.avg_pool(g0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        D1 = Discriminator1('Discriminator1')
        grad_of_vis = grad(SOURCE_VIS)
        D1_real = D1.discrim(SOURCE_VIS, reuse=False)
        D1_fake = D1.discrim(generated_img, reuse=True)

        D2 = Discriminator2('Discriminator2')
        D2_real = D2.discrim(SOURCE_ir, reuse=False)
        D2_fake = D2.discrim(generated_img_ds, reuse=True)

        ''' LOSS FUNCTION '''
        # Loss for Generator
        G_loss_GAN_D1 = -tf.reduce_mean(tf.log(D1_fake + eps))
        G_loss_GAN_D2 = -tf.reduce_mean(tf.log(D2_fake + eps))
        G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2

        LOSS_IR = Fro_LOSS(generated_img_ds - SOURCE_ir)
        LOSS_VIS = L1_LOSS(grad(generated_img) - grad_of_vis)
        G_loss_norm = LOSS_IR + 1.2 * LOSS_VIS
        G_loss = G_loss_GAN + 0.8 * G_loss_norm

        # Loss for Discriminator1
        D1_loss_real = -tf.reduce_mean(tf.log(D1_real + eps))
        D1_loss_fake = -tf.reduce_mean(tf.log(1. - D1_fake + eps))
        D1_loss = D1_loss_fake + D1_loss_real

        # Loss for Discriminator2
        D2_loss_real = -tf.reduce_mean(tf.log(D2_real + eps))
        D2_loss_fake = -tf.reduce_mean(tf.log(1. - D2_fake + eps))
        D2_loss = D2_loss_fake + D2_loss_real

        current_iter = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=current_iter,
                                                   decay_steps=int(n_batches), decay_rate=DECAY_RATE,
                                                   staircase=False)

        # theta_de = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'deconv_ir')
        theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')
        theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2')

        G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step=current_iter,
                                                                         var_list=theta_G)
        G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step=current_iter,
                                                                     var_list=theta_G)
        D1_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D1_loss, global_step=current_iter,
                                                                              var_list=theta_D1)
        D2_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D2_loss, global_step=current_iter,
                                                                              var_list=theta_D2)
        # 获取可训练变量：
        # theta_G：获取生成器（Generator）中的所有可训练变量。
        # theta_D1：获取判别器1（Discriminator1）中的所有可训练变量。
        # theta_D2：获取判别器2（Discriminator2）中的所有可训练变量。
        # 定义优化器：
        # G_GAN_solver：使用RMSProp优化器最小化生成器的GAN损失（G_loss_GAN），并更新生成器的参数。
        # G_solver：使用RMSProp优化器最小化生成器的总损失（G_loss），并更新生成器的参数。
        # D1_solver：使用梯度下降优化器最小化判别器1的损失（D1_loss），并更新判别器1的参数。
        # D2_solver：使用梯度下降优化器最小化判别器2的损失（D2_loss），并更新判别器2的参数

        clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
        clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
        clip_D2 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D2]
        # 这段代码的功能是对生成器（Generator）、判别器1（Discriminator1）和判别器2（Discriminator2）的参数进行裁剪，
        # 确保它们的值在 -8到8之间。

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=500)
        # 初始化全局变量：sess.run(tf.global_variables_initializer())：
        # 这行代码调用TensorFlow的global_variables_initializer方法，初始化图中的所有全局变量。这是训练模型前的必要步骤，确保所有变量都有初始值。
        # 创建保存模型的Saver对象：saver = tf.train.Saver(max_to_keep=500)：
        # 这行代码创建了一个Saver对象，用于保存和恢复模型。max_to_keep = 500参数表示最多保留500个检查点文件，防止保存过多的检查点文件占用大量磁盘空间

        tf.summary.scalar('G_Loss_D1', G_loss_GAN_D1)
        tf.summary.scalar('G_Loss_D2', G_loss_GAN_D2)
        tf.summary.scalar('D1_real', tf.reduce_mean(D1_real))
        tf.summary.scalar('D1_fake', tf.reduce_mean(D1_fake))
        tf.summary.scalar('D2_real', tf.reduce_mean(D2_real))
        tf.summary.scalar('D2_fake', tf.reduce_mean(D2_fake))
        tf.summary.scalar('Learning rate', learning_rate)
        # tf.summary.scalar的功能是记录name的标量数据，这个标量可以在TensorBoard中可视化，帮助监控生成器在训练过程中的表现

        merged = tf.summary.merge_all()
        # 合并所有摘要
        writer = tf.summary.FileWriter("logs/", sess.graph)
        # 创建一个文件写入器，用于将摘要数据写入指定目录log/，sess.graph将计算图写入日志文件，用于可视化

        # ** Start Training **
        step = 0
        count_loss = 0
        num_imgs = source_imgs.shape[0]

        for epoch in range(EPOCHS):
            np.random.shuffle(source_imgs)
            # 对source_img随机打乱
            for batch in range(n_batches):
                ir_batch = np.zeros(shape=(BATCH_SIZE, int(patch_size / rc), int(patch_size / rc)),
                                    dtype=np.float32)
                # 数组的形状为 (BATCH_SIZE, int(patch_size / rc), int(patch_size / rc))，
                # 表示每个批次包含 BATCH_SIZE 个红外图像，每个图像的尺寸为 int(patch_size / rc) x int(patch_size / rc)。
                step += 1
                current_iter = step
                VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
                ir_or_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
                for b in range(BATCH_SIZE):
                    ir_batch[b, :, :] = scipy.ndimage.zoom(ir_or_batch[b, :, :], 0.25)
                VIS_batch = np.expand_dims(VIS_batch, -1)
                ir_batch = np.expand_dims(ir_batch, -1)
                FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_ir: ir_batch}

                it_g = 0
                it_d1 = 0
                it_d2 = 0
                # run the training step
                if batch % 2 == 0:
                    sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
                    it_d1 += 1
                    sess.run([D2_solver, clip_D2], feed_dict=FEED_DICT)
                    it_d2 += 1
                else:
                    sess.run([G_solver, clip_G], feed_dict=FEED_DICT)
                    it_g += 1
                g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict=FEED_DICT)

                if batch % 2 == 0:
                    while d1_loss > 1.9 and it_d1 < 20:
                        sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
                        d1_loss = sess.run(D1_loss, feed_dict=FEED_DICT)
                        it_d1 += 1
                    while d2_loss > 1.9 and it_d2 < 20:
                        sess.run([D2_solver, clip_D2], feed_dict=FEED_DICT)
                        d2_loss = sess.run(D2_loss, feed_dict=FEED_DICT)
                        it_d2 += 1
                else:
                    while (d1_loss < 1 or d2_loss < 1) and it_g < 20:
                        sess.run([G_GAN_solver, clip_G], feed_dict=FEED_DICT)
                        g_loss, d1_loss, d2_loss = sess.run([G_loss, D1_loss, D2_loss], feed_dict=FEED_DICT)
                        it_g += 1
                    while (g_loss > 200) and it_g < 20:
                        sess.run([G_solver, clip_G], feed_dict=FEED_DICT)
                        g_loss = sess.run(G_loss, feed_dict=FEED_DICT)
                        it_g += 1
                print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))
                if batch % 10 == 0:
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('G_loss: %s, D1_loss: %s, D2_loss: %s' % (
                        g_loss, d1_loss, d2_loss))
                    print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

                result = sess.run(merged, feed_dict=FEED_DICT)
                writer.add_summary(result, step)
                if step % logging_period == 0:
                    saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')

                is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
                if is_last_step or step % logging_period == 0:
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
                        epoch + 1, EPOCHS, step, lr, elapsed_time))

    writer.close()
    saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def grad(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g
