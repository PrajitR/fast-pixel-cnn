import fast_pixel_cnn_pp.model as model
import fast_pixel_cnn_pp.fast_nn as fast_nn
import fast_pixel_cnn_pp.plotting as plotting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=16,
    help='Number of images to generate simultaneously')
parser.add_argument(
    '-i',
    '--image_size',
    type=int,
    default=32,
    help='Height and width of the image')
parser.add_argument(
    '-s', '--seed', type=int, default=2702, help='Seed for random generation')
parser.add_argument(
    '-c',
    '--checkpoint',
    type=str,
    default='/home/mbz/pixel_cnn_pp/params_cifar.ckpt',
    help='Location of the pretrained checkpoint')
parser.add_argument(
    '-v',
    '--save_dir',
    type=str,
    default='/tmp',
    help='Location to save generated images to')
args = parser.parse_args()

g = tf.Graph()
with g.as_default():
    print('Creating model')
    input_channels = 4  # 3 channels for RGB and 1 channel of all ones 
    image_size = (args.batch_size, args.image_size, args.image_size,
                  input_channels)

    row_input = tf.placeholder(
        tf.float32, [args.batch_size, 1, args.image_size, input_channels],
        name='row_input')
    pixel_input = tf.placeholder(
        tf.float32, [args.batch_size, 1, 1, input_channels],
        name='pixel_input')
    row_id = tf.placeholder(tf.int32, [], name='row_id')
    col_id = tf.placeholder(tf.int32, [], name='col_id')
    ema = tf.train.ExponentialMovingAverage(0.9995)

    model_spec = tf.make_template('model', model.model_spec)
    sample, fast_nn_out, v_stack = model_spec(
        row_input, pixel_input, row_id, col_id, image_size, seed=args.seed)

    all_cache_variables = [
        v for v in tf.global_variables() if 'cache' in v.name
    ]
    initialize_cache = tf.variables_initializer(all_cache_variables)
    reset_cache = fast_nn.reset_cache_op()

    vars_to_restore = {
        k: v
        for k, v in ema.variables_to_restore().items() if 'cache' not in k
    }
    saver = tf.train.Saver(vars_to_restore)

    output_images = np.zeros(
        (args.batch_size, args.image_size, args.image_size, 3))

    sess = tf.Session()
    sess.run(initialize_cache)
    print('Loading checkpoint %s' % args.checkpoint)
    saver.restore(sess, args.checkpoint)

    batch = 0
    while True:
        print('Generating')
        sess.run(reset_cache)
        start_time = time.time()
        for row in range(args.image_size):
            # Implicit downshift.
            if row == 0:
                x_row_input = np.zeros(
                    (args.batch_size, 1, args.image_size, input_channels))
            else:
                x_row_input = output_images[:, (row - 1):row, :, :]
                x_row_input = np.concatenate(
                    (x_row_input, np.ones(
                        (args.batch_size, 1, args.image_size, 1))),
                    axis=3)

            sess.run(v_stack, {row_input: x_row_input, row_id: row})

            for col in range(args.image_size):
                # Implicit rightshift.
                if col == 0:
                    x_pixel_input = np.zeros(
                        (args.batch_size, 1, 1, input_channels))
                else:
                    x_pixel_input = output_images[:, row:(row + 1),
                                                  (col - 1):col, :]
                    x_pixel_input = np.concatenate(
                        (x_pixel_input, np.ones((args.batch_size, 1, 1, 1))),
                        axis=3)

                feed_dict = {
                    row_id: row,
                    col_id: col,
                    pixel_input: x_pixel_input
                }
                pixel_output = sess.run(sample, feed_dict)
                output_images[:, row:(row + 1),
                              col:(col + 1), :] = pixel_output

        end_time = time.time()
        print('Time taken to generate %d images: %.2f seconds' %
              (args.batch_size, end_time - start_time))

        plt.close('all')
        image_tile = plotting.img_tile(
            output_images, border_color=1.0, stretch=True)
        plotting.plot_img(image_tile)
        plt.savefig(os.path.join(args.save_dir, 'images_%d.png' % batch))

        batch += 1

plt.show()
