from . import model
from . import fast_nn

import tensorflow as tf
import numpy as np

import os
import unittest


class FastPixelCNNPPEndToEndTest(tf.test.TestCase):
    def test_end_to_end(self):
        with self.test_session() as sess:
            print('Creating model')
            image_size = (10, 32, 32, 4)
            batch_size, image_height, image_width, image_channels = image_size

            # Create placeholders.
            row_input = tf.placeholder(
                tf.float32, [batch_size, 1, image_width, image_channels],
                name='row_input')
            pixel_input = tf.placeholder(
                tf.float32, [batch_size, 1, 1, image_channels],
                name='pixel_input')
            row_id = tf.placeholder(tf.int32, [], name='row_id')
            col_id = tf.placeholder(tf.int32, [], name='col_id')
            ema = tf.train.ExponentialMovingAverage(0.9995)

            # Create the model.
            model_spec = tf.make_template('model', model.model_spec)
            sample, fast_nn_out, v_stack = model_spec(
                row_input, pixel_input, row_id, col_id, image_size)

            # Initialize the caches.
            cache_variables = [
                v for v in tf.global_variables() if 'cache' in v.name
            ]
            sess.run(tf.variables_initializer(cache_variables))

            # Load the pretrained model
            print('Restoring variables')
            vars_to_restore = {
                k: v
                for k, v in ema.variables_to_restore().items()
                if 'cache' not in k
            }
            saver = tf.train.Saver(vars_to_restore)
            ckpt_path = None
            assert ckpt_path, 'Provide a path to the checkpoint in this file'
            saver.restore(sess, ckpt_path)

            # Create the fixed random input.
            np.random.seed(2702)
            x = np.random.randint(0, 256, size=(10, 32, 32, 3))
            x = np.cast[np.float32]((x - 127.5) / 127.5)
            x_pad = np.concatenate(
                (x, np.ones((batch_size, 32, 32, 1))), axis=3)
            x_downshift = fast_nn.down_shift(x_pad)
            x_rightshift = fast_nn.right_shift(x_pad)

            # Holds the output.
            num_output_features = 10 * 10
            output_features = np.zeros(
                (batch_size, 32, 32, num_output_features))

            # Compute all features.
            print('Computing features')
            sess.run(fast_nn.reset_cache_op())
            for row in range(image_height):
                x_row_input = x_downshift[:, row:(row + 1), :, :]
                sess.run(v_stack, {row_input: x_row_input, row_id: row})

                for col in range(image_width):
                    x_pixel_input = x_rightshift[:, row:(row + 1),
                                                 col:(col + 1), :]
                    feed_dict = {
                        row_id: row,
                        col_id: col,
                        pixel_input: x_pixel_input
                    }
                    pixel_features = sess.run(fast_nn_out, feed_dict)
                    output_features[:, row:(row + 1), col:(
                        col + 1), :] = pixel_features

            ground_truth_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             'ground_truth_output.npy')
            ground_truth_features = np.load(ground_truth_file)
            total_features = np.prod(output_features[0].shape)
            for i in range(batch_size):
                self.assertTrue(
                    np.allclose(
                        output_features[i, :, :, :],
                        ground_truth_features[i, :, :, :],
                        atol=1e-4))
