from . import nn
from . import fast_nn

import tensorflow as tf
import numpy as np

import math
import unittest
from collections import namedtuple

Placeholders = namedtuple(
    'Placeholders',
    ['full_input', 'pixel_input', 'row_input', 'row_id', 'col_id'])


class FastPixelCNNPPTest(tf.test.TestCase):
    def test_down_shifted_conv2d_layer1_stride1(self):
        self._test_down_shifted(num_layers=1)

    def test_down_shifted_conv2d_layer1_stride1_1by3(self):
        self._test_down_shifted(num_layers=1, filter_size=[1, 3])

    def test_down_shifted_conv2d_layer2_stride1(self):
        self._test_down_shifted(num_layers=2)

    def test_down_shifted_conv2d_layer3_stride1(self):
        self._test_down_shifted(num_layers=3)

    def test_down_shifted_conv2d_layer2_stride1_2(self):
        self._test_down_shifted(num_layers=2, strides=[1, 2])

    def test_down_shifted_conv2d_layer1_stride2(self):
        self._test_down_shifted(num_layers=1, strides=[2])

    def test_down_shifted_conv2d_layer3_stride1_2_2(self):
        self._test_down_shifted(num_layers=3, strides=[1, 2, 2])

    def test_down_shifted_deconv2d_layer1_stride2(self):
        self._test_down_shifted(num_layers=1, strides=[2], layers=['deconv'])

    def test_down_shifted_deconv2d_layer2_stride2(self):
        self._test_down_shifted(
            num_layers=2, strides=[2, 2], layers=['deconv', 'deconv'])

    def test_down_shifted_conv2d_deconv2d_layer2_stride2(self):
        self._test_down_shifted(
            num_layers=2, strides=[2, 2], layers=['conv', 'deconv'])

    def test_down_shifted_conv2d_conv2d_2_deconv_2_layer2(self):
        self._test_down_shifted(
            num_layers=3, strides=[1, 2, 2], layers=['conv', 'conv', 'deconv'])

    def test_down_shifted_conv2d_conv2d_deconv2d_deconv2d_layer4_stride2(self):
        self._test_down_shifted(
            num_layers=4,
            strides=[2, 2, 2, 2],
            layers=['conv', 'conv', 'deconv', 'deconv'])

    def test_down_shifted_conv2d_deconv2d_conv2d_deconv2d_layer4_stride2(self):
        self._test_down_shifted(
            num_layers=4,
            strides=[2, 2, 2, 2],
            layers=['conv', 'deconv', 'conv', 'deconv'])

    def test_down_right_shifted_conv2d_layer1_stride1_1by3(self):
        self._test_down_right_shifted(
            batch_size=1, size=8, num_layers=1, filter_size=[1, 3])

    def test_down_right_shifted_conv2d_layer1_stride1(self):
        self._test_down_right_shifted(batch_size=1, size=8, num_layers=1)

    def test_down_right_shifted_conv2d_layer2_stride1(self):
        self._test_down_right_shifted(batch_size=1, size=8, num_layers=2)

    def test_down_right_shifted_conv2d_layer3_stride1(self):
        self._test_down_right_shifted(batch_size=1, size=8, num_layers=3)

    def test_down_right_shifted_conv2d_layer1_stride2(self):
        self._test_down_right_shifted(
            batch_size=1, size=8, num_layers=1, strides=[2])

    def test_down_right_shifted_conv2d_layer2_stride2(self):
        self._test_down_right_shifted(
            batch_size=1, size=8, num_layers=2, strides=[2, 2])

    def test_down_right_shifted_conv2d_layer3_stride2(self):
        self._test_down_right_shifted(
            batch_size=1, size=16, num_layers=3, strides=[2, 1, 2])

    def test_down_right_shifted_deconv2d_layer1_stride2(self):
        self._test_down_right_shifted(
            batch_size=1, size=4, num_layers=1, layers=['deconv'], strides=[2])

    def test_down_right_shifted_deconv2d_layer2_stride2(self):
        self._test_down_right_shifted(
            batch_size=1,
            size=4,
            num_layers=2,
            layers=['deconv', 'deconv'],
            strides=[2, 2])

    def test_down_right_shifted_deconv2d_layer3_stride2(self):
        self._test_down_right_shifted(
            batch_size=1,
            size=4,
            num_layers=3,
            layers=['deconv', 'deconv', 'deconv'],
            strides=[2, 2, 2])

    def test_down_right_shifted_conv2d_deconv2d_layer2_stride2(self):
        self._test_down_right_shifted(
            num_layers=2, strides=[2, 2], layers=['conv', 'deconv'])

    def test_down_right_shifted_conv2d_conv2d_2_deconv_2_layer2(self):
        self._test_down_right_shifted(
            num_layers=3, strides=[1, 2, 2], layers=['conv', 'conv', 'deconv'])

    def test_down_right_shifted_conv2d_conv2d_deconv2d_deconv2d_layer4_stride2(
            self):
        self._test_down_right_shifted(
            num_layers=4,
            strides=[2, 2, 2, 2],
            layers=['conv', 'conv', 'deconv', 'deconv'])

    def test_down_right_shifted_conv2d_deconv2d_conv2d_deconv2d_layer4_stride2(
            self):
        self._test_down_right_shifted(
            num_layers=4,
            strides=[2, 2, 2, 2],
            layers=['conv', 'deconv', 'conv', 'deconv'])

    def test_sum_rightshift_downshift(self):
        self._test_sum_rightshift_downshift(size=32)

    def test_gated_resnet_vstack_only_basic(self):
        self._gated_resnet_vstack_only()

    def test_gated_resnet_vstack_only_use_h(self):
        self._gated_resnet_vstack_only(use_h=True)

    def test_gated_resnet_vstack_only_basic_3layers(self):
        self._gated_resnet_vstack_only(num_layers=3)

    def test_gated_resnet_vstack_only_use_h_3layers(self):
        self._gated_resnet_vstack_only(use_h=True, num_layers=3)

    def test_gated_resnet_vstack_only_use_extra_row_input(self):
        self._gated_resnet_vstack_only(use_extra_row_input=True)

    def test_gated_resnet_vstack_only_use_extra_row_input_3layers(self):
        self._gated_resnet_vstack_only(use_extra_row_input=True, num_layers=3)

    def test_gated_resnet_vstack_only_use_extra_row_input_and_use_h(self):
        self._gated_resnet_vstack_only(use_extra_row_input=True, use_h=True)

    def test_gated_resnet_vstack_only_use_extra_row_input__and_use_h_3layers(
            self):
        self._gated_resnet_vstack_only(
            use_extra_row_input=True, use_h=True, num_layers=3)

    def test_gated_resnet_hstack_basic(self):
        self._gated_resnet_hstack()

    def test_gated_resnet_hstack_use_h(self):
        self._gated_resnet_hstack(use_h=True)

    def test_gated_resnet_hstack_basic_3layers(self):
        self._gated_resnet_hstack(num_layers=3)

    def test_gated_resnet_hstack_use_h_3layers(self):
        self._gated_resnet_hstack(use_h=True, num_layers=3)

    def test_gated_resnet_hstack_use_extra_pixel_input(self):
        self._gated_resnet_hstack(use_extra_pixel_input=True)

    def test_gated_resnet_hstack_use_extra_pixel_input_3layers(self):
        self._gated_resnet_hstack(use_extra_pixel_input=True, num_layers=3)

    def test_gated_resnet_hstack_use_extra_pixel_input_and_use_h(self):
        self._gated_resnet_hstack(use_extra_pixel_input=True, use_h=True)

    def test_gated_resnet_hstack_use_extra_pixel_input_and_use_h_3layers(self):
        self._gated_resnet_hstack(
            use_extra_pixel_input=True, use_h=True, num_layers=3)

    def _get_placeholders(self, image_size):
        '''Creates all placeholders.'''
        batch_size, size, _, input_channels = image_size
        full_input = tf.placeholder(
            tf.float32, [batch_size, size, size, input_channels],
            name='full_input')
        pixel_input = tf.placeholder(
            tf.float32, [batch_size, 1, 1, input_channels], name='pixel_input')
        row_input = tf.placeholder(
            tf.float32, [batch_size, 1, size, input_channels],
            name='row_input')
        row_id = tf.placeholder(tf.int32, [], name='row_id')
        col_id = tf.placeholder(tf.int32, [], name='col_id')
        return Placeholders(full_input, pixel_input, row_input, row_id, col_id)

    def _setup_test_equal(self, sess, nn_out, full_input, image_size,
                          output_image_size):
        '''Sets up both _test_*_equals() methods by initializing variables and outputs.'''
        np.random.seed(2702)
        x = np.random.randn(*image_size)
        # nn layers use data dependent initialization.
        # Data dependent initialization requires a batch of initial data,
        # which we pass through with a feed dict.
        sess.run(tf.global_variables_initializer(), {full_input: x})

        # Calculate ground truth output.
        ground_truth_output = sess.run(nn_out, {full_input: x})

        # Create variable that holds output.
        if output_image_size is None:
            output_image_size = image_size
        fast_output = np.zeros(output_image_size)

        # Calculate the increase in output size compared to the input size.
        # This is useful when only deconv (upsampling) layers are used.
        side_length = image_size[2]
        width_ratio = output_image_size[2] // image_size[2]
        image_increase_factor = max(1, width_ratio)

        # Reset the cache to be safe.
        sess.run(fast_nn.reset_cache_op())

        return x, ground_truth_output, fast_output, side_length, image_increase_factor

    def _test_rows_equal(self,
                         sess,
                         fast_nn_out,
                         nn_out,
                         placeholders,
                         image_size,
                         output_image_size=None,
                         run_every=1):
        '''Tests if vertical stack outputs (one row at a time) of our code and OpenAI code are equal.'''
        (x, ground_truth_output, fast_output, side_length,
         image_increase_factor) = self._setup_test_equal(
             sess, nn_out, placeholders.full_input, image_size,
             output_image_size)

        # Generate fast output.
        for row in range(side_length):
            x_row_input = x[:, row:(row + 1), :, :]
            # image_increase_factor is relevant when only deconvs are used.
            # It just runs each row of input multiple times to populate the upsampled output.
            for inner_iteration in range(image_increase_factor):
                row_compensated = image_increase_factor * row + inner_iteration
                feed_dict = {
                    placeholders.row_input: x_row_input,
                    placeholders.row_id: row_compensated
                }
                row_output = sess.run(fast_nn_out, feed_dict)

                if row_compensated % run_every == 0:
                    # The run_every division is for downsampling,
                    # because the output is smaller than the input.
                    output_row = row_compensated // run_every
                    fast_output[:, output_row:(output_row + 1),
                                                :, :] = row_output

        # Within a tolerance.
        self.assertTrue(np.allclose(ground_truth_output, fast_output))
        # Exact match.
        self.assertTrue(
            np.max(np.abs(ground_truth_output - fast_output)) == 0.0)

    def _test_pixels_equal(self,
                           sess,
                           fast_nn_out,
                           nn_out,
                           placeholders,
                           image_size,
                           output_image_size=None,
                           run_every=1,
                           atol=1e-6):
        '''Tests if horizontal stack outputs (one pixel at a time) of our code and OpenAI code are equal.'''
        (x, ground_truth_output, fast_output, side_length,
         image_increase_factor) = self._setup_test_equal(
             sess, nn_out, placeholders.full_input, image_size,
             output_image_size)

        # Generate fast output.
        for row in range(side_length):
            # image_increase_factor is relevant when only deconvs are used.
            # It just runs each row and column of input multiple times to populate the upsampled output.
            for inner_row_iteration in range(image_increase_factor):
                row_compensated = image_increase_factor * row + inner_row_iteration
                x_row_input = x[:, row:(row + 1), :, :]
                for col in range(side_length):
                    x_pixel_input = x[:, row:(row + 1), col:(col + 1), :]
                    for inner_col_iteration in range(image_increase_factor):
                        col_compensated = image_increase_factor * col + inner_col_iteration
                        feed_dict = {
                            placeholders.pixel_input: x_pixel_input,
                            placeholders.row_id: row_compensated,
                            placeholders.col_id: col_compensated,
                            placeholders.row_input: x_row_input
                        }

                        pixel_output = sess.run(fast_nn_out, feed_dict)

                        # The run_every division is for downsampling,
                        # because the output is smaller than the input.
                        if row_compensated % run_every == 0 and col_compensated % run_every == 0:
                            output_row = row_compensated // run_every
                            output_col = col_compensated // run_every
                            fast_output[:, output_row:(output_row + 1),
                                        output_col:(output_col + 1
                                                    ), :] = pixel_output

        self.assertTrue(
            np.allclose(ground_truth_output, fast_output, atol=atol))

    def _setup_conv_tests(self, batch_size, size, channels, filter_size,
                          strides, layers, num_layers):
        '''Sets up the conv tests by computing basic layer information.'''
        image_size = (batch_size, size, size, channels)
        full_filter_size = filter_size + [channels]
        if strides is None:
            strides = [1 for _ in range(num_layers)]
        assert len(strides) == num_layers
        if layers is None:
            layers = ['conv' for _ in range(num_layers)]
        assert len(layers) == num_layers
        return image_size, full_filter_size, strides, layers

    def _compute_conv_fast_nn_out(self, compute_output_func, network_input,
                                  image_size, strides, layers):
        '''Computes cached convolutions, handling downsampling and upsampling.'''
        batch_size, size, _, nr_filters = image_size
        num_layers = len(layers)

        # Computes the final output size taking into account downsampling and upsampling.
        output_size = size
        for stride, layer_type in zip(strides, layers):
            if layer_type == 'conv':
                output_size = output_size // stride
            else:
                output_size = output_size * stride
        output_image_size = (batch_size, output_size, output_size, nr_filters)

        # When running only deconvs, the output size gets bigger than the input size.
        # For generation, each input must be run multiple times to populate the output.
        image_increase_factor = max(output_size // size, 1)
        cumulative_stride = max(1, image_increase_factor)

        fast_nn_out = network_input
        counters = {}
        layer_input_size = size

        # Run the network.
        for layer_num in range(num_layers):
            stride = strides[layer_num]
            layer_type = layers[layer_num]

            # The run_every of one layer is the cache_every of the next layer.
            # These increase after downsampling since fewer inputs correspond to an output.
            # These decrease after downsampling since more inputs correspond to an output.
            cache_every = cumulative_stride
            if layer_type == 'conv':
                run_every = cumulative_stride * stride
            else:
                run_every = max(1, cumulative_stride // stride)

            input_image_size = (batch_size, layer_input_size, layer_input_size,
                                nr_filters)
            cumulative_stride = run_every

            fast_nn_out = compute_output_func(fast_nn_out, layer_type,
                                              input_image_size, stride,
                                              cache_every, run_every, counters)

            # The size of the input to the next layer.
            if layer_type == 'conv':
                layer_input_size = layer_input_size // stride  # Downsampling.
            else:
                layer_input_size = layer_input_size * stride  # Upsampling.

        return fast_nn_out, output_image_size, run_every

    def _test_down_shifted(self,
                           batch_size=10,
                           size=16,
                           channels=7,
                           num_layers=1,
                           filter_size=[2, 3],
                           strides=None,
                           layers=None,
                           nonlinearity=tf.sigmoid):
        '''Tests the down_shifted convolution for the vertical stack.'''

        def get_conv_function(module, layer_type):
            '''Returns the matching conv or deconv function.'''
            if layer_type == 'conv':
                return module.down_shifted_conv2d
            elif layer_type == 'deconv':
                return module.down_shifted_deconv2d
            else:
                raise ValueError('Unknown layer_type %s' % layer_type)

        image_size, full_filter_size, strides, layers = self._setup_conv_tests(
            batch_size, size, channels, filter_size, strides, layers,
            num_layers)

        with self.test_session() as sess:
            placeholders = self._get_placeholders(image_size)

            # OpenAI output.
            def compute_ground_truth(init):
                nn_out = placeholders.full_input
                counters = {}
                for layer_num in range(num_layers):
                    stride = strides[layer_num]
                    layer_func = get_conv_function(nn, layers[layer_num])
                    nn_out = layer_func(
                        nn_out,
                        num_filters=channels,
                        filter_size=filter_size,
                        stride=[stride, stride],
                        nonlinearity=nonlinearity,
                        counters=counters,
                        init=init)
                return nn_out

            compute_ground_truth(init=True)
            tf.get_variable_scope().reuse_variables()
            nn_out = compute_ground_truth(init=False)

            # Our output.
            def compute_output_func(fast_nn_out, layer_type, input_image_size,
                                    stride, cache_every, run_every, counters):
                layer_func = get_conv_function(fast_nn, layer_type)
                return layer_func(
                    fast_nn_out,
                    network_info=(input_image_size, full_filter_size),
                    stride=stride,
                    row=placeholders.row_id,
                    cache_every=cache_every,
                    run_every=run_every,
                    counters=counters,
                    nonlinearity=nonlinearity)

            fast_nn_out, output_image_size, run_every = self._compute_conv_fast_nn_out(
                compute_output_func, placeholders.row_input, image_size,
                strides, layers)

            self._test_rows_equal(
                sess,
                fast_nn_out,
                nn_out,
                placeholders,
                image_size,
                output_image_size=output_image_size,
                run_every=run_every)

    def _test_down_right_shifted(self,
                                 batch_size=10,
                                 size=16,
                                 channels=7,
                                 num_layers=1,
                                 filter_size=[2, 2],
                                 strides=None,
                                 layers=None,
                                 nonlinearity=tf.sigmoid):
        '''Tests the down_shifted convolution for the vertical stack.'''

        def get_conv_function(module, layer_type):
            '''Returns the matching conv or deconv function.'''
            if layer_type == 'conv':
                return module.down_right_shifted_conv2d
            elif layer_type == 'deconv':
                return module.down_right_shifted_deconv2d
            else:
                raise ValueError('Unknown layer_type %s' % layer_type)

        image_size, full_filter_size, strides, layers = self._setup_conv_tests(
            batch_size, size, channels, filter_size, strides, layers,
            num_layers)

        with self.test_session() as sess:

            placeholders = self._get_placeholders(image_size)

            # OpenAI output.
            def compute_ground_truth(init):
                nn_out = placeholders.full_input
                counters = {}
                for layer_num in range(num_layers):
                    stride = strides[layer_num]
                    layer_func = get_conv_function(nn, layers[layer_num])
                    nn_out = layer_func(
                        nn_out,
                        num_filters=channels,
                        filter_size=filter_size,
                        stride=[stride, stride],
                        nonlinearity=nonlinearity,
                        counters=counters,
                        init=init)
                return nn_out

            compute_ground_truth(init=True)
            tf.get_variable_scope().reuse_variables()
            nn_out = compute_ground_truth(init=False)

            # Our output.
            def compute_output_func(fast_nn_out, layer_type, input_image_size,
                                    stride, cache_every, run_every, counters):
                layer_func = get_conv_function(fast_nn, layer_type)
                return layer_func(
                    fast_nn_out,
                    network_info=(input_image_size, full_filter_size),
                    row=placeholders.row_id,
                    col=placeholders.col_id,
                    cache_every=cache_every,
                    run_every=run_every,
                    counters=counters,
                    nonlinearity=nonlinearity)

            fast_nn_out, output_image_size, run_every = self._compute_conv_fast_nn_out(
                compute_output_func, placeholders.pixel_input, image_size,
                strides, layers)

            self._test_pixels_equal(
                sess,
                fast_nn_out,
                nn_out,
                placeholders,
                image_size,
                output_image_size=output_image_size,
                run_every=run_every)

    def _gated_resnet_vstack_only(self,
                                  batch_size=10,
                                  size=16,
                                  channels=7,
                                  num_layers=1,
                                  filter_size=[2, 3],
                                  use_h=False,
                                  use_extra_row_input=False,
                                  nonlinearity=tf.sigmoid):
        '''Tests the gated resnet layers for the vertical stack.'''
        image_size = (batch_size, size, size, channels)
        full_filter_size = filter_size + [channels]

        np.random.seed(2702)
        with self.test_session() as sess:
            placeholders = self._get_placeholders(image_size)

            # Conditional information and skip connections.
            h, a = None, None
            if use_h:
                h = tf.constant(
                    np.random.randn(batch_size, 20), dtype=tf.float32)
            if use_extra_row_input:
                a = placeholders.full_input

            # OpenAI output.
            def compute_ground_truth(init):
                counters = {}
                nn_out = placeholders.full_input
                for _ in range(num_layers):
                    nn_out = nn.gated_resnet(
                        nn_out,
                        a=a,
                        h=h,
                        conv=nn.down_shifted_conv2d,
                        nonlinearity=nonlinearity,
                        counters=counters,
                        init=init)
                return nn_out

            compute_ground_truth(init=True)
            tf.get_variable_scope().reuse_variables()
            nn_out = compute_ground_truth(init=False)

            # Our output.
            counters = {}
            fast_nn_out = placeholders.row_input
            if use_extra_row_input:
                a = placeholders.row_input
            for _ in range(num_layers):
                fast_nn_out = fast_nn.gated_resnet_vstack_only(
                    fast_nn_out, (image_size, full_filter_size),
                    placeholders.row_id,
                    extra_row_input=a,
                    h=h,
                    cache_every=1,
                    run_every=1,
                    nonlinearity=nonlinearity,
                    counters=counters)

        self._test_rows_equal(sess, fast_nn_out, nn_out, placeholders,
                              image_size)

    def _gated_resnet_hstack(self,
                             batch_size=10,
                             size=16,
                             channels=7,
                             filter_size=[2, 2],
                             num_layers=1,
                             use_h=False,
                             use_extra_pixel_input=False,
                             nonlinearity=tf.sigmoid):
        '''Tests the gated resnet layers for the horizontal stack.'''
        image_size = (batch_size, size, size, channels)
        full_filter_size = filter_size + [channels]

        with self.test_session() as sess:
            placeholders = self._get_placeholders(image_size)

            # Conditional information and skip connections.
            h, a = None, placeholders.full_input
            if use_h:
                h = tf.constant(
                    np.random.randn(batch_size, 20), dtype=tf.float32)
            if use_extra_pixel_input:
                a = tf.concat([a, 2 * placeholders.full_input], 3)

            # OpenAI output.
            def compute_ground_truth(init):
                counters = {}
                nn_out = placeholders.full_input
                for _ in range(num_layers):
                    nn_out = nn.gated_resnet(
                        nn_out,
                        a=a,
                        h=h,
                        conv=nn.down_right_shifted_conv2d,
                        nonlinearity=nonlinearity,
                        counters=counters,
                        init=init)
                return nn_out

            compute_ground_truth(init=True)
            tf.get_variable_scope().reuse_variables()
            nn_out = compute_ground_truth(init=False)

            # Our output.
            extra_pixel_input = None
            if use_extra_pixel_input:
                extra_pixel_input = 2 * placeholders.pixel_input

            counters = {}
            fast_nn_out = placeholders.pixel_input
            for _ in range(num_layers):
                fast_nn_out = fast_nn.gated_resnet_hstack(
                    fast_nn_out,
                    placeholders.row_input, (image_size, full_filter_size),
                    h=h,
                    row=placeholders.row_id,
                    col=placeholders.col_id,
                    cache_every=1,
                    run_every=1,
                    extra_pixel_input=extra_pixel_input,
                    nonlinearity=nonlinearity,
                    counters=counters)

        self._test_pixels_equal(sess, fast_nn_out, nn_out, placeholders,
                                image_size)

    def _test_sum_rightshift_downshift(self,
                                       batch_size=10,
                                       size=16,
                                       channels=7,
                                       nonlinearity=tf.sigmoid):
        '''Tests the sum of the vertical and horizontal stack.'''
        image_size = (batch_size, size, size, channels)

        with self.test_session() as sess:
            placeholders = self._get_placeholders(image_size)

            # OpenAI output.
            def compute_ground_truth(init):
                counters = {}
                nn_v_stack = nn.down_shifted_conv2d(
                    placeholders.full_input,
                    num_filters=channels,
                    filter_size=[1, 3],
                    stride=[1, 1],
                    nonlinearity=nonlinearity,
                    counters=counters,
                    init=init)
                nn_h_stack = nn.down_right_shifted_conv2d(
                    placeholders.full_input,
                    num_filters=channels,
                    filter_size=[2, 1],
                    stride=[1, 1],
                    nonlinearity=nonlinearity,
                    counters=counters,
                    init=init)
                return nn_v_stack + nn_h_stack

            compute_ground_truth(init=True)
            tf.get_variable_scope().reuse_variables()
            nn_out = compute_ground_truth(init=False)

            # Our output
            counters, stride, cache_every, run_every = {}, 1, 1, 1
            fast_nn_v_stack = fast_nn.down_shifted_conv2d(
                placeholders.row_input,
                network_info=(image_size, [1, 3, channels]),
                stride=stride,
                row=placeholders.row_id,
                cache_every=cache_every,
                run_every=run_every,
                counters=counters,
                nonlinearity=nonlinearity)
            fast_nn_h_stack = fast_nn.down_right_shifted_conv2d(
                placeholders.pixel_input,
                network_info=(image_size, [2, 1, channels]),
                row=placeholders.row_id,
                col=placeholders.col_id,
                cache_every=cache_every,
                run_every=run_every,
                counters=counters,
                nonlinearity=nonlinearity)
            fast_nn_out = fast_nn.sum_rightshift_downshift(
                fast_nn_h_stack, fast_nn_v_stack, placeholders.col_id)

        self._test_pixels_equal(sess, fast_nn_out, nn_out, placeholders,
                                image_size)
