from . import fast_nn
from . import nn

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import numpy as np

UPDATE_V_STACK = 'update_v_stack'


def undo_zeroth_row_bias_when_downshifting(row_output, row):
    '''The down_shifted_conv2d adds a bias to the row of all zeros. This removes that bias.'''
    return tf.cond(
        tf.equal(row, 0), lambda: tf.zeros_like(row_output),
        lambda: row_output)


def undo_zeroth_column_bias_when_rightshifting(pixel_output, col):
    '''The down_shifted_conv2d adds a bias to the column of all zeros. This removes that bias.'''
    return tf.cond(
        tf.equal(col, 0), lambda: tf.zeros_like(pixel_output),
        lambda: pixel_output)


def cache_v_stack_variable(v_stack_variable):
    '''Caches vertical stack hidden states. This avoids the need to pass the computed
        vertical stack in the feed_dict, which would involve CPU to GPU transfers.'''
    cache = tf.Variable(
        initial_value=np.zeros(v_stack_variable.get_shape().as_list()),
        name='v_stack_cache',
        dtype=tf.float32)
    update_v_stack_cache = cache.assign(v_stack_variable)
    tf.add_to_collection(UPDATE_V_STACK, update_v_stack_cache)
    reset_cache = cache.assign(tf.zeros_like(cache))
    tf.add_to_collection(fast_nn.RESET_CACHE_COLLECTION, reset_cache)
    return cache


def model_spec(row_input,
               pixel_input,
               row,
               col,
               image_size,
               h=None,
               nr_resnet=5,
               nr_filters=160,
               nr_logistic_mix=10,
               resnet_nonlinearity='concat_elu',
               seed=None):
    '''Creates the model. Follows the same model_spec structure as the original PixelCNN++.'''
    counters = {}
    with arg_scope(
        [
            fast_nn.down_shifted_conv2d, fast_nn.down_right_shifted_conv2d,
            fast_nn.down_shifted_deconv2d, fast_nn.down_right_shifted_deconv2d,
            fast_nn.gated_resnet_vstack_only, fast_nn.gated_resnet_hstack,
            nn.dense
        ],
            counters=counters):

        # Parse resnet nonlinearity argument.
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise ('resnet nonlinearity ' + resnet_nonlinearity +
                   ' is not supported')

        with arg_scope(
            [fast_nn.gated_resnet_vstack_only, fast_nn.gated_resnet_hstack],
                nonlinearity=resnet_nonlinearity,
                h=h):

            u_filter = [2, 3, nr_filters]
            ul_filter = [2, 2, nr_filters]
            cache_every, run_every = 1, 1

            ## Downsampling pass.

            # The initial computation to the network. Importantly, it is assumed that the
            # vertical stack inputs are already downshifted, and the horizontal stack inputs
            # are already rightshifted. 
            v_stack = []
            u_list_input = fast_nn.down_shifted_conv2d(
                row_input, (image_size, u_filter),
                stride=1,
                row=row,
                cache_every=cache_every,
                run_every=run_every)
            u_list = [
                undo_zeroth_row_bias_when_downshifting(u_list_input, row)
            ]
            v_stack.append(u_list[-1])

            downshift_hstack_input = fast_nn.down_shifted_conv2d(
                row_input, (image_size, [1, 3, nr_filters]),
                stride=1,
                row=row,
                cache_every=cache_every,
                run_every=run_every)
            downshift_hstack_input = undo_zeroth_row_bias_when_downshifting(
                downshift_hstack_input, row)
            downshift_hstack_input = cache_v_stack_variable(
                downshift_hstack_input)
            v_stack.append(downshift_hstack_input)
            rightshift_hstack_input = fast_nn.down_right_shifted_conv2d(
                pixel_input, (image_size, [2, 1, nr_filters]),
                row=row,
                col=col,
                cache_every=cache_every,
                run_every=run_every)
            rightshift_hstack_input = undo_zeroth_column_bias_when_rightshifting(
                rightshift_hstack_input, col)
            ul_list = [
                fast_nn.sum_rightshift_downshift(rightshift_hstack_input,
                                                 downshift_hstack_input, col)
            ]

            # Gated resnet layers.
            image_size = (image_size[0], image_size[1], image_size[2],
                          nr_filters)
            for rep in range(nr_resnet):
                u_list.append(
                    fast_nn.gated_resnet_vstack_only(
                        u_list[-1], (image_size, u_filter),
                        row=row,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))
                v_stack.append(u_list[-1])
                ul_list.append(
                    fast_nn.gated_resnet_hstack(
                        ul_list[-1],
                        cache_v_stack_variable(u_list[-1]), (image_size,
                                                             ul_filter),
                        row=row,
                        col=col,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))

            # Downsample.
            cache_every, run_every = 1, 2
            u_list.append(
                fast_nn.down_shifted_conv2d(
                    u_list[-1], (image_size, u_filter),
                    stride=2,
                    row=row,
                    cache_every=cache_every,
                    run_every=run_every))
            v_stack.append(u_list[-1])
            ul_list.append(
                fast_nn.down_right_shifted_conv2d(
                    ul_list[-1], (image_size, ul_filter),
                    row=row,
                    col=col,
                    cache_every=cache_every,
                    run_every=run_every))

            cache_every, run_every = 2, 2
            image_size = (image_size[0], image_size[1] // 2,
                          image_size[2] // 2, nr_filters)

            # Gated resnet layers.
            for rep in range(nr_resnet):
                u_list.append(
                    fast_nn.gated_resnet_vstack_only(
                        u_list[-1], (image_size, u_filter),
                        row=row,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))
                v_stack.append(u_list[-1])
                ul_list.append(
                    fast_nn.gated_resnet_hstack(
                        ul_list[-1],
                        cache_v_stack_variable(u_list[-1]), (image_size,
                                                             ul_filter),
                        row=row,
                        col=col,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))

            # Downsample.
            cache_every, run_every = 2, 4
            u_list.append(
                fast_nn.down_shifted_conv2d(
                    u_list[-1], (image_size, u_filter),
                    stride=2,
                    row=row,
                    cache_every=cache_every,
                    run_every=run_every))
            v_stack.append(u_list[-1])
            ul_list.append(
                fast_nn.down_right_shifted_conv2d(
                    ul_list[-1], (image_size, ul_filter),
                    row=row,
                    col=col,
                    cache_every=cache_every,
                    run_every=run_every))

            cache_every, run_every = 4, 4
            image_size = (image_size[0], image_size[1] // 2,
                          image_size[2] // 2, nr_filters)

            # Gated resnet layers.
            for rep in range(nr_resnet):
                u_list.append(
                    fast_nn.gated_resnet_vstack_only(
                        u_list[-1], (image_size, u_filter),
                        row=row,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))
                v_stack.append(u_list[-1])
                ul_list.append(
                    fast_nn.gated_resnet_hstack(
                        ul_list[-1],
                        cache_v_stack_variable(u_list[-1]), (image_size,
                                                             ul_filter),
                        row=row,
                        col=col,
                        cache_every=cache_every,
                        run_every=run_every,
                        nonlinearity=resnet_nonlinearity))

            # Upsampling pass.
            u = u_list.pop()
            ul = ul_list.pop()
            for rep in range(nr_resnet):
                u = fast_nn.gated_resnet_vstack_only(
                    u, (image_size, u_filter),
                    extra_row_input=u_list.pop(),
                    row=row,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)
                v_stack.append(u)
                ul = fast_nn.gated_resnet_hstack(
                    ul,
                    cache_v_stack_variable(u), (image_size, ul_filter),
                    extra_pixel_input=ul_list.pop(),
                    row=row,
                    col=col,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)

            # Upsample.
            cache_every, run_every = 4, 2
            u = fast_nn.down_shifted_deconv2d(
                u, (image_size, u_filter),
                stride=2,
                row=row,
                cache_every=cache_every,
                run_every=run_every)
            v_stack.append(u)
            ul = fast_nn.down_right_shifted_deconv2d(
                ul, (image_size, ul_filter),
                row=row,
                col=col,
                cache_every=cache_every,
                run_every=run_every)

            cache_every, run_every = 2, 2
            image_size = (image_size[0], image_size[1] * 2, image_size[2] * 2,
                          nr_filters)

            # Gated resnet layers.
            for rep in range(nr_resnet + 1):
                u = fast_nn.gated_resnet_vstack_only(
                    u, (image_size, u_filter),
                    extra_row_input=u_list.pop(),
                    row=row,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)
                v_stack.append(u)
                ul = fast_nn.gated_resnet_hstack(
                    ul,
                    cache_v_stack_variable(u), (image_size, ul_filter),
                    extra_pixel_input=ul_list.pop(),
                    row=row,
                    col=col,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)

            # Upsample.    
            cache_every, run_every = 2, 1
            u = fast_nn.down_shifted_deconv2d(
                u, (image_size, u_filter),
                stride=2,
                row=row,
                cache_every=cache_every,
                run_every=run_every)
            v_stack.append(u)
            ul = fast_nn.down_right_shifted_deconv2d(
                ul, (image_size, ul_filter),
                row=row,
                col=col,
                cache_every=cache_every,
                run_every=run_every)

            cache_every, run_every = 1, 1
            image_size = (image_size[0], image_size[1] * 2, image_size[2] * 2,
                          nr_filters)

            # Gated resnet layers.
            for rep in range(nr_resnet + 1):
                u = fast_nn.gated_resnet_vstack_only(
                    u, (image_size, u_filter),
                    extra_row_input=u_list.pop(),
                    row=row,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)
                v_stack.append(u)
                ul = fast_nn.gated_resnet_hstack(
                    ul,
                    cache_v_stack_variable(u), (image_size, ul_filter),
                    extra_pixel_input=ul_list.pop(),
                    row=row,
                    col=col,
                    cache_every=cache_every,
                    run_every=run_every,
                    nonlinearity=resnet_nonlinearity)

            assert len(u_list) == 0
            assert len(ul_list) == 0

            x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)
            sample = nn.sample_from_discretized_mix_logistic(
                x_out, nr_logistic_mix, seed=seed)
            cache_v_stack = tf.group(*tf.get_collection(UPDATE_V_STACK))

            return sample, x_out, cache_v_stack
