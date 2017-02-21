from . import nn

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
import numpy as np

from collections import namedtuple
import math

LayerInfo = namedtuple('LayerInfo', [
    'image_size', 'batch', 'image_height', 'image_width', 'image_channels',
    'filter_size', 'filter_height', 'filter_width', 'filter_channels',
    'input_channels', 'nonlinearity'
])

RESET_CACHE_COLLECTION = 'reset_cache'


def down_shift(image):
    '''Shift all rows down by one, using zeros as the first row and throwing away the last row.'''
    all_image_except_last_row = image[:, :-1, :, :]
    zero_row = np.zeros_like(image[:, :1, :, :])
    return np.concatenate([zero_row, all_image_except_last_row], axis=1)


def right_shift(image):
    '''Shift all columns right by one, using zeros as the first column and throwing away the last column.'''
    all_image_except_last_column = image[:, :, :-1, :]
    zero_column = np.zeros_like(image[:, :, :1, :])
    return np.concatenate([zero_column, all_image_except_last_column], axis=2)


def _extract_layer_info(network_info, input_, nonlinearity):
    '''Utility function to extract information about the current layer.'''
    image_size, filter_size = network_info
    batch, image_height, image_width, image_channels = image_size
    filter_height, filter_width, filter_channels = filter_size
    input_channels = int(input_.get_shape()[-1])
    if nonlinearity is None:
        nonlinearity = tf.identity
    return LayerInfo(image_size, batch, image_height, image_width,
                     image_channels, filter_size, filter_height, filter_width,
                     filter_channels, input_channels, nonlinearity)


def _create_cache(batch, cache_height, cache_width, channels):
    '''Creates a cache, which is used to avoid redundant computation.'''
    cache = tf.Variable(
        initial_value=np.zeros((batch, cache_height, cache_width, channels)),
        dtype=tf.float32,
        name='cache',
        trainable=False)
    # Reset the cache between generations.
    reset_cache = cache.assign(tf.zeros_like(cache))
    tf.add_to_collection(RESET_CACHE_COLLECTION, reset_cache)
    return cache


def reset_cache_op():
    '''Returns an op to reset all created caches. Used between different generation calls.'''
    return tf.group(*tf.get_collection(RESET_CACHE_COLLECTION))


def _get_conv_variables(filter_size, input_channels, scope_name, counters):
    '''Creates and returns variables used for convolution.'''
    filter_height, filter_width, filter_channels = filter_size
    with tf.variable_scope(nn.get_name(scope_name, counters)):
        V = tf.get_variable(
            'V',
            [filter_height, filter_width, input_channels, filter_channels],
            dtype=tf.float32)
        g = tf.get_variable('g', [filter_channels], dtype=tf.float32)
        b = tf.get_variable('b', [filter_channels], dtype=tf.float32)
    return V, g, b


def _get_conv2d_variables(filter_size, input_channels, counters):
    '''Creates and returns the variables used for a normal 2D convolution.'''
    V, g, b = _get_conv_variables(filter_size, input_channels, 'conv2d',
                                  counters)
    filter_channels = filter_size[-1]
    W = tf.reshape(g, [1, 1, 1, filter_channels]) * tf.nn.l2_normalize(
        V, [0, 1, 2])  # Weight normalization.
    return W, b


def _get_deconv2d_variables(filter_size, input_channels, counters):
    '''Creates and returns the variables used for a 2D transposed convolution (deconvolution).'''
    V, g, b = _get_conv_variables(filter_size, input_channels, 'deconv2d',
                                  counters)
    filter_channels = filter_size[-1]
    W = tf.reshape(g, [1, 1, filter_channels, 1]) * tf.nn.l2_normalize(
        V, [0, 1, 3])  # Weight normalization.
    return W, b


def _mod_equal_0(row_or_col, every):
    '''Returns a boolean tensor representing (row_or_col % every == 0)'''
    return tf.equal(tf.mod(row_or_col, every), 0)


def _roll_cache(cache):
    '''Pop off the oldest row of the cache to make space for the newest row of input.'''
    batch, _, cache_width, channels = cache.get_shape()
    without_dropped_row = cache[:, 1:, :, :]
    zero_row = tf.zeros([batch, 1, cache_width, channels])
    rolled_cache = tf.concat([without_dropped_row, zero_row], 1)
    return cache.assign(rolled_cache)


@add_arg_scope
def down_shifted_conv2d(row_input,
                        network_info,
                        stride,
                        row,
                        cache_every,
                        run_every,
                        nonlinearity=None,
                        counters={}):
    '''Performs a convolution for the vertical stack.'''
    li = _extract_layer_info(network_info, row_input, nonlinearity)

    ## Create cache.
    cache_height = li.filter_height  # Just large enough to fit the filter.
    padding = li.filter_width // 2  # Horizontal padding to make VALID convolution maintain the width of input. 
    cache_width = li.image_width + 2 * padding  # Cache width is the image width plus padding to the left and right.
    cache = _create_cache(li.batch, cache_height, cache_width,
                          li.input_channels)

    ## Update cache.
    should_cache = _mod_equal_0(row, cache_every)
    cache_func = lambda: cache[:, -1:, padding:(padding + li.image_width), :].assign(row_input)
    do_nothing_cache_func = lambda: row_input
    assign_to_cache = tf.cond(should_cache, cache_func, do_nothing_cache_func)

    ## Compute output.
    W, b = _get_conv2d_variables(li.filter_size, li.input_channels, counters)
    with tf.control_dependencies([assign_to_cache]):
        should_run = _mod_equal_0(row, run_every)

        # Compute output for the entire row.
        run_func = lambda: li.nonlinearity(tf.nn.conv2d(cache, W, [1, 1, stride, 1], 'VALID') + b)

        output_width = int(math.ceil(li.image_width / float(stride)))
        do_nothing_run_func = lambda: tf.zeros([li.batch, 1, output_width, li.filter_channels])

        outputs = tf.cond(should_run, run_func, do_nothing_run_func)
        outputs.set_shape([li.batch, 1, output_width, li.filter_channels])

        # Ensure that roll_cache() is run, and only after computing the outputs.
        with tf.control_dependencies([outputs]):
            roll_cache_op = tf.cond(should_cache, lambda: _roll_cache(cache),
                                    lambda: cache)
            with tf.control_dependencies([roll_cache_op]):
                outputs = tf.identity(outputs)

    return outputs


@add_arg_scope
def down_right_shifted_conv2d(pixel_input,
                              network_info,
                              row,
                              col,
                              cache_every,
                              run_every,
                              nonlinearity=None,
                              counters={}):
    '''Performs a convolution for the horizontal stack.'''
    li = _extract_layer_info(network_info, pixel_input, nonlinearity)

    ## Create cache.
    cache_height = li.filter_height  # Just large enough to fit the filter.
    left_pad = li.filter_width - 1  # Only need left padding because always convolving to the left.
    cache_width = li.image_width + left_pad
    cache = _create_cache(li.batch, cache_height, cache_width,
                          li.input_channels)
    cache_col = col // cache_every  # Accounts for downsampling due to stride in previous layers.

    ## Update cache.
    should_cache = tf.logical_and(
        _mod_equal_0(row, cache_every), _mod_equal_0(col, cache_every))

    pixel_col = cache_col + left_pad  # Accounts for padding in the cache.
    cache_func = lambda: cache[:, -1:, pixel_col:(pixel_col + 1), :].assign(pixel_input)

    do_nothing_cache_func = lambda: pixel_input

    assign_to_cache = tf.cond(should_cache, cache_func, do_nothing_cache_func)

    ## Compute output.
    W, b = _get_conv2d_variables(li.filter_size, li.input_channels, counters)
    with tf.control_dependencies([assign_to_cache]):
        should_run = tf.logical_and(
            _mod_equal_0(row, run_every), _mod_equal_0(col, run_every))

        # Extract the local neighborhood of the current column in the cache to be convolved with the filter.
        # This is simply a matrix multiply, since the neighborhood is the size of the filter.
        width_start = cache_col
        width_end = width_start + li.filter_width
        cache_neighborhood = cache[:, :, width_start:width_end, :]
        run_func = lambda: li.nonlinearity(tf.nn.conv2d(cache_neighborhood, W, [1, 1, 1, 1], 'VALID') + b)

        do_nothing_run_func = lambda: tf.zeros([li.batch, 1, 1, li.filter_channels])

        outputs = tf.cond(should_run, run_func, do_nothing_run_func)
        outputs.set_shape([li.batch, 1, 1, li.filter_channels])

        # Ensure that roll_cache() is run, and only after computing the outputs.
        with tf.control_dependencies([outputs]):
            # Roll out an entire row of the cache only after generating output for the last column.
            is_end_of_row = tf.equal(cache_col, li.image_width - 1)
            should_roll = tf.logical_and(should_cache, is_end_of_row)
            maybe_roll = tf.cond(should_roll, lambda: _roll_cache(cache),
                                 lambda: cache)
            with tf.control_dependencies([maybe_roll]):
                outputs = tf.identity(outputs)

    return outputs


def _create_deconv_cache(li, stride):
    '''Creates the cache for the two deconv layers.'''
    cache_height = li.filter_height  # Just large enough to fit the filter.
    # The deconv will increases the number of outputs `stride` times. 
    # The extra width comes from the tf.nn.conv2d_transpose() operation.
    cache_width = li.image_width * stride + li.filter_width - 1
    cache = _create_cache(li.batch, cache_height, cache_width,
                          li.filter_channels)
    return cache, cache_height, cache_width


@add_arg_scope
def down_shifted_deconv2d(row_input,
                          network_info,
                          row,
                          cache_every,
                          run_every,
                          stride=2,
                          nonlinearity=None,
                          counters={}):
    '''Performs a transposed convolution for the vertical stack.'''
    li = _extract_layer_info(network_info, row_input, nonlinearity)

    ## Create cache.
    cache, cache_height, cache_width = _create_deconv_cache(li, stride)

    ## Update cache.
    should_cache = _mod_equal_0(row, cache_every)

    W, b = _get_deconv2d_variables(li.filter_size, li.input_channels, counters)

    def cache_func():
        # Compute deconv output for the entire row.
        outputs = tf.nn.conv2d_transpose(
            row_input,
            W,
            output_shape=[
                li.batch, cache_height, cache_width, li.filter_channels
            ],
            strides=[1, stride, stride, 1],
            padding='VALID')
        outputs = li.nonlinearity(outputs + b)

        # Store the output in the cache.
        with tf.control_dependencies([outputs]):
            # With stride=2, this is simply cache.assign(outputs) since the old rows in the cache
            # will all have been rolled out. 
            update_cache = cache.assign(cache + outputs)
        return update_cache

    do_nothing_cache_func = lambda: tf.zeros_like(cache)

    assign_to_cache = tf.cond(should_cache, cache_func, do_nothing_cache_func)

    ## Compute output.
    with tf.control_dependencies([assign_to_cache]):
        should_run = _mod_equal_0(row, run_every)

        def run_func():
            # The cache stores the deconv output, so just return the next (first) row and roll.
            output = cache[:, 0:1, 1:-1, :]
            with tf.control_dependencies([output]):
                with tf.control_dependencies([_roll_cache(cache)]):
                    output = tf.identity(output)
            return output

        do_nothing_run_func = lambda: tf.zeros([li.batch, 1, cache_width - 2, li.filter_channels])

        outputs = tf.cond(should_run, run_func, do_nothing_run_func)
        outputs.set_shape([li.batch, 1, cache_width - 2, li.filter_channels])

    return outputs


@add_arg_scope
def down_right_shifted_deconv2d(pixel_input,
                                network_info,
                                row,
                                col,
                                cache_every,
                                run_every,
                                stride=2,
                                nonlinearity=None,
                                counters={}):
    '''Performs a transposed convolution for the horizontal stack.'''
    li = _extract_layer_info(network_info, pixel_input, nonlinearity)

    ## Create cache.
    cache, cache_height, cache_width = _create_deconv_cache(li, stride)

    ## Update cache.
    should_cache = tf.logical_and(
        _mod_equal_0(row, cache_every), _mod_equal_0(col, cache_every))

    W, b = _get_deconv2d_variables(li.filter_size, li.input_channels, counters)

    def cache_func():
        outputs = tf.nn.conv2d_transpose(
            pixel_input,
            W,
            output_shape=[
                li.batch, li.filter_height, li.filter_width, li.filter_channels
            ],
            strides=[1, stride, stride, 1],
            padding='VALID')
        outputs = li.nonlinearity(outputs + b)

        # Store the output in the cache.
        with tf.control_dependencies([outputs]):
            cache_col = col // cache_every
            update_cache = cache[:, :, (stride * cache_col):(stride * (
                cache_col + 1)), :].assign(outputs)
        return update_cache

    do_nothing_cache_func = lambda: tf.zeros([li.batch, li.filter_height, li.filter_width, li.filter_channels])

    assign_to_cache = tf.cond(should_cache, cache_func, do_nothing_cache_func)

    ## Compute output.
    with tf.control_dependencies([assign_to_cache]):
        should_run = tf.logical_and(
            _mod_equal_0(row, run_every), _mod_equal_0(col, run_every))

        def run_func():
            output_col = col // run_every
            output = cache[:, 0:1, output_col:(output_col + 1), :]

            # Only roll after the end of the row has been reached.
            with tf.control_dependencies([output]):
                is_end_of_row = tf.equal(output_col,
                                         cache_width - li.filter_width)
                maybe_roll = tf.cond(is_end_of_row, lambda: _roll_cache(cache),
                                     lambda: cache)
                with tf.control_dependencies([maybe_roll]):
                    output = tf.identity(output)
            return output

        do_nothing_run_func = lambda: tf.zeros([li.batch, 1, 1, li.filter_channels])

        outputs = tf.cond(should_run, run_func, do_nothing_run_func)
        outputs.set_shape([li.batch, 1, 1, li.filter_channels])

    return outputs


def sum_rightshift_downshift(rightshifted_pixel, downshifted_row, col):
    '''Sums the vertical and horizontal stack.'''
    downshifted_pixel = downshifted_row[:, :, col:(col + 1), :]
    return rightshifted_pixel + downshifted_pixel


def _conditional_info(h, batch, filter_channels, counters):
    '''Computes the conditional information for the resnet layer.'''
    with tf.variable_scope(nn.get_name('conditional_weights', counters)):
        hw = tf.get_variable(
            'hw',
            shape=[h.get_shape()[-1], 2 * filter_channels],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.05),
            trainable=True)
        conditional_info = tf.reshape(
            tf.matmul(h, hw), [batch, 1, 1, 2 * filter_channels])
        return conditional_info


def _gated_nonlinearity(out):
    a, b = tf.split(out, 2, 3)
    return a * tf.nn.sigmoid(b)


@add_arg_scope
def gated_resnet_vstack_only(row_input,
                             network_info,
                             row,
                             cache_every,
                             run_every,
                             extra_row_input=None,
                             h=None,
                             nonlinearity=None,
                             counters={}):
    '''Performs gated resnet computations for the vertical stack.'''
    li = _extract_layer_info(network_info, row_input, nonlinearity)

    out = li.nonlinearity(row_input)
    out = down_shifted_conv2d(
        out,
        network_info,
        stride=1,
        row=row,
        cache_every=cache_every,
        run_every=run_every,
        nonlinearity=None,
        counters=counters)
    if extra_row_input is not None:
        # For skip connections between downsampling and upsampling layers.
        out += nn.nin(
            li.nonlinearity(extra_row_input),
            li.filter_channels,
            counters=counters)

    out = li.nonlinearity(out)
    network_info = (li.image_size, (li.filter_height, li.filter_width, 2 *
                                    li.filter_channels))
    out = down_shifted_conv2d(
        out,
        network_info,
        stride=1,
        row=row,
        cache_every=cache_every,
        run_every=run_every,
        nonlinearity=None,
        counters=counters)

    if h is not None:
        out += _conditional_info(h, li.batch, li.filter_channels, counters)

    out = row_input + _gated_nonlinearity(out)
    return out


@add_arg_scope
def gated_resnet_hstack(pixel_input,
                        v_stack_row_input,
                        network_info,
                        row,
                        col,
                        cache_every,
                        run_every,
                        extra_pixel_input=None,
                        h=None,
                        nonlinearity=None,
                        counters={}):
    '''Performs gated resnet computations for the horizontal stack.'''
    li = _extract_layer_info(network_info, pixel_input, nonlinearity)

    out = li.nonlinearity(pixel_input)
    out = down_right_shifted_conv2d(
        out,
        network_info,
        row=row,
        col=col,
        cache_every=cache_every,
        run_every=run_every,
        nonlinearity=None,
        counters=counters)

    # Horizontal stack also takes in as input the vertical stack.
    cache_col = col // cache_every  # Compensates for striding in previous layers.
    v_stack_pixel = v_stack_row_input[:, :, cache_col:(cache_col + 1), :]
    v_shape = v_stack_pixel.get_shape()
    v_stack_pixel.set_shape([li.batch, 1, 1, li.input_channels])

    if extra_pixel_input is not None:
        # For skip connections between downsampling and upsampling layers.
        v_stack_pixel = tf.concat([v_stack_pixel, extra_pixel_input], 3)

    out += nn.nin(
        li.nonlinearity(v_stack_pixel), li.filter_channels, counters=counters)
    out = li.nonlinearity(out)
    network_info = (li.image_size, (li.filter_height, li.filter_width, 2 *
                                    li.filter_channels))
    out = down_right_shifted_conv2d(
        out,
        network_info,
        row=row,
        col=col,
        cache_every=cache_every,
        run_every=run_every,
        nonlinearity=None,
        counters=counters)

    if h is not None:
        out += _conditional_info(h, li.batch, li.filter_channels, counters)

    out = pixel_input + _gated_nonlinearity(out)
    return out
