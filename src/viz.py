import argparse
import csv
import datetime
import io
import logging
import os
import glob
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from matplotlib import rcParams
params = {
   'axes.labelsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [4.5, 4.5],
   'savefig.dpi': 400
   }
rcParams.update(params)

from generator import DataGenerator

from sklearn.metrics import confusion_matrix
from skimage.transform import resize

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, default=None,
                        help='Path of the saved ConvNet to visualize.')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for filenames of the outputs and logs.')
    parser.add_argument('--log', action='store_true',
                        help='Whether or not to create a log in log-dir.')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='The log directory.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Testing output directory.')

    parser.add_argument('--type', type=str, choices=['filter', 'kernel', 'filt_act', 'class_act'],
                        default=None, nargs='+',
                        help='Aspect of the model to visualize')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data sample to visualize.')
    parser.add_argument('--label-idx', type=int, default=0,
                        help='Index of the label in the model output.')
    parser.add_argument('--grid-dim', type=tuple, default=None,
                        help='Tuple of ints for the shape of subplots. Also determines the number of filters to plot.')

    return parser


def _determine_grid(grid_dim):
    """
    Get the grid size as specified or set defaults if not valid.

    :param grid_dim: tuple of ints
    """
    nrow = 2
    ncol = 5
    if grid_dim:
        nrow = grid_dim[0] if grid_dim[0] > 0 else nrow
        ncol = grid_dim[1] if grid_dim[1] > 0 else ncol
    return nrow, ncol


def _format_plots(axs):
    """
    Format the axes.

    :param axs: a single or list of matplotlib.pyplot.Axes objects
    """
    try:
        axs[0]
    except TypeError:
        axs = [axs]

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])


def normalize(a):
    a = (a - a.min()) / (a.max() - a.min())
    a = np.nan_to_num(a)
    return a


def visualize_filters(model, grid_dims=None, output_dir=None):
    nrow, ncol = _determine_grid(grid_dim)
    filters, biases = model.layers[0].get_weights()

    # normalize the filter weights to construct an image of the filter
    min_filter, max_filter = filters.min(), filters.max()
    norm_filters = (filters - min_filter) / (max_filter - min_filter)

    fig, axs = plt.subplots(1, args.n_filters)
    _format_plots(axs.flatten())

    for i, ax in enumerate(axs):
        filt = filters[:, :, i]
        ax.imshow(filt.T)

    fig.tight_layout()
    out_path = os.path.join(output_dir or '', 'filters.png')
    fig.savefig(out_path)


def visualize_kernel(model, grid_dim=None, output_dir=None, epochs=100, step_size=1):
    nrow, ncol = _determine_grid(grid_dim)

    data = np.random.random((1, 64, 12)) - 0.5

    fig_data, ax = plt.subplots(1, 1)
    ax.imshow(np.squeeze(data))
    fig_data.tight_layout()
    fig_data.savefig(f'data_kernel.png')

    for i, layer in enumerate(model.layers):
        # maximize activations to target layer using gradient ascent
        if not isinstance(layer, keras.layers.Conv1D):
            continue

        submodel = keras.models.Model([model.inputs[0]], [layer.output])
        input_data = tf.Variable(tf.cast(data, tf.float32))

        q = []
        for filter_idx in range(layer.filters):
            for _ in range(epochs):
                with tf.GradientTape() as tape:
                    outputs = submodel(input_data)
                    loss_val = tf.reduce_mean(outputs[:, :, filter_idx])

                grads = tape.gradient(loss_val, input_data)
                norm_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
                input_data.assign_add(norm_grads * step_size)
            q.append((loss_val.numpy(), np.squeeze(input_data.numpy())))
        
        q = sorted(q, key=lambda x: x[0], reverse=True)
        plot_nrow = layer.filters // 6 + 1 if not nrow else nrow
        plot_ncol = ncol

        fig_filter, axs_filter = plt.subplots(plot_nrow, plot_ncol)
        _format_plots(axs_filter.flatten())

        for ax, (loss_val, input_data) in zip(axs_filter.flatten(), q):
            loss_val = np.around(loss_val, 3)
            ax.set_ylabel(f'Loss={str(loss_val)}')
            ax.imshow(input_data, cmap='gist_ncar')

        fig_filter.savefig(f'kernel_{layer.name}.png')
        logging.info(f'Done layer: {layer.name}')


def visualize_filt_activations(model, grid_dim=None, output_dir=None, data=None):
    nrow, ncol = _determine_grid(grid_dim)

    if data is None:
        data = np.random.random((1, 64, 12)) - 0.5

    fig_data, ax = plt.subplots(1, 1)
    norm_data = np.apply_along_axis(normalize, 0, np.squeeze(data))
    ax.imshow(norm_data)
    fig_data.tight_layout()
    fig_data.savefig(f'data_filtact.png')

    for i, layer in enumerate(model.layers):
        # run data until the layer to get a feature map for the data
        if not isinstance(layer, keras.layers.Conv1D):
            continue

        submodel = keras.models.Model([model.inputs[0]], [layer.output])
        activations = submodel.predict(data)[0]

        fig_activation, ax_activation = plt.subplots(1, 1)
        _format_plots(ax_activation)

        resized_activations = resize(activations, (data.shape[1], activations.shape[1]))
        ax_activation.imshow(resized_activations, cmap='gray')

        fig_activation.subplots_adjust(wspace=0.1, hspace=0.1)
        fig_activation.savefig(f'filtact_{layer.name}.png')
        logging.info(f'Done layer: {layer.name}')


def visualize_class_activations(model, grid_dim=None, output_dir=None, data=None, label_idx=None):
    nrow, ncol = _determine_grid(grid_dim)

    if data is None:
        data = np.random.random((1, 64, 12)) - 0.5

    fig_data, ax = plt.subplots(1, 1)
    norm_data = np.apply_along_axis(normalize, 0, np.squeeze(data))
    ax.imshow(norm_data)
    fig_data.tight_layout()
    fig_data.savefig(f'data_clact.png')

    for i, layer in enumerate(model.layers):
        if not isinstance(layer, keras.layers.Conv1D):
            continue

        submodel = keras.models.Model([model.inputs], [layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_output, predictions = submodel(data)
            pred = predictions[:, label_idx]

        activations = conv_output[0]
        grads = tape.gradient(pred, conv_output)[0]

        weights = tf.reduce_mean(grads, axis=0)

        cam = np.ones(activations.shape[0], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[:, i]

        cam = np.maximum(cam, 0)
        activations = normalize(cam)

        fig_clact, axs_clact = plt.subplots(1, 2)
        _format_plots(axs_clact.flatten())

        resized_activations = resize(np.expand_dims(activations, axis=1), (data.shape[1], 1))

        axs_clact[0].imshow(norm_data)
        axs_clact[1].imshow(resized_activations, cmap='jet')

        fig_clact.subplots_adjust(wspace=-0.8, hspace=0.1)
        fig_clact.savefig(f'cam_{layer.name}.png')
        logging.info(f'Done layer: {layer.name}')


def main(args):
    model = keras.models.load_model(args.model_path, compile=False)
    model.summary()

    args.type = args.type or []

    data = np.random.random((1, 64, 12)) - 0.5
    if args.data_path:
        data = np.load(args.data_path)

        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=0)

    common_kwargs = dict(output_dir=args.output_dir, grid_dim=args.grid_dim)

    if 'filter' in args.type:
        visualize_filters(model, **common_kwargs)

    if 'kernel' in args.type:
        visualize_kernels(model, **common_kwargs)
    
    if 'filt_act' in args.type:
        visualize_filt_activations(model, data=data, **common_kwargs)

    if 'class_act' in args.type:
        visualize_class_activations(
            model, data=data, label_idx=args.label_idx, **common_kwargs)


if __name__ == '__main__':
    args = init_parser().parse_args()
    args._date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    args._pathname = '_'.join([s for s in [args.name, args._date] if s])

    handlers=[logging.StreamHandler()]
    if args.log:
        logname = f'log_{args._pathname}.log'
        fh = logging.FileHandler(os.path.join(args.log_dir or '', logname))
        handlers.append(fh)

    logging.basicConfig(
        format="[%(asctime)s] %(name)s (%(lineno)s) %(levelname)s: %(message)s",
        level=logging.INFO, handlers=handlers)

    main(args)
