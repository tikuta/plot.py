#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import seaborn as sns
import os
if 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('agg')
from matplotlib import pyplot as plt

sns.set_style("ticks")


def parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Graph plotter with matplotlib')
    parser.add_argument('input', nargs='+', type=str, help='CSV path to input, "label:path:color"')
    parser.add_argument('-o', '--output', nargs=1, type=str, default=[None], help='output path')
    parser.add_argument('-x', '--xlabel', nargs=1, type=str, default=[None], help='xlabel string')
    parser.add_argument('-y', '--ylabel', nargs=1, type=str, default=[None], help='ylabel string')
    parser.add_argument('--xlim', nargs=1, type=str, default=[None], help='xrange separated with ":", "XMIN:XMAX"')
    parser.add_argument('--ylim', nargs=1, type=str, default=[None], help='yrange separated with ":",  "YMIN:YMAX"')
    parser.add_argument('--xscale', nargs=1, type=str, default=['1.0'], help='xscale to be multiplied')
    parser.add_argument('--yscale', nargs=1, type=str, default=['1.0'], help='yscale to be multiplied')

    parser.add_argument('--moving-average', dest='skip', nargs=1, type=int, default=[None], help='shorten the data with moving average')

    parser.add_argument('--title', nargs=1, type=str, default=[''], help='graph title')

    parser.add_argument('--auto-legend', dest='legend', action='store_true', help='treat filenames as legend, default')
    parser.add_argument('--no-auto-legend', dest='legend', action='store_false', help='do not treat filename as legend')
    parser.set_defaults(legend=True)

    parser.add_argument('--alpha', nargs=1, type=float, default=[1.0], help='alpha value of drawings')

    return parser


def moving_average(x, y, n):
    xx = np.cumsum(x, dtype=np.float)
    xx[n:] = xx[n:] - xx[:-n]
    xx = xx[n-1:] / n

    yy = np.cumsum(y, dtype=np.float)
    yy[n:] = yy[n:] - yy[:-n]
    yy = yy[n-1:] / n

    return xx[::n], yy[::n]


def plot(input_file, label, color, xscale, yscale, skip, alpha, fill):
    data = np.loadtxt(input_file, delimiter=',')

    if len(data.shape) == 1:
        xx = np.arange(len(data)) * xscale
        yy = data * yscale
    else:
        xx = data.T[0] * xscale
        yy = data.T[1] * yscale

    if skip and len(xx) > skip:
        xx, yy = moving_average(xx, yy, skip)

    if fill:
        plt.fill_between(xx, 0, yy, where=0<=yy, facecolor=color)
    else:
        plt.plot(xx, yy, label=label, alpha=alpha, color=color)


if __name__ == '__main__':
    args = parser().parse_args()

    xscale = float(eval(args.xscale[0]))
    yscale = float(eval(args.yscale[0]))

    colors = {"UV0": "#C9D6E0",
              "UV1": "#3E74B7",
              "UV2": "#701970",
              "UV3": "#EB352E"}

    for f in args.input:
        params = f.split(':')

        label, filepath, color = None, f, None
        if len(params) == 2:
            if os.path.isfile(params[0]):
                filepath, color = params
            elif os.path.isfile(params[1]):
                label, filepath = params
        elif len(params) == 3:
            label, filepath, color = params

        # special colors
        if color in colors:
            color = colors['color']

        fill = label == 'FILL'

        plot(filepath, label, color, xscale, yscale, args.skip[0], args.alpha[0], fill)

    if args.legend:
        plt.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102))

    if args.xlabel[0]:
        plt.xlabel(args.xlabel[0])
    if args.ylabel[0]:
        plt.ylabel(args.ylabel[0])
    plt.title(args.title[0] if args.title[0] else "")

    if args.xlim[0]:
        plt.xlim([float(v) for v in args.xlim[0].split(':')])
    if args.ylim[0]:
        plt.ylim([float(v) for v in args.ylim[0].split(':')])

    if args.output[0] is None:
        plt.show()
    else:
        plt.savefig(args.output[0], bbox_inches='tight')
