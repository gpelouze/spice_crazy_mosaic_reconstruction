#!/usr/bin/env python3

import argparse

from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import papy.plot
import skimage

import common


class ScalingFunctions:
    @staticmethod
    def log(x, b):
        x = (b-1)*x + 1
        return np.log(x) / np.log(b)


def fill_gaps(img):
    s = np.array(img.shape)
    xy = np.indices(s) - s.reshape(-1, 1, 1) / 2
    r = np.sqrt(np.sum(xy**2, axis=0))
    r0 = r.max() / np.sqrt(2) * 0.75
    disk_mask = (r < r0)

    gap_mask = (img == 0) & disk_mask

    fill_value = np.nanmedian(img[disk_mask])

    img[gap_mask] = fill_value


def process_img(img, spec_win, vmins=None, vmaxs=None):
    if vmins:
        vmin = vmins[spec_win]
    else:
        vmin = np.nanpercentile(img, 1)
    if vmaxs:
        vmax = vmaxs[spec_win]
    else:
        vmax = np.nanpercentile(img, 99.99)
    print(spec_win, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
    img = ScalingFunctions.log(img, 100)
    img = np.clip(img, 0, 1)
    img[np.isnan(img)] = 0

    return img


def color_list_to_cmap(color):
    name = color.lstrip('#')
    name = 'lab_camp_' + name

    color = mpl.colors.to_rgb(color)

    _, a, b = skimage.color.rgb2lab(color)
    L = np.linspace(0, 100, 256)
    a = np.repeat(a, L.size)
    b = np.repeat(b, L.size)
    colors = np.stack([L, a, b]).T
    rgb = skimage.color.lab2rgb(colors)
    rgb[0] = [0, 0, 0]

    # convert to mpl colormap
    rgb_names = ('red', 'green', 'blue')
    x = np.linspace(0, 1, len(rgb))
    cdict = {k: list(zip(x, v, v))
             for k, v in zip(rgb_names, rgb.T)}
    return mpl.colors.LinearSegmentedColormap(name, cdict)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--spec-win', required=True,
                   help='spectral window')
    args = p.parse_args()
    common.validate_spectral_window(args.spec_win)

    vmins = {
        'Ne VIII 770 - SH': .02,
        'C III 977 - SH': .05,
        'Ly Beta 1025 - SH': .15,
        'O VI 1032 - SH': .06,
        }
    cmaps = {
        'Ly Beta 1025 - SH': color_list_to_cmap('#746a8b'),
        'C III 977 - SH': color_list_to_cmap('#6a798b'),
        'O VI 1032 - SH': color_list_to_cmap('#6a8b80'),
        'Ne VIII 770 - SH': color_list_to_cmap('#a9804c'),
        }

    hdul = fits.open(f'io/mosaic_{args.spec_win}.fits')
    img = hdul[0].data

    fill_gaps(img)
    img = process_img(img, args.spec_win, vmins=vmins)

    # cut and rebin
    ny, nx = img.shape
    if nx > 4000:
        img = papy.num.rebin(img, (4, 4), method=np.nanmean, cut_to_bin=True)

    plt.clf()
    ax = papy.plot.get_imshowsave_ax(img.shape, plt.gcf(), clearfig=True)
    ax.imshow(
        img,
        vmin=0,
        vmax=1,
        cmap=cmaps[args.spec_win],
        )
    plt.savefig(f'io/mosaic_processed_{args.spec_win}_processed.pdf')
    plt.savefig(f'io/mosaic_processed_{args.spec_win}_processed.png')
