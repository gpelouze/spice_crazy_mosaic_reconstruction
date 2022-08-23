#!/usr/bin/env python3

import argparse
import os

from astropy.io import fits
import matplotlib as mpl
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import common


def get_all_slot_images(filenames, spec_win):
    all_imgs = []
    for filename in tqdm.tqdm(filenames, desc='Opening data'):
        filename = common.SpiceUtils.ias_fullpath(filename)
        with fits.open(filename) as hdul:
            hdu = hdul[spec_win]
            imgs = hdu.data[0].T
        all_imgs.append(imgs)
    return np.array(all_imgs)


def compute_slot_response(I):
    """ Compute slot response function for a series of images

    Parameters
    ==========
    I : array of shape (..., ny, nx)
        Images

    Returns
    =======
    slot_resp : array of shape (ny, nx)
        Slot response function
    """
    if I.ndim > 3:
        I = I.reshape((-1, *I.shape[-2:]))

    slot_resp = np.median(I, axis=0)
    vmin = np.nanpercentile(slot_resp, .5)
    vmax = np.nanpercentile(slot_resp, 99.5)
    # vmin = np.nanmedian(slot_resp[:, :11])
    # vmax = np.nanpercentile(slot_resp[:, 25:41], 99)
    slot_resp = (slot_resp - vmin) / (vmax - vmin)
    # slot_resp = np.clip(slot_resp, 0, 1)
    return slot_resp


def load_slot_response(spec_win):
    """ Load slot response function for a given spectral window
    Parameters
    ==========
    spec_win : str
        Name of the spectral window
    Returns
    =======
    slot_resp : array of shape (ny, nx)
        Slot response function
    """
    slot_resp = np.load(os.path.join(
        'io', 'slot_response',
        f'{spec_win}_slot_response.npy'
        ))
    # slot_resp = np.clip(slot_resp, 0, None)
    return slot_resp

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--spec-win', required=True,
                   help='spectral window')
    args = p.parse_args()
    common.validate_spectral_window(args.spec_win)
    filenames = common.get_mosaic_filenames()

    # Compute
    slot_resp = load_slot_response(args.spec_win)

    # Save
    filename = f'io/slot_response_{args.spec_win}.fits'
    hdul = fits.HDUList([fits.PrimaryHDU(slot_resp)])
    hdul.writeto(filename, overwrite=True)

    # Plot
    plt.clf()
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 2],
                               left=.05, right=.98)
    gs.tight_layout(plt.gcf())
    ax2 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax1.plot(np.nanmedian(slot_resp, axis=0))
    ax1.set_xlabel('$X$ [px]')
    ax1.set_ylabel('Response')
    m = ax2.imshow(slot_resp, vmin=0, vmax=1, aspect=.25)
    # plt.colorbar(m, label='Response')
    ax2.set_xlabel('$X$ [px]')
    ax2.set_ylabel('$Y$ [px]')
    plt.savefig(f'io/slot_response_{args.spec_win}.pdf')
    # plt.show()