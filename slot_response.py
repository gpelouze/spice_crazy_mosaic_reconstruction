#!/usr/bin/env python3

import argparse

from astropy.io import fits
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
            imgs = imgs[:, :, ::-1]
        all_imgs.append(imgs)
    return np.array(all_imgs)


def slot_response(I):
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


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--spec-win', required=True,
                   help='spectral window')
    args = p.parse_args()
    common.validate_spectral_window(args.spec_win)
    filenames = common.get_mosaic_filenames()

    # Compute
    I = get_all_slot_images(filenames, args.spec_win)
    slot_resp = slot_response(I)

    # Save
    filename = f'io/slot_response_{args.spec_win}.fits'
    hdul = fits.HDUList([fits.PrimaryHDU(slot_resp)])
    hdul.writeto(filename, overwrite=True)

    # Plot
    plt.clf()
    plt.subplot(131)
    slot_resp_y = np.nanmedian(slot_resp, axis=1)
    w = np.where(slot_resp_y > 0.5)[0]
    if w.size > 2:
        w1 = w[0]
        w2 = w[-1]
        plt.axvline(w1, color='C1', label=f'{w1}')
        plt.axvline(w2, color='C2', label=f'{w2}')
        print(args.spec_win, w1, w2)
    plt.plot(slot_resp_y, 'C0')
    plt.legend()
    plt.xlabel('$Y$ [px]')
    plt.ylabel('$r$')
    plt.subplot(132)
    plt.plot(np.nanmedian(slot_resp, axis=0))
    plt.xlabel('$X$ [px]')
    plt.ylabel('$r$')
    plt.subplot(133)
    plt.imshow(slot_resp, vmin=0, vmax=1, aspect=.25)
    plt.colorbar(label='$r$')
    plt.xlabel('$X$ [px]')
    plt.ylabel('$Y$ [px]')
    plt.savefig(f'io/slot_response_{args.spec_win}.pdf')
