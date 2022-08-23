#!/usr/bin/env python3

import argparse

from astropy import wcs
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import papy.plot
import tqdm
import yaml

import common


def get_tile(hdul):
    """ Get tile data for an HDU list

    Returns
    =======
    Tx : array of shape (ny, nx)
        Helioprojective longitude in arcsec
    Ty : array of shape (ny, nx)
        Helioprojective latitude in arcsec
    I : array of shape (ny, nx)
        Intensity
    """
    tile_wcs = wcs.WCS(hdul[0].header)
    I = hdul[0].data.T
    weights = hdul[1].data.T

    return tile_wcs, I, weights


def get_all_tiles(filenames):
    tiles_wcs = []
    I = []
    weights = []
    for filename in tqdm.tqdm(filenames, desc='Loading tiles'):
        with fits.open(filename) as hdul:
            this_tile_wcs, this_I, this_weights = get_tile(hdul)
        tiles_wcs.append(this_tile_wcs)
        I.append(this_I)
        weights.append(this_weights)
    return tiles_wcs, np.array(I), np.array(weights)


def ensure_int(arr):
    arr_int = arr.round().astype('int')
    if not np.all(np.isclose(arr, arr_int)):
        msg = f"array doesn't contain integer values: {arr}"
        raise ValueError(msg)
    return arr_int


def merge_tiles(tiles_wcs, I, weights, common_wcs):
    assembled_shape = (common_wcs.wcs.crpix * 2).astype(int)

    assembled_I = np.full(assembled_shape, 0, dtype=I[0].dtype)
    assembled_weights = np.full(assembled_shape, 0, dtype=weights[0].dtype)
    merged_map = np.full(assembled_shape, False, dtype=bool)
    for this_wcs, this_I, this_weights in tqdm.tqdm(
            zip(tiles_wcs, I, weights),
            desc='Merging tiles',
            total=len(I)):
        # Size of tile
        nx, ny = ensure_int(this_wcs.wcs.crpix) * 2
        # Index of tile central pixel in the assembled mosaic
        ix0, iy0 = ensure_int(
            this_wcs.wcs.crval / this_wcs.wcs.cdelt + common_wcs.wcs.crpix
            )
        # Note: cripx and crval/cdelt are set to integer values in
        # process_rasters.py, but we still check this with ensure_int() for
        # extra safety
        sl = (
            slice(ix0 - nx//2, ix0 + nx//2),
            slice(iy0 - ny//2, iy0 + ny//2),
            )
        mask = (this_weights != 0) & np.isfinite(this_weights)
        # cut-out borders
        mask[:30] = False
        mask[-60:] = False
        mask[:, :20] = False
        mask &= ~merged_map[sl]  # keep earlier tiles
        assembled_I[sl][mask] = this_I[mask]
        assembled_weights[sl][mask] = this_weights[mask]
        merged_map[sl][mask] = True
    # assembled_I /= assembled_weights

    header = common_wcs.to_header()
    hdul = fits.HDUList([
        fits.PrimaryHDU(assembled_I.T, header=header),
        ])

    return hdul


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--spec-win', required=True,
                   help='spectral window')
    args = p.parse_args()
    common.validate_spectral_window(args.spec_win)

    filenames = [f'io/tiles_{args.spec_win}_{i:03d}.fits' for i in range(25)]

    tiles_wcs, I, weights = get_all_tiles(filenames)

    with open(f'io/tiles_{args.spec_win}_common_wcs.yml') as f:
        common_wcs = wcs.WCS(yaml.safe_load(f))

    hdul = merge_tiles(tiles_wcs, I, weights, common_wcs)

    filenames = hdul.writeto(f'io/mosaic_{args.spec_win}.fits', overwrite=True)

    img = hdul[0].data
    img[np.isnan(img)] = 0
    plt.clf()
    ax = papy.plot.get_imshowsave_ax(img.shape, plt.gcf(), clearfig=True)
    ax.imshow(
        img,
        vmin=np.nanpercentile(img, 1),
        vmax=np.nanpercentile(img, 99.5),
        cmap='gray',
        )
    plt.savefig(f'io/mosaic_{args.spec_win}_preview.pdf')
