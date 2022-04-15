#!/usr/bin/env python3

import argparse
import functools
import itertools
import multiprocessing as mp

from astropy import wcs
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import papy.plot
import tqdm
import yaml

from slot_response import slot_response
import common


def get_slot_images(hdu):
    """ Get slot images for an HDU

    Returns
    =======
    Tx : list of arrays of shape (ny, nx)
        Helioprojective longitude in arcsec, for each slit position
    Ty : list of arrays of shape (ny, nx)
        Helioprojective latitude in arcsec, for each slit position
    imgs : list of arrays of shape (ny, nx)
        Image for each slit position
    """
    # get world coordinates
    nt, nD, ny, nx = hdu.data.shape
    ix = np.arange(0, nx)
    iy = np.arange(0, ny)
    iD = np.arange(0, nD)
    it = np.arange(0, nt)
    px = np.meshgrid(ix, iy, iD, it, indexing='ij')
    w = wcs.WCS(hdu.header)
    world = w.pixel_to_world(*px)

    # Remove time axis and reformat data
    # shape for new arrays: nslit, nTy, nTx
    Txs = world[0][:, :, :, 0]
    Tys = world[1][:, :, :, 0]
    imgs = hdu.data[0].T
    assert Txs.shape == Tys.shape
    assert Txs.shape == imgs.shape

    Txs = common.ang_to_pipi(Txs)
    Tys = common.ang_to_pipi(Tys)
    Txs = Txs.to('arcsec').value
    Tys = Tys.to('arcsec').value

    imgs = imgs[:, :, ::-1]

    return Txs, Tys, imgs


def get_all_slot_images(filenames, spec_win):
    all_Txs = []
    all_Tys = []
    all_imgs = []
    for filename in tqdm.tqdm(filenames, desc='Opening data'):
        filename = common.SpiceUtils.ias_fullpath(filename)
        with fits.open(filename) as hdul:
            hdu = hdul[spec_win]
            Txs, Tys, imgs = get_slot_images(hdu)
        all_Txs.append(Txs)
        all_Tys.append(Tys)
        all_imgs.append(imgs)
    return np.array(all_Txs), np.array(all_Tys), np.array(all_imgs)


def get_common_grid(Tx, Ty, new_CDELT=1):
    Tx_wid = 2 * np.max([np.max(x) for x in np.abs(Tx)])
    Ty_wid = 2 * np.max([np.max(y) for y in np.abs(Ty)])
    wid_world = np.max([Tx_wid, Ty_wid])
    wid_px = int(np.ceil(wid_world / new_CDELT))
    # add margin
    wid_px += 4
    # make even
    wid_px += wid_px % 2
    new_CRVAL = 0
    new_CRPIX = wid_px / 2

    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [new_CDELT, new_CDELT]
    w.wcs.crval = [new_CRVAL, new_CRVAL]
    w.wcs.crpix = [new_CRPIX, new_CRPIX]
    w.wcs.ctype = ['HPLN-TAN', 'HPLT-TAN']
    w.wcs.cunit = ['arcsec', 'arcsec']

    return w


def get_common_tile_size(Tx, Ty, common_wcs):
    cdeltx = u.Quantity(common_wcs.wcs.cdelt[0], common_wcs.wcs.cunit[0])
    cdelty = u.Quantity(common_wcs.wcs.cdelt[1], common_wcs.wcs.cunit[1])
    cdeltx = cdeltx.to('arcsec').value
    cdelty = cdelty.to('arcsec').value
    nx = int(np.ceil(Tx.ptp(axis=(1, 2, 3)).max() / cdeltx))
    ny = int(np.ceil(Ty.ptp(axis=(1, 2, 3)).max() / cdelty))
    # add margin
    nx += 4
    ny += 4
    # make even
    nx += nx % 2
    ny += ny % 2
    return nx, ny


def get_new_raster_coordinates(common_wcs, nx, ny, Tx, Ty):
    new_CDELTx = u.Quantity(common_wcs.wcs.cdelt[0], common_wcs.wcs.cunit[0])
    new_CDELTy = u.Quantity(common_wcs.wcs.cdelt[1], common_wcs.wcs.cunit[1])
    new_CDELTx = new_CDELTx.to('arcsec').value
    new_CDELTy = new_CDELTy.to('arcsec').value

    new_CRPIXx = nx / 2
    new_CRPIXy = ny / 2

    new_CRVALx = np.round(np.mean(Tx) / new_CDELTx) * new_CDELTx
    new_CRVALy = np.round(np.mean(Ty) / new_CDELTy) * new_CDELTy

    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [new_CDELTx, new_CDELTy]
    w.wcs.crval = [new_CRVALx, new_CRVALy]
    w.wcs.crpix = [new_CRPIXx, new_CRPIXy]
    w.wcs.ctype = ['HPLN-TAN', 'HPLT-TAN']
    w.wcs.cunit = ['arcsec', 'arcsec']

    ix = np.arange(nx)
    iy = np.arange(ny)
    px = np.meshgrid(ix, iy, indexing='ij')
    new_Tx, new_Ty = w.pixel_to_world(*px)
    new_Tx = common.ang_to_pipi(new_Tx)
    new_Ty = common.ang_to_pipi(new_Ty)
    new_Tx = new_Tx.to('arcsec').value
    new_Ty = new_Ty.to('arcsec').value

    return w, new_Tx, new_Ty


def assemble_raster(slot_resp, common_wcs, nx, ny, Tx, Ty, I):
    new_wcs, new_Tx, new_Ty = get_new_raster_coordinates(
        common_wcs, nx, ny, Tx, Ty
        )

    assembled_I = np.full(new_Tx.shape, 0, dtype=I.dtype)
    assembled_weights = np.full(new_Tx.shape, 0, dtype=I.dtype)
    for this_Tx, this_Ty, this_I in zip(Tx, Ty, I):
        new_I = common.remap(new_Tx, new_Ty, this_I, this_Tx, this_Ty)
        new_slot_resp = common.remap(
            new_Tx, new_Ty, slot_resp, this_Tx, this_Ty
            )
        new_I[np.isnan(new_I)] = 0
        new_slot_resp[np.isnan(new_slot_resp)] = 0
        assembled_I += new_I
        assembled_weights += new_slot_resp

    assembled_I /= assembled_weights

    header = new_wcs.to_header()
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(assembled_I.T, header=header),
            fits.ImageHDU(assembled_weights.T, header=header),
            ]
        )

    return hdul


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument(
        '--spec-win', required=True,
        help='spectral window'
        )
    p.add_argument(
        '-c', '--cores', type=int,
        help='multiprocessing cores'
        )
    args = p.parse_args()
    common.validate_spectral_window(args.spec_win)
    filenames = common.get_mosaic_filenames()

    Tx, Ty, I = get_all_slot_images(filenames, args.spec_win)
    slot_resp = slot_response(I)

    # Crop above and below the slit
    iymin, iymax = 103, 863
    Tx = Tx[..., iymin:iymax + 1, :]
    Ty = Ty[..., iymin:iymax + 1, :]
    I = I[..., iymin:iymax + 1, :]
    slot_resp = slot_resp[iymin:iymax + 1, :]

    common_wcs = get_common_grid(Tx, Ty, new_CDELT=1)
    tile_nx, tile_ny = get_common_tile_size(Tx, Ty, common_wcs)

    # Assemble rasters
    job_worker = functools.partial(
        assemble_raster,
        slot_resp,
        common_wcs,
        tile_nx,
        tile_ny,
        )
    job_iter = zip(Tx, Ty, I)
    if args.cores is not None:
        p = mp.Pool(args.cores)
        try:
            hduls = p.starmap(job_worker, job_iter, chunksize=1)
        finally:
            p.terminate()
    else:
        job_iter = tqdm.tqdm(job_iter, total=len(I), desc='Remapping')
        hduls = list(itertools.starmap(job_worker, job_iter))

    with open(f'io/tiles_{args.spec_win}_common_wcs.yml', 'w') as f:
        d = dict(common_wcs.to_header())
        yaml.safe_dump(d, f, sort_keys=False)

    for i, hdul in enumerate(hduls):
        hdul.writeto(f'io/tiles_{args.spec_win}_{i:03d}.fits', overwrite=True)

        img = hdul[0].data
        plt.clf()
        ax = papy.plot.get_imshowsave_ax(img.shape, plt.gcf(), clearfig=True)
        ax.imshow(
            img,
            vmin=np.nanpercentile(img, 1),
            vmax=np.nanpercentile(img, 99.5),
            cmap='gray',
            )
        plt.savefig(f'io/tiles_{args.spec_win}_preview_{i:03d}.pdf')
