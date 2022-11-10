#!/usr/bin/env python3

import os

import numpy as np


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