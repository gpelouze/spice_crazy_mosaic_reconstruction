import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec
from astropy.io import fits
import tqdm


def get_slot_response(img):
    return np.nanmedian(img[100:880], axis=0)


def get_slot_response_center(slot_response):
    x = np.arange(1, len(slot_response) + 1, 1)
    return np.nansum(slot_response * x) / np.nansum(slot_response)


def normalize_slot_response(img, slot_response, m, sr_center):
    # Choosing the intervals for which we will need to 'cut' the measured
    # slot_response
    second = round(sr_center) - 12
    third = round(sr_center) + 12
    first = 3  # first used value
    fourth = len(slot_response) - 3  # last used value

    # Conditions used to separate some weird slot responses
    # (e.g. cases where the image center is at negative x due to
    # the flat-field correction).
    wrong1 = (second < first)
    wrong2 = (fourth < third)
    wrong3 = (sr_center >= len(slot_response) - 5)
    wrong4 = (sr_center <= 5)

    if wrong1 or wrong2 or wrong3 or wrong4:
        # we have 'weird' values (e.g. negative weighted average)
        raise ValueError

    switch_initial = 0  # number to determine what happened
    switch_final = 0
    # First and last limit value
    x_left = 0
    x_right = 0
    for i in range(first, second + 1):
        if (slot_response[i] - slot_response[i + 1] > 0
                and switch_initial == 0):
            # first maximum
            # we got to the first maximum:
            switch_initial = 1
            # first slot response limit:
            x_left = i
    # Same but from the right
    for i in range(fourth, third - 1, -1):
        if slot_response[i] - slot_response[i - 1] > 0 and switch_final == 0:
            # last maximum
            # when we get to the last maximum:
            switch_final = 1
            # position of the limit:
            x_right = i
    if switch_initial == 0:
        # the first if didn't work
        for i in range(first, second + 1):
            if slot_response[i] - slot_response[i + 1] == 0:
                # first straight line if no maximum
                switch_initial = 2
                # position of the limit
                x_left = i
    # Same but from the right
    if switch_final == 0:
        # the first if didn't work
        for i in range(fourth, third - 1, -1):
            if slot_response[i] - slot_response[i - 1] == 0:
                # first straight line if no maximum
                switch_final = 2
                # position of the limit:
                x_right = i

    x_initial = 0
    x_final = 0
    if switch_initial == 1:
        # first if worked
        # position of the limit is the minimum between the two last limits:
        x_initial = np.nanargmin(slot_response[x_left:second + 1]) + x_left
    elif switch_initial == 2:
        # second if worked (i.e. no maximum but a straight line)
        # position of the limit is the last limit:
        x_initial = x_left
    # Same but from the right
    if switch_final == 1:
        # first if worked
        # position of the limit is the minimum between the two last limits:
        x_final = np.nanargmin(slot_response[third:x_right + 1]) + third
    elif switch_final == 2:
        # second if worked (i.e. no maximum but a straight line)
        # position of the limit is the last limit:
        x_final = x_right
    if x_initial == 0:
        # e.g. all nan slice was encountered, we need to restrain the borders
        first += m
        x_initial = np.nanargmin(slot_response[first:second + 1]) + first
    if x_final == 0:
        fourth -= m
        x_final = np.nanargmin(slot_response[third:fourth + 1]) + third

    # remove negative values
    slot_response_weighted_average = np.where(
        slot_response < 0, 0.0, slot_response
        )
    # taking out the values before our calculated limit
    slot_response_weighted_average[:x_initial] = 0.0
    slot_response_weighted_average[x_final + 1:] = 0.0

    #  Final slot response values by normalisation (30"/1,098")
    integral = np.nansum(slot_response_weighted_average) / 30 * 1.098
    slot_response = slot_response / integral
    img = img / integral

    # Calculating the weighted average of the final slot_response
    dispersion_array = np.arange(1, len(slot_response) + 1, 1)  # px
    image_center = np.nansum(
        slot_response_weighted_average * dispersion_array
        ) / np.nansum(slot_response_weighted_average)  # 'center' pixel

    return img, image_center, x_initial, x_final


def gen_individual_images(fits_files, spec_win_list, output_dir, m=2):
    """ Determine the weighted average for each slot response image """
    os.makedirs(output_dir, exist_ok=True)

    images = {spec_win: [] for spec_win in spec_win_list}
    i = 0

    for fits_fn in tqdm.tqdm(fits_files):
        hdul = fits.open(fits_fn)

        for spec_win in spec_win_list:
            for img in hdul[spec_win].data[0].T:
                i += 1

                slot_response = get_slot_response(img)
                sr_center = get_slot_response_center(slot_response)

                try:
                    img, sr_center, _, _ = normalize_slot_response(
                        img, slot_response, m, sr_center)
                except ValueError:
                    continue

                img_fn = f'{output_dir}/{spec_win}_{i}.npz'
                np.savez(
                    img_fn,
                    I=img,
                    filename=fits_fn,
                    image_center=sr_center,
                    )
                images[spec_win].append(img_fn)

        hdul.close()

    return images


def gen_slot_response(images, win_centers, output_dir, m=1):
    os.makedirs(output_dir, exist_ok=True)
    for spec_win, mosaic_win_center in win_centers.items():

        data = []
        for img_fn in images[spec_win]:
            dat = np.load(img_fn)
            sr_center = dat['image_center']
            win_center_distance = abs(sr_center - mosaic_win_center)
            data.append([dat['I'], win_center_distance, sr_center])

        def sorter(d):
            quality_factors = d[1]
            return quality_factors

        data = sorted(data, key=sorter)[:920]

        im = [d[0] for d in data]

        im = np.nanmedian(im, axis=0)

        slot_response = get_slot_response(im)
        sr_center = get_slot_response_center(slot_response)

        im, image_center, x_initial, x_final = normalize_slot_response(
            im, slot_response, m, sr_center)

        def adjust_ax_pos(ax, cax):
            p_cb = cax.get_position()
            p_im = ax.get_position()
            tx = p_cb.xmin - p_im.xmin - p_im.width - .02
            ax.set_position(p_im.translated(tx, 0))

        fig = plt.figure(clear=True)
        gs = mpl.gridspec.GridSpec(
            1, 3, width_ratios=[.8, .15, 2],
            left=0.04, right=.98,
            bottom=.11, top=.9,
            wspace=.3,
            )
        ax_resp = fig.add_subplot(gs[0])
        ax_cb = fig.add_subplot(gs[1])
        ax_median = fig.add_subplot(gs[2])
        plt.suptitle(f"Wide-slit response for {spec_win}")
        imm = ax_resp.imshow(im, vmin=0, vmax=1, aspect=.25, origin='lower')
        ax_resp.set_xlabel('$X$ [px]')
        ax_resp.set_ylabel('$Y$ [px]')
        plt.colorbar(imm, cax=ax_cb)
        ax_median.plot(get_slot_response(im), color='k')
        ax_median.set_xlabel('$X$ [px]')
        ax_cb.set_ylim(0, 1)
        ax_median.set_ylim(0, 1)
        ax_median.set(yticklabels=[])
        ax_resp.ticklabel_format(axis='y', style='sci', scilimits=(1, 5))
        adjust_ax_pos(ax_resp, ax_cb)
        plt.savefig(
            f'{output_dir}/{spec_win}_slot_response.pdf'
            )

        np.save(f'{output_dir}/{spec_win}_slot_response.npy', im)


if __name__ == '__main__':

    mosaic_win_centers = {
        'Ne VIII 770 - SH': 43,
        'C III 977 - SH': 35,
        'Ly Beta 1025 - SH': 35,
        'O VI 1032 - SH': 33,
        }
    spec_wins = list(mosaic_win_centers.keys())

    fits_filenames = (
        glob.glob('io/fits_crazy_mosaic/*.fits')
        + glob.glob('io/fits_windowed_flat/*.fits')
        )

    slot_images = gen_individual_images(fits_filenames, spec_wins,
                                        'io/slot_response/wins')
    gen_slot_response(slot_images, mosaic_win_centers,
                      'io/slot_response')
