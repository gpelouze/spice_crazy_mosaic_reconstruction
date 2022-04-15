#!/usr/bin/env python3

import re
import os

from astropy import units as u
from dateutil.parser import parse as parse_date
import numpy as np
import pandas as pd
import scipy.interpolate as si


class SpiceUtils:
    re_spice_L123_filename = re.compile(
        r"""
        solo
        _(?P<level>L[123])
        _spice
            (?P<concat>-concat)?
            -(?P<slit>[wn])
            -(?P<type>(ras|sit|exp))
            (?P<db>-db)?
            (?P<int>-int)?
        _(?P<time>\d{8}T\d{6})
        _(?P<version>V\d{2})
        _(?P<SPIOBSID>\d+)-(?P<RASTERNO>\d+)
        \.fits
        """,
        re.VERBOSE)

    @staticmethod
    def read_spice_uio_catalog():
        """
        Read UiO text table SPICE FITS files catalog
        http://astro-sdc-db.uio.no/vol/spice/fits/spice_catalog.txt

        Return
        ------
        pandas.DataFrame
            Table

        Example queries that can be done on the result:

        * `df[(df.LEVEL == "L2") & (df["DATE-BEG"] >= "2020-11-17") \
          & (df["DATE-BEG"] < "2020-11-18") & (df.XPOSURE > 60.)]`
        * `df[(df.LEVEL == "L2") \
          & (df.STUDYDES == "Standard dark for cruise phase")]`

        Source: https://spice-wiki.ias.u-psud.fr/doku.php/data:data_analysis_manual:read_catalog_python
        """
        cat_file = os.path.join(
            os.getenv('SOLO_ARCHIVE', '/archive/SOLAR-ORBITER/'),
            'SPICE/fits/spice_catalog.txt')
        columns = list(pd.read_csv(cat_file, nrows=0).keys())
        date_columns = ['DATE-BEG', 'DATE', 'TIMAQUTC']
        df = pd.read_table(
            cat_file, skiprows=1, names=columns, na_values="MISSING",
            parse_dates=date_columns, low_memory=False
            )
        df.LEVEL = df.LEVEL.apply(lambda string: string.strip())
        df.STUDYTYP = df.STUDYTYP.apply(lambda string: string.strip())
        return df

    @staticmethod
    def parse_filename(filename):
        m = SpiceUtils.re_spice_L123_filename.match(filename)
        if m is None:
            raise ValueError(f'could not parse SPICE filename: {filename}')
        return m.groupdict()

    @staticmethod
    def ias_fullpath(filename):
        d = SpiceUtils.parse_filename(filename)
        date = parse_date(d['time'])

        fullpath = os.path.join(
            os.getenv('SOLO_ARCHIVE', '/archive/SOLAR-ORBITER/'),
            'SPICE/fits/',
            'level' + d['level'].lstrip('L'),
            f'{date.year:04d}/{date.month:02d}/{date.day:02d}',
            filename)

        return fullpath


def ang_to_pipi(a):
    """ Convert any angle to the range ]-pi, pi]

    Parameters
    ==========
    a : float or astropy quantity
        Angle to convert. If it has no quantity, assumes it is in radians.

    Returns
    =======
    a : float or astropy quantity
        Converted angle.
    """
    if hasattr(a, 'unit'):
        pi = u.Quantity(np.pi, 'rad').to(a.unit)
    else:
        pi = np.pi
    return - ((- a + pi) % (2*pi) - pi)


def remap(new_Tx, new_Ty, img, Tx, Ty):
    points = np.array([Tx, Ty]).reshape((2, -1)).T
    values = img.flatten()
    new_points = np.array([new_Tx, new_Ty]).reshape((2, -1)).T
    new_values = si.griddata(points, values, new_points)
    new_values = new_values.reshape(new_Tx.shape)
    return new_values


def validate_spectral_window(spec_win):
    allowed_spec_wins = [
        'Ne VIII 770 - SH',
        'C III 977 - SH',
        'Ly Beta 1025 - SH',
        'O VI 1032 - SH',
        ]
    if spec_win not in allowed_spec_wins:
        raise ValueError(f'invalid window: {spec_win}')


def get_mosaic_filenames():
    cat = SpiceUtils.read_spice_uio_catalog()
    filters = (
        (cat['LEVEL'] == 'L2')
        & (cat['MISOSTUD'] == '2093')
        & (cat['DATE-BEG'] >= '2022-03-07T06:59:59')
        & (cat['DATE-BEG'] <= '2023-03-07T11:29:59')
        )
    res = cat[filters]
    filenames = list(res['FILENAME'])
    return filenames
