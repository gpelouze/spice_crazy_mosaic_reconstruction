# Assemble SPICE crazy mosaic

## Pipeline

1. `process_rasters.py`: process individual rasters, merging all slot
    - input: 
        - SPICE L2 FITS
    - output:
        - `io/tiles_{spec_win}_common_wcs.yml`
        - `io/tiles_{spec_win}_{i:03d}.fits`
        - `io/tiles_{spec_win}_preview_{i:03d}.pdf`

2. `assemble_rasters.py`: assemble rasters into the mosaic
    - input:
        - `io/tiles_{spec_win}_common_wcs.yml`
        - `io/tiles_{spec_win}_{i:03d}.fits`
    - output:
        - `io/mosaic_{spec_win}.fits`
        - `io/mosaic_{spec_win}_preview.pdf`

3. `render.py`: render the mosaic into a pretty image
    - input:
        - `io/mosaic_{spec_win}.fits`
    - output:
        - `io/mosaic_processed_{spec_win}_processed.pdf`
        - `io/mosaic_processed_{spec_win}_processed.png`

## Other scripts

- `common.py`
- `slot_response.py`: compute the slot response from a series of images
    - output:
        - `io/slot_response_{spec_win}.fits`
        - `io/slot_response_{spec_win}.pdf`
