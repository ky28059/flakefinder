# Flake finder
> **Warning:** currently targets monolayer graphene on SiO2, with specific Leica color settings
This is not ready for general use

### Usage
`python monolayer_finder.py --q <PATH>`

The input file specified by <PATH> should be a text document containing ONLY Input and Output paths for all imagesets you wish to queue, formatted as below:

```
InputDir: C:\DummyIn\01_19_23_1\Scan 002\TileScan_001\, OutputDir: C:\DummyIn\01_19_23_1\Scanned\
InputDir: C:\DummyIn\01_18_23_1b\Scan 002\TileScan_001\, OutputDir: C:\DummyIn\01_18_23_1b\Scanned\
```

The code expects the TileScan_001 directory to contain all the images from a scan, plus the leicametadata folder with a .xlif inside to position the images

Do not specify directories on the Google Drive or the computer may crash upon execution.
The OutputDir will be created if it does not exist, and a Summary.txt file of performance and parameters will be written to each OutputDir after each InputDir has been analyzed.
Performance ~1-10s per 20 megapixel file at scale=1, k=4 on the Leica computer, depending on how many filter steps the image passes

## Parameters
* 'threadsave' : how many logical processors the program does NOT use on your CPU. Can reduce to improve computer usability while program is running at cost of runtime
* 'boundflag' : 0/1 to turn off/on a function that calculates flake area and border pixels. Processing intensive but gives useful feedback to both user and developer
* `k`: scale factor for the image before performing DB scan. Currently set to 4, imscale = (256*k,171*k) - for full 20MP resolution, k=21
Beyond this you'll need to mess with the epsilon parameter of DBScan.
* `t_rgb_dist`: Initial pixel thresholding. Pixels have to be within euclidian distance (radius) of RGB value to be included in DB scan
* `t_hue_dist`: Max distance from target hue in HSV space for a pixel's hue to be considered "good" for DB scan - currently unused
* `t_color_match_count`: Minimum number of "good" pixels an image must have before we'll bother with DBscan. Otherwise,
the image is discarded. This makes for fast filtering of images with no relevant content.
* `t_min_cluster_pixel_count`: minimum points a DBScan cluster can have to be valid
* `t_max_cluster_pixel_count`: maximum points a DBScan cluster can have to be valid
* `scale`: The output resolution of images with identified flakes, relative to the input resolution. Does not affect DBscan. Set to 1 for identical input/output

### Suggested Tuning:
- Set k to 4 for higher quality results (with somewhat slower processing time)
- Increase filters `t_hue_dist` and especially `t_rgb_dist` to include more potential images.
- tweak t_[min/max]_cluster_pixel_count to exclude clusters obviously too small/large to be flakes
