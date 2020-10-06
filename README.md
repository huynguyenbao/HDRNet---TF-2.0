# Deep Bilateral Learning for Real-Time Image Enhancements
Unofficial PyTorch implementation of 'Deep Bilateral Learning for Real-Time Image Enhancement', SIGGRAPH 2017 https://groups.csail.mit.edu/graphics/hdrnet/

Python 3.6

### Dependencies

To install the Python dependencies, run:

    pip install -r requirements.txt
    
## Datasets
    HDR+ Burst Photography Dataset - https://hdrplusdata.org/dataset.html

### Getting the data
To get started, using the subset of bursts (153 bursts, 37 GiB).

    gsutil -m cp -r gs://hdrplusdata/20171106_subset .


## Usage
    
To train a model, run the following command:

    python train.py 
    --raw_path="/content/drive/My Drive/HDR+ Dataset/20171106_subset/results_20171023/*/merged.dng"
    --hdr_path="/content/drive/My Drive/HDR+ Dataset/20171106_subset/results_20171023/*/final.jpg"

To test image run:

    python inference.py --pretrain_dir="weights//ckpt" --input_path=<raw image path> --output_path=<saved image path>
    

## Known issues

* PointwiseNN implemented not like paper.

