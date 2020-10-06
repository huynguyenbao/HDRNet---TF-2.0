from model import HDRPointwiseNN
import tensorflow as tf
import argparse
from skimage import exposure
import matplotlib.pyplot as plt
import rawpy
import os

def load_raw(path, size=None):
    with rawpy.imread(path) as raw:
        image = raw.postprocess()
        if size is not None:
            image = tf.image.resize(image, [size, size])
            image = tf.cast(image, tf.float32)

        return image / 255.0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrain_dir', type=str, default= "weights//ckpt")
    parser.add_argument('--low_size', type=int, default=None)
    parser.add_argument('--full_size', type=int, default=None)

    config = parser.parse_args()

    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    # Create Model
    model = HDRPointwiseNN()

    # Read raw
    lower = load_raw(config.input_path, config.low_size)[None]
    fuller = load_raw(config.input_path, config.full_size)[None]

    # Load pretrained weight
    if config.pretrain_dir:
        model((lower, fuller))
        model.load_weights(config.pretrain_dir)
    

    # Inference
    hdr_image = model((lower, fuller))[0]
    hdr_image = exposure.rescale_intensity(hdr_image.numpy(), out_range=(0.0,255.0)).astype(np.uint8)

    # Save Image
    plt.imsave(config.output_path, hdr_image)


