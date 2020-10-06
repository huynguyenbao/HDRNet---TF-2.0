# !pip install rawpy
import rawpy
import tensorflow as tf

def load_raw(path, size=None):
    path = bytes.decode(path.numpy())
    with rawpy.imread(path) as raw:
        image = raw.postprocess()

        if size is not None:
            image = tf.image.resize(image, size)
            image = tf.cast(image, tf.float32)

        return image / 255.0

def load_jpg(path, size=None):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, 3)

    if size is not None:
        image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32)

    return image / 255.0


def preprocess_data(raw_path, hdr_path, low_size=256, full_size=1024, batch_size=1):
    BATCH_SIZE = batch_size

    raw_files = tf.data.Dataset.list_files(raw_path, shuffle=False)
    hdr_files = tf.data.Dataset.list_files(hdr_path, shuffle=False)

    lowers = raw_files.map(lambda x: tf.py_function(load_raw, [x, [low_size, low_size]], tf.float32)).batch(BATCH_SIZE)
    fullers = raw_files.map(lambda x: tf.py_function(load_raw, [x, [full_size, full_size]], tf.float32)).batch(BATCH_SIZE)
    targets = hdr_files.map(lambda x: tf.py_function(load_jpg, [x, [full_size, full_size]], tf.float32)).batch(BATCH_SIZE)

    return lowers, fullers, targets