import sys
import tensorflow as tf
import numpy as np
import time
from preprocess import preprocess_data
from model import HDRPointwiseNN
import argparse
import os


def progress(epoch, trained_sample ,total_sample, bar_length=25, loss=0, message=""):
    percent = float(trained_sample) / float(total_sample)
    hashes = '#' * int(tf.round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rEpoch {0}: [{1}] {2}%  ----- Loss: {3}".format(epoch, hashes + spaces, int(round(percent * 100)), float(loss)) + message)
    sys.stdout.flush()

def L2(y_pred, y_truth):
    return tf.reduce_mean(tf.pow(y_pred - y_truth, 2))

def train_step(model, sample_lower, sample_fuller, sample_target, optimizer):
    variables = model.variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        pred = model((sample_lower, sample_fuller))
        loss = L2(sample_target, pred)

    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return loss

def fit(model, lowers, fullers, targets, checkpoints_folder="weights//ckpt", epochs=1, lr=1e-3):
    optimizer = tf.keras.optimizers.Adam(lr)
    for e in range(epochs):
        start = time.time()
        total_loss = []
        for ite, (sample_lower, sample_fuller, sample_target) in tf.data.Dataset.zip((lowers, fullers, targets)).enumerate():
            loss = train_step(model, sample_lower, sample_fuller, sample_target, optimizer)
            progress(e+1, (ite+1), len(lowers), loss=loss)
            total_loss.append(loss)
        end = time.time()
        print("\nEpoch {0}: ---------- Avg Loss: {1}, time exection: {2}".format(e+1, sum(total_loss) / len(total_loss), end - start))
            
        if (e+1) % 10 == 0:
            model.save_weights(checkpoints_folder)
            print("Model has been saved at epoch: {}.".format(e+1))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoints_folder', type=str, default="weights//ckpt")
    parser.add_argument('--pretrain_dir', type=str)
    parser.add_argument('--raw_path', type=str)
    parser.add_argument('--hdr_path', type=str)
    parser.add_argument('--low_size', type=int, default=256)
    parser.add_argument('--full_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)

    config = parser.parse_args()

    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    # Get Data
    lowers, fullers, targets = preprocess_data (config.raw_path, 
                                                config.hdr_path, 
                                                config.low_size, 
                                                config.full_size, 
                                                config.batch_size)

    # Create Model
    model = HDRPointwiseNN()
    
    # Build Model
    sample_lower = iter(lowers).next()
    sample_fuller = iter(fullers).next()
    model((sample_lower, sample_fuller))
    
    # Load pretrained weight
    if config.pretrain_dir:
        model.load_weights(config.pretrain_dir)

    # Train Model
    fit(model, lowers, fullers, targets, config.checkpoints_folder, config.epochs, config.lr)