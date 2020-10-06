import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, padding='same', stride=1, use_bias=True, activation=tf.keras.activations.relu, batch_norm=False):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, use_bias=use_bias)
        self.batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.activation = activation if activation else None

    def call(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.keras.layers.ReLU(), use_bias=True, batch_norm=False):
        super(FC, self).__init__()

        self.fc = tf.keras.layers.Dense(units, use_bias=use_bias)
        self.batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else None
        self.activation = activation if activation else None

    def call(self, x):
        x = self.fc(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x
  
class Coeffs(tf.keras.layers.Layer):
    def __init__(self, batch_norm=False):
        super(Coeffs, self).__init__()
        # luminance bin
        self.lb = 8 
        # spatial bins
        self.sb = 16
        # Lower image size
        self.nsize = 256

        # Number of layers in extractor
        n_extractor_layers = int(np.log2(self.nsize/self.sb))
        
        # Features Extractor
        self.extractor = tf.keras.Sequential()
        for i in range(n_extractor_layers):
            use_batch_norm = batch_norm if i > 0 else False
            filters = 2**(i + 1) * self.lb
            self.extractor.add(ConvBlock(filters, stride=2, batch_norm=use_batch_norm))
    
        # Local Features Extractor
        self.local_extractor = tf.keras.Sequential()
        self.local_extractor.add(ConvBlock(8*self.lb, batch_norm=batch_norm))
        self.local_extractor.add(ConvBlock(8*self.lb, activation=None, batch_norm=False))

        # Global Feature Extractor
        n_global_layers = int(np.log2(self.sb/4))
        self.global_extractor = tf.keras.Sequential()
        for i in range(n_global_layers):
            self.global_extractor.add(ConvBlock(8*self.lb, 3, stride=2, batch_norm=batch_norm))

        self.global_extractor.add(tf.keras.layers.Flatten())
        self.global_extractor.add(FC(32*self.lb, batch_norm=batch_norm))
        self.global_extractor.add(FC(16*self.lb, batch_norm=batch_norm))
        self.global_extractor.add(FC(8*self.lb, activation=None, batch_norm=batch_norm))

        # Fusion activation
        self.relu = tf.keras.layers.ReLU()

        self.conv_out = ConvBlock(96, activation=None, batch_norm=False)

    def call(self, x):
        bs = x.shape[0]

        x = self.extractor(x)
        
        # Local Extractor
        local_features = self.local_extractor(x)
        # Global Extractor
        global_features = self.global_extractor(x)
        global_bs, global_units = global_features.shape
        global_features = tf.reshape(global_features, (global_bs, 1, 1, global_units))

        # Fusion 
        fusion = local_features + global_features
        x = self.relu(fusion)

        x = self.conv_out(x)
        x = tf.reshape(x, [bs, 12, self.lb, self.sb, self.sb])

        return x

class GuideNN(tf.keras.layers.Layer):
    def __init__(self, batch_norm=False):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(16, kernel_size=1, batch_norm=batch_norm)
        self.conv2 = ConvBlock(1, kernel_size=1,activation=tf.keras.activations.tanh, batch_norm=batch_norm)

    def call(self, x):
        return self.conv2(self.conv1(x))

class Slice(tf.keras.layers.Layer):
    def __init__(self):
        super(Slice, self).__init__()
    
    def call(self, A, guide):  
        bs, H, W, _ = guide.shape

        grid = tf.meshgrid(tf.linspace(-1, 1, num=W),
                        tf.linspace(-1, 1, num=H),
                        indexing='ij') 
        mesh = tf.reshape(tf.stack(grid, axis=-1), (-1, 2))[None]
        mesh = tf.repeat(mesh, 12, axis=0)
        mesh = tf.repeat(mesh[None], bs, axis=0)
        mesh = tf.cast(mesh, tf.float32)

        intensity = tf.reshape(guide, (bs, 1, W*H, 1))
        intensity = tf.repeat(intensity, 12, axis=1)
        intensity = tf.cast(intensity, tf.float32)

        guidemap = tf.concat([mesh, intensity], axis=-1)

        x_ref_min = [ -1, -1, -1 ]
        x_ref_max = [  1,  1,  1 ]
        coeffs = tfp.math.batch_interp_regular_nd_grid(guidemap, x_ref_min, x_ref_max, A, axis=-3)

        coeffs = tf.reshape(coeffs, (bs, 12, H, W))
        coeffs = tf.transpose(coeffs, [0,2,3,1])
        return coeffs

class ApplyCoeffs(tf.keras.layers.Layer):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
    
    def call(self, coeff, full_res_input):  
        '''
        Affine:
        r = a11*r + a12*g + a13*b + a14
        g = a21*r + a22*g + a23*b + a24
        b = a31*r + a32*g + a33*b + a34
        ...
        '''

        R = tf.reduce_sum(full_res_input * coeff[:, :, :, 0:3], axis=-1)[...,None] + coeff[:, :, :, 3:4]
        G = tf.reduce_sum(full_res_input * coeff[:, :, :, 4:7], axis=-1)[...,None]  + coeff[:, :, :, 7:8]
        B = tf.reduce_sum(full_res_input * coeff[:, :, :,  8:11], axis=-1)[...,None]  + coeff[:, :, :, 11:12]

        return tf.concat([R, G, B], axis=-1)

class HDRPointwiseNN(tf.keras.Model):
  def __init__(self):
    super(HDRPointwiseNN, self).__init__()
    self.coeffs = Coeffs()
    self.guideNN = GuideNN()
    self.slice = Slice()
    self.apply_coeffs = ApplyCoeffs()
  
  def call(self, input, training=False):
    lowers, fullers = input

    coeffs = self.coeffs(lowers)
    guide = self.guideNN(fullers)
    slice_coeffs = self.slice(coeffs, guide)
    out = self.apply_coeffs(slice_coeffs, fullers)

    return out

