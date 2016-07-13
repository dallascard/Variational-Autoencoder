import numpy as np
import time
import os
from VAE import VAE
import cPickle
import gzip


hu_encoder = 400
hu_decoder = 400
n_latent = 20
continuous = False
n_epochs = 40

if continuous:
    print "Loading Freyface data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = open('freyfaces.pkl', 'rb')
    x = cPickle.load(f)
    f.close()
    x_train = x[:1500]
    x_valid = x[1500:]
else:
    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()

path = "./"

print "instantiating model"
model = VAE(continuous, hu_encoder, hu_decoder, n_latent, x_train)

model.load_parameters(path)

x = model.decoder(x_train[0], np.random.randn(n_latent))

print x


