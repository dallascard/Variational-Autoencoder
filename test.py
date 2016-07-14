import numpy as np
import time
import os
from VAE import VAE
from dVAE import dVAE
import cPickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm


hu_encoder = 400
hu_decoder = 400
n_latent = 10
continuous = False

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
model = dVAE(continuous, hu_encoder, hu_decoder, n_latent, x_train)

print "Loading paramters"
model.load_parameters(path)

dim_sq = x_train[0].size
width = int(np.sqrt(dim_sq))

print "Doing predictions"
for i in range(10):
    z = np.zeros(n_latent, dtype=np.float32)
    #z_random = np.array(np.random.randn(n_latent), dtype=np.float32)
    z[2] = (i-5)/2.0
    x = model.decode(z)

    plt.imshow(np.reshape(x, (width, width)), cmap='Greys')
    filename = 'test' + str(i) + '.png'
    plt.savefig(filename)

for i in range(n_latent):
    z = np.zeros(n_latent, dtype=np.float32)
    #z_random = np.array(np.random.randn(n_latent), dtype=np.float32)
    z[i] = 100
    x = model.get_class_exemplar(z)

    plt.imshow(np.reshape(x, (width, width)), cmap='Greys')
    filename = 'exemplar' + str(i) + '.png'
    plt.savefig(filename)

