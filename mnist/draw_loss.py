#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from pylab import *
import matplotlib.pyplot as plt

import caffe

#caffe.set_device(0)
#caffe.set_mode_gpu()

solver = caffe.SGDSolver('../examples/mnist/lenet_solver.prototxt')

niter = 10000

display_iter = 100

test_iter = 100

test_interval = 500

# train loss
train_loss = zeros(ceil(niter * 1.0 / display_iter))
# test loss
test_loss = zeros(ceil(niter * 1.0 / test_interval))
# test accuracy
test_acc = zeros(ceil(niter * 1.0 / test_interval))

solver.step(1)

_train_loss = 0; _test_loss = 0; _accuracy = 0

for it in range(niter):
    
    solver.step(1)
    
    _train_loss += solver.net.blobs['loss'].data
    if it % display_iter == 0:
        
        train_loss[it // display_iter] = _train_loss / display_iter
        _train_loss = 0

    if it % test_interval == 0:
        for test_it in range(test_iter):
           
            solver.test_nets[0].forward()
            
            _test_loss += solver.test_nets[0].blobs['loss'].data
            
            _accuracy += solver.test_nets[0].blobs['accuracy'].data
      
        test_loss[it / test_interval] = _test_loss / test_iter
        
        test_acc[it / test_interval] = _accuracy / test_iter
        _test_loss = 0
        _accuracy = 0


print '\nplot the train loss and test accuracy\n'
_, ax1 = plt.subplots()
ax2 = ax1.twinx()

# train loss 
ax1.plot(display_iter * arange(len(train_loss)), train_loss, 'g')
# test loss
ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')
# test accuracy
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
savefig('myloss.png')
