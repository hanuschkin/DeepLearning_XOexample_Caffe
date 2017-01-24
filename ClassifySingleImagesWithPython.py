# passing single example images to the CNN
# option to same feature maps to file 
#
# A. Hanuschkin (2016)


import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

caffe_root = '/home/deeplearning/caffe_FAST/'  
sys.path.insert(0, caffe_root + 'python')
import caffe

do_plot_inputfig  = 1 # display the tested input image
do_plot_CNN_infos = 0 # export feature maps 


# load the model
# usa a deploy prototxt w/o input output definition! Otherwise input is read from validation list (overwriting data layer)
# http://stackoverflow.com/questions/29124840/prediction-in-caffe-exception-input-blob-arguments-do-not-match-net-inputs#31391076
net = caffe.Net('ANet_3conv_deploy.prototxt','ANet_3conv_1strun_snapshot_iter_3000.caffemodel', caffe.TEST);

# test for a couple of input images
for jj in range(3):
		## Data in the input data is in net.blobs['data'].data
		shouldbe = 'cross'
                img = cv2.imread('./allfiles/cr1000.bmp',0)
		if jj==0:
			shouldbe = 'circle'
			img = cv2.imread('./allfiles/ci1000.bmp',0)
		if jj==1:
                        shouldbe = 'circle'
  			img = cv2.imread('./allfiles/ci100.bmp',0)
                if jj==2:
                        shouldbe = 'cross'
                        img = cv2.imread('./allfiles/cr100.bmp',0)


		img_input = np.zeros((1,3,116,116))
		for i in range(3):
		 img_input[:,i,:,:] = img[np.newaxis,np.newaxis, :, :]

		net.blobs['data'].data[...] = img_input

		if do_plot_inputfig:
		  plt.figure
		  plt.pcolor( net.blobs['data'].data[0,0,:,:])
		  plt.title('DATA input')
		  plt.show
		  plt.pause(1)

		print "#################################"
		# propagete the input data 
		out = net.forward(data=img_input)

                # check network prediction
                print 'label is ',shouldbe,'\t',
                if out['ip1'][0][0] > out['ip1'][0][1]:
                 print " - predicted label circle"
                else:
                 print " - predicted label cross"
		
		# plot features/weights and feature maps
		if do_plot_CNN_infos:
		 # print input data 
                 cv2.imwrite('InputData.jpg', img) 
	
		 # print the fetaure maps of the first conv layer 
		 for i in range(16):
		  cv2.imwrite('FeatureMaps_conv1_' + str(i) + '.jpg', 255*net.blobs['conv1'].data[0,i])

		 for i in range(32):
		  cv2.imwrite('FeatureMaps_conv2_' + str(i) + '.jpg', 255*net.blobs['conv2'].data[0,i])

		 print "shape of conv1 feature maps", net.blobs['conv1'].data.shape 
		 print "shape of conv2 feature maps", net.blobs['conv2'].data.shape
		 print "shape of conv3 feature maps", net.blobs['conv3'].data.shape

		 ## now get the weights... 
		 print "shape of conv1 features", net.params['conv1'][0].data.shape 
		 print "shape of conv2 features", net.params['conv2'][0].data.shape
		 print "shape of conv3 features", net.params['conv3'][0].data.shape

