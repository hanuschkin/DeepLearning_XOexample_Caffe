# DeepLearning_XOexample_Caffe
A brief and illustrative example how convolutional neural networks (CNNs) work (Caffe implementation):  
Two simple image categories (X and O images)

The example is inspired by Brandon Rohrer's 'Data Science and Robots' blog. 

Please visit the following web pages for more details
http://www.optophysiology.uni-freiburg.de/Research/research_DL/CNNsWithMatlabAndCaffe
http://brohrer.github.io/how_convolutional_neural_networks_work.html

###
0) check for GPU 
nvidia-smi

1) train the network 
~/caffe_FAST/build/tools/caffe train --solver=ANet_3conv_solver.prototxt --gpu 0 

2) test the network performance 
~/caffe_FAST/build/tools/caffe test --model=ANet_3conv.prototxt --weights=ANet_3conv_1strun_snapshot_iter_5000.caffemodel --gpu 0 --iterations 10

3) run a Python script to pass single images to the trained network 
python ClassifySingleImagesWithPython.py 
