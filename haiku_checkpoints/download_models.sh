#!/bin/bash
#Optical Flow 
wget https://storage.googleapis.com/perceiver_io/optical_flow_checkpoint.pystate

#Image Classification
wget https://storage.googleapis.com/perceiver_io/imagenet_conv_preprocessing.pystate
wget https://storage.googleapis.com/perceiver_io/imagenet_fourier_position_encoding.pystate
wget https://storage.googleapis.com/perceiver_io/imagenet_learned_position_encoding.pystate

#Masked Language Modelling
wget https://storage.googleapis.com/perceiver_io/language_perceiver_io_bytes.pickle

#Video Autoencoding
wget https://storage.googleapis.com/perceiver_io/video_autoencoding_checkpoint.pystate