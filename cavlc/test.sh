#!/bin/bash
for i in {4..4}
do
	./main /home/dong/Documents/MATLAB/ml_v7/8bit_img/ori/$i-  /home/dong/Documents/MATLAB/ml_v7/8bit_img/decode/$i-  test-
done 
tar cvf test.tar.gz test-[0-9].dat

