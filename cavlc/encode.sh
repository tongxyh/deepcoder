#!/bin/bash
for i in {1..5}
do
	for j in {1..8}
	do 
		./main /home/dong/Documents/MATLAB/ml_v7/8bit_img/ori/$i-$j.bmp  /home/dong/Documents/MATLAB/ml_v7/8bit_img/decode/$i-$j.bmp  v7-$i-$j-encode.dat
	done
done 

