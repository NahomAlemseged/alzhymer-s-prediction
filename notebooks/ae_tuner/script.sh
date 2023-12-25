#!/bin/bash

# Define hyperparamters:

lr=(1e-5 1e-4 1e-3 1e-2 1e-1)
n_hidden=(0 180 360 720 1080 1440)
n_latent=$(seq 10 10 360)
activ=("relu" "selu")

for i in ($seq 10 10 360);do
	for j in $lr;do
		for k in $n_hidden;do
			for l in $activ; do
				echo $i $j $k $l
			done
		done
	done
done 
