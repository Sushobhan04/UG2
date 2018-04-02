import scipy.ndimage as im
import numpy as np
import os
import h5py
import random
import cv2
from pyblur.pyblur import LinearMotionBlur_random
from UG2.utils import data as data_utils

def hist_match(source, template):
	"""
	Adjust the pixel values of a grayscale image such that its histogram
	matches that of a target image

	Arguments:
	-----------
		source: np.ndarray
			Image to transform; the histogram is computed over the flattened
			array
		template: np.ndarray
			Template image; can have different dimensions to source
	Returns:
	-----------
		matched: np.ndarray
			The transformed output image
	"""

	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
											return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)

def gaussian_blur(inp, sigma = 1.0):
	return im.filters.gaussian_filter(inp, sigma)

def motion_blurred(inp):
	return LinearMotionBlur_random(inp)

def motionBlur3D(inp):
	nDims = inp.shape[0]
	imgMotionBlurred = np.empty(inp.shape)
	for dimIndex in range(nDims):
		img = inp[dimIndex,:]
		imgMotionBlurred[dimIndex,:] = motion_blurred(img)
	return imgMotionBlurred

def noisy(image, noise_typ):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy

	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
				for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	
	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy

def createBlurredDataset(nTK, factor, dataset_name, data_path, traindedModels_path):
	filename = [os.path.join(data_path, dataset_name+".h5")]
	for file in filename:
		with h5py.File(file,'r') as curr_data:
			blurredImageIndex = 0
			data  = np.array(curr_data['data']).astype(np.float32)
			label = np.array(curr_data['label']).astype(np.float32)
			numImages    = label.shape[0]
			dataBlurred  = np.empty([data.shape[0]*nTK,data.shape[1],data.shape[2],data.shape[3]])          
			labelBlurred = np.empty([label.shape[0]*nTK,label.shape[1],label.shape[2],label.shape[3]])          
			for imageIndex in range(numImages):
				img = label[imageIndex,:]
				for kernelIndex in range(nTK):                 
					imgGaussian = gaussian_blur(img, sigma = random.uniform(0, 2.5))              
					imgMotion   = motionBlur3D(imgGaussian)
					imgMotion   = np.transpose(imgMotion,(1,2,0))                 
					imgMotionResized = cv2.resize(imgMotion, (int(imgMotion.shape[0]/factor),int(imgMotion.shape[1]/factor)))
					imgMotionResized = np.transpose(imgMotionResized,(2,0,1))
					dataBlurred[blurredImageIndex,:]  = imgMotionResized
					labelBlurred[blurredImageIndex,:] = img
					blurredImageIndex= blurredImageIndex+1
	data_utils.create_h5(dataBlurred, labelBlurred, traindedModels_path, dataset_name+"BlurrednTK"+str(nTK)+".h5")
	print("Created the h5 dataset")       
	return(dataBlurred,labelBlurred)		