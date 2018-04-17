import scipy.ndimage as im
import numpy as np
import os
import h5py
import cv2
from UG2.lib.pyblur import LinearMotionBlur
from UG2.utils import data as data_utils
import copy
import random 

def hist_match_grey(source, template):
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

def hist_match(source, template):
	equalized_img = []

	for channel in range(source.shape[0]):
		equalized_img.append(hist_match_grey(source[channel], template[channel]))

	return np.array(equalized_img)

def gaussian_blur(inp, sigma = (1.0, 1.0, 0.0)):
	temp_img = im.filters.gaussian_filter(inp, sigma)

	return temp_img

def motionBlur3D(inp, dim, theta, linetype):
	imgMotionBlurred = np.empty(inp.shape)
	for dimIndex in range(inp.shape[2]):
		img = inp[:,:,dimIndex]
		imgMotionBlurred[:,:,dimIndex] = LinearMotionBlur(img, dim, theta, linetype)
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

def blur_images(images, nTK, scale_factor, flags = [1, 1], gaussian_blur_range = (0, 1)):
	output_data  = []
	output_label = []
	for image in images:
		for kernelIndex in range(nTK):
			temp_image = np.copy(image)
			if flags[0]:
				sigmaRandom = np.random.uniform(gaussian_blur_range[0], gaussian_blur_range[1]) 
				temp_image  = gaussian_blur(temp_image, sigma = (sigmaRandom, sigmaRandom,0))  

			if flags[1]:
				dim         = np.random.choice([3, 5, 7, 9], 1)
				theta       = np.random.uniform(0.0, 360.0)
				temp_image  = motionBlur3D(temp_image, dim[0], theta, "full")

			if scale_factor != 1:
				temp_image  = cv2.resize(temp_image, (0, 0), fx = 1.0/scale_factor, fy = 1.0/scale_factor)
			
			output_data.append(np.transpose(temp_image,(2,0,1)))
			output_label.append(np.transpose(image,(2,0,1)))

	return output_data, output_label


def calculate_bbox(box, size, buffer_size = 0):
	center = [(box[0] + box[2])//2, (box[1] + box[3])//2]
	
	dim = np.array([box[2] - box[0], box[3] - box[1]]) + np.array([buffer_size, buffer_size])
	
	# xmin = max(0, center[0] - dim//2)
	# ymin = max(0, center[1] - dim//2)
	# xmax = min(size[1], center[0] + dim//2)
	# ymax = min(size[0], center[1] + dim//2)
	
	xmin = center[0] - dim//2
	ymin = center[1] - dim//2
	xmax = center[0] + dim//2
	ymax = center[1] + dim//2
	
#     print(center, dim)
	
	return [xmin, ymin, xmax, ymax]

def crop_image(img, box, dim = 224):
	img = np.copy(img)

	if img.shape[0] == 3:
		size = img.shape[1:3]
	else:
		size = img.shape[0:2]
		
	center = [(box[0] + box[2])//2, (box[1] + box[3])//2]
	box_size = [box[2] - box[0], box[3] - box[1]]

	roi = dim

	if box_size[0] > dim or box_size[1] > dim:
		roi = max(box_size[0], box_size[1])

		if roi > size[0] or roi > size[1]:
			roi = min(size[0], size[1])

	# get new xmin and ymin
	if center[0] - roi//2 >= 0:
		if center[0] + roi//2 <= size[1]:
			xmin = center[0] - roi//2
		else:
			xmin = size[1] - roi
	else:
		xmin = 0

	if center[1] - roi//2 >= 0:
		if center[1] + roi//2 <= size[0]:
			ymin = center[1] - roi//2
		else:
			ymin = size[0] - roi
	else:
		ymin = 0

	# print(xmin, ymin, xmax, ymax, dim)

	ymax = min(size[0], ymin + roi)
	xmax = min(size[1], xmin + roi)

	if img.shape[0] == 3:
		final_img = img[:, ymin:ymin + roi, xmin: xmin + roi]
	else:
		final_img = img[ymin:ymin + roi, xmin: xmin + roi]

	return final_img

def resize_bin(img, bins):
	dim = img.shape[0]
	selected_b = None

	for b in bins:
		if dim <= b:
			selected_b = b
			break

		selected_b = bins[-1]


	final_img = cv2.resize(img, (selected_b, selected_b))

	return final_img, bins.index(selected_b)
