import scipy.ndimage as im
import numpy as np
import os
import h5py

def create_h5(data, label, path, file_name):

	file = h5py.File(os.path.join(path, file_name), 'w')
	file.create_dataset("data", data = data)
	file.create_dataset("label", data = label)
	file.close()

def patchify(image, size):
	num_patches = [image.shape[1]//size[0], image.shape[2]//size[1]]
	patches = np.zeros((num_patches[0]*num_patches[1], 3, size[0], size[1]))

	for i in range(num_patches[0]):
		for j in range(num_patches[1]):
			patches[i*num_patches[1] + j] = image[:, i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]]

	return patches

def create_dataset(factor, num_images, patch_size, dataset_path, destination_path, dataset_name):

	folder = "image_SRF_" + str(factor)
	lr_suffix = "_".join(["SRF", str(factor), "LR"])
	hr_suffix = "_".join(["SRF", str(factor), "HR"])

	lr_stack = []
	hr_stack = []

	for i in range(1, 1+num_images):
		lr_img = im.imread(os.path.join(dataset_path, folder, "img_" + str(i).zfill(3) + "_" + lr_suffix + ".png"))
		hr_img = im.imread(os.path.join(dataset_path, folder, "img_" + str(i).zfill(3) + "_" + hr_suffix + ".png"))

		lr_img = np.transpose(lr_img, (2, 0, 1))
		hr_img = np.transpose(hr_img, (2, 0, 1))

		lr_stack.append(patchify(lr_img, size = patch_size))
		hr_stack.append(patchify(hr_img, size = patch_size*factor))

	lr_set = np.concatenate(lr_stack)
	hr_set = np.concatenate(hr_stack)

	create_h5(data = lr_set, label = hr_set, path = destination_path, file_name = dataset_name)

	print("data of shape ", lr_set.shape, "and label of shape ", hr_set.shape, " created of type", lr_set.dtype)