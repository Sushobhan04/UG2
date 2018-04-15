import scipy.ndimage as im
import numpy as np
import os
import h5py
import torch
from torch.autograd import Variable
import pickle
from torch.utils.data import Dataset, DataLoader
import pickle
from UG2.utils import image as image_utils
import xml.etree.ElementTree as ET
import cv2
import glob
import json

class DatasetFromFile(Dataset):
	def __init__(self, path, data_file, img_size = None, data_format = "h5", transform = None):
		super(DatasetFromFile, self).__init__()
		self.transform = transform
		self.img_size = img_size
		self.data_format = data_format

		if data_format == "h5":
			self.data, self.label = self.load_h5_data(path, data_file)
		elif data_format == "binary":
			self.data, self.label = self.load_binary_data(path, data_file)
		elif data_format == "h5_bbox":
			self.data, self.label = self.load_h5_bbox_data(path, data_file)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if self.data_format == "h5_bbox":
			return {"data": self.data[idx], "label": self.label[idx]}
		else:
			return {"data": self.data[idx], "label": self.label[idx]}

	def load_binary_data(self, path, data_file):
		data_file = os.path.join(path, data_file)

		d = unpickle(data_file)
		x = d['data']
		y = d['labels']
	#     mean_image = d['mean']

		x = x/np.float32(255)
	#     mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0

		y_one_hot = np.zeros((len(y), 1000), dtype = np.float32)

		for i in range(len(y)):
			y_one_hot[i, y[i]-1] = 1.0
		data_size = x.shape[0]

	#     x -= mean_image

		img_size2 = self.img_size * self.img_size

		x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
		x = x.reshape((x.shape[0], self.img_size, self.img_size, 3)).transpose(0, 3, 1, 2)
		
		return x,y_one_hot

	def load_h5_data(self, path, data_file):

		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			data = np.array(curr_data['data'])
			label = np.array(curr_data['label'])

		# print np.max(data), np.max(label)

		if data.dtype == np.uint8:
			data = data.astype(np.float32)/255.0
			label = label.astype(np.float32)/255.0

		return data, label

	def load_h5_bbox_data(self, path, data_file):
		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			label = np.array(curr_data['label'])
			bbox = np.array(curr_data['bbox'])

			np_data = np.array(curr_data['data'])
			np_data = np.transpose(np_data, (0, 3, 1, 2))

			if np_data.dtype == np.uint8:
				np_data = np_data.astype(np.float32)/255.0

			crop_data = []

			for i,bb in enumerate(bbox):
				crop_data.append(np_data[i, :, bb[1]:bb[3], bb[0]:bb[2]])

		label_one_hot = np.zeros((label.shape[0], 48), dtype = np.float32)

		for i in range(label.shape[0]):
			label_one_hot[i, label[i]] = 1.0

		return crop_data, label_one_hot

class ImagenetDataset(Dataset):
	def __init__(self, path, data_file, img_size, data_format = "h5", transform = None):
		super(ImagenetDataset, self).__init__()
		self.transform = transform
		self.img_size = img_size

def unpickle(file):
	with open(file, 'rb') as fo:
	  dict = pickle.load(fo)
	return dict

def convert_to_torch_tensor(tensor, cuda = True, from_numpy = True, requires_grad = False, dtype = "float32"):
	if from_numpy:
		if dtype == "float32":
			tensor = torch.FloatTensor(tensor)
		elif dtype == "int64":
			tensor = torch.LongTensor(tensor)

	if cuda:
		tensor = tensor.cuda()

	tensor = Variable(tensor)

	if requires_grad:
		tensor.requires_grad = True

	return tensor

def create_h5(data, label, path, file_name):

	with h5py.File(os.path.join(path, file_name), 'w') as file:
		file.create_dataset("data", data = data)
		file.create_dataset("label", data = label)

def patchify(image, size):
	num_patches = [image.shape[1]//size[0], image.shape[2]//size[1]]
	patches = np.zeros((num_patches[0]*num_patches[1], 3, size[0], size[1]))

	for i in range(num_patches[0]):
		for j in range(num_patches[1]):
			patches[i*num_patches[1] + j] = image[:, i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]]

	return patches

def create_bsd_dataset(factor, num_images, patch_size, dataset_path, destination_path, dataset_name):

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

def create_dataset(data_source_path, source_name_files, image_format, destination_path, dataset_name, num_files, patch_size, blur_parameters):

	hr_image = []
	if image_format == ".png":
		for i in range(num_files):
			image_name          = os.path.join(data_source_path,source_name_files[i]+image_format)
			hr_image.append(im.imread(image_name))
	else:
		with h5py.File(os.path.join(data_source_path, source_name_files+image_format),'r') as file:
			data = np.array(file["data"])
			hr_image.extend(data)
	num_images = len(hr_image)
	
	print("Number of images in the dataset: "+str(num_images))
	
	gen_data, gen_label = image_utils.blur_images(hr_image, blur_parameters["nTK"] ,blur_parameters["scale_factor"], blur_parameters["flags"], blur_parameters["gaussian_blur_range"])

	lr_stack = []
	hr_stack = []

	for i in range(num_images):
		lr_stack.append(patchify(gen_data[i], size = patch_size))
		hr_stack.append(patchify(gen_label[i], size = patch_size*blur_parameters["scale_factor"]))
	lr_set = np.concatenate(lr_stack)
	hr_set = np.concatenate(hr_stack)

	lr_set_training = lr_set.astype(np.uint8)
	hr_set_training = hr_set.astype(np.uint8)

	create_h5(data = lr_set_training, label = hr_set_training, path = destination_path, file_name = dataset_name+".h5")
	print("data of shape ", lr_set_training.shape, "and label of shape ", hr_set_training.shape, " created of type", lr_set_training.dtype)

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def parse_imagenet(path, file, img_size = 16):
	data_file = os.path.join(data_folder, file)

	d = unpickle(data_file)
	x = d['data']
	y = d['labels']
#     mean_image = d['mean']

	x = x/np.float32(255)
#     mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0
	y = [i-1 for i in y]
	data_size = x.shape[0]

#     x -= mean_image

	img_size2 = img_size * img_size

	x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
	x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
	
	return x,y


def parse_imagenet_bbox(imagenet_wnids, path):

	imagenet_bbox = {"wnids": [], "bbox": []}

	for wnid in imagenet_wnids:
		for file in os.listdir(os.path.join(path, wnid)):
			img_file = file.split(".")[0]
			e = ET.parse(os.path.join(path, wnid, file)).getroot()
			
			for bbox in e.iter('bndbox'):
				xmin = int(bbox.find('xmin').text)
				ymin = int(bbox.find('ymin').text)
				xmax = int(bbox.find('xmax').text)
				ymax = int(bbox.find('ymax').text)
				
				imagenet_bbox["wnids"].append(img_file)
				imagenet_bbox["bbox"].append([xmin, ymin, xmax, ymax])

	return imagenet_bbox

def create_imagenet_dataset(imagenet_bbox, imagenet_labels, source_path, destination_path, file_name_prefix, bins, buffer_size = 0, batch_size = 2000):
	count = np.zeros(len(bins), dtype = np.int)
	data = [[] for i in range(len(bins))]
	label = [[] for i in range(len(bins))]

	file = None
	for img_file, bbox in zip(imagenet_bbox["wnids"], imagenet_bbox["bbox"]):
		wnid = img_file.split("_")[0]

		img = cv2.imread(os.path.join(source_path, wnid, img_file+".JPEG"))
		bb = image_utils.calculate_bbox(bbox, img.shape[0:2], buffer_size = buffer_size)

		cropped_img = image_utils.crop_image(img, bb)
		final_img, bin_index = image_utils.resize_bin(crop_image, bins)



		count[bin_index] = count[bin_index] + 1
		data[bin_index].append(final_img)
		label[bin_index].append(imagenet_labels.index(wnid))

		if count[bin_index]%batch_size == 0 and int(count[bin_index]/batch_size) > 0:
			print(count[bin_index]/batch_size)

			with h5py.File(os.path.join(destination_path, file_name_prefix + "_bin_" + str(bins[bin_index]) + str(int(count[bin_index]/batch_size))+".h5"), "w") as file:
				file.create_dataset("data", data = np.array(data[bin_index]))
				file.create_dataset("label", data = np.array(label[bin_index]))
				file.close()

			data[bin_index] = []
			data[bin_index] = []

		data.create_dataset(str(count%max_images), data = final_img)
		label.append(imagenet_labels.index(wnid))

	file.close()

def index_to_labels(index, data_path = "/data/UG2_data/", file1 = "UG2_label_names.txt", file2 = "imagenet_to_UG2_labels.txt"):
	# Read the label_number
	# Map to label_name
	with open(os.path.join(data_path, file2), 'r') as outfile:
		imagenet_to_UG2_labels = list(json.load(outfile))    
	imageNetIndex = imagenet_to_UG2_labels[index]
	if imageNetIndex == -1:
		print("UG2Class not found in list")
	with open(os.path.join(data_path, file1), 'r') as outfile:
		UG2_set_labels = list(json.load(outfile))    
	UG2label = UG2_set_labels[imageNetIndex]
	return UG2label

def create_classifier_labels( source_path, source_file, destination_path, destination_file):
	
	with h5py.File(os.path.join(source_path, source_file),'r') as file:
		data = np.array(file["data"])
		label = np.array(file["label"])
		class_idx  = np.array(file["class"])

	files = glob.glob(destination_path+"*")
	for f in files:
		os.remove(f)
	with open(os.path.join(destination_path,"Image2labelMapping.txt"),'w') as f:
		for i,img in enumerate(data):
			cv2.imwrite(os.path.join(destination_path,destination_file+str(i)+".png"),img)	
			label_name = index_to_labels(class_idx[i])
			image_label = destination_file+str(i)+"\t"+label_name
			if i!= class_idx.shape[0]:
				image_label = image_label+"\n"
			f.write(image_label)	


def create_labeled_blurry_dataset(source_path, source_file, destination_path, destination_file, blur_parameters):

	hr_image    = []
	label_image = []

	print("Path of input file: "+os.path.join(source_path, source_file))
	with h5py.File(os.path.join(source_path, source_file),'r') as file:
		data = np.array(file["data"])
		label_index  = np.array(file["label"])
		hr_image.extend(data)
		label_image.extend(label_index)

	num_images = len(hr_image)
	print("Number of images in the dataset: "+str(num_images))

	gen_data, gen_label = image_utils.blur_images(hr_image, blur_parameters["nTK"] ,blur_parameters["scale_factor"], blur_parameters["flags"], blur_parameters["gaussian_blur_range"])
	gen_data = np.asarray(gen_data)

	data_blurry     = gen_data.astype(np.uint8)
	data_blurry     = data_blurry.transpose(0, 2, 3, 1)
	label_blurry    = np.asarray(label_image)

	create_h5(data = data_blurry, label = label_blurry, path = destination_path, file_name = destination_file)
	print("Created file at: "+os.path.join(destination_path, destination_file))	
	print("data of shape ", data_blurry.shape, "and label of shape ", label_blurry.shape, " created of type", data_blurry.dtype)


def create_mixed_dataset(source_path, source_files, destination_path, destination_file):

	hr_image      = []
	label_image   = []
	for source_file in source_files:
		print("Path of input file: "+os.path.join(source_path, source_file))
		with h5py.File(os.path.join(source_path, source_file),'r') as file:
			data = np.array(file["data"])
			label_index  = np.array(file["label"])
			hr_image.extend(data[range(100)])
			label_image.extend(label_index[range(100)]) 

	hr_image        = np.asarray(hr_image)		
	data_blurry     = hr_image.astype(np.uint8)
	label_blurry    = np.asarray(label_image)

	create_h5(data = data_blurry, label = label_blurry, path = destination_path, file_name = destination_file)
	print("Created file at: "+os.path.join(destination_path, destination_file))	
	print("data of shape ", data_blurry.shape, "and label of shape ", label_blurry.shape, " created of type", data_blurry.dtype)