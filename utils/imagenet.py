import numpy as np
import os
import h5py
import cv2
from UG2.utils import data as data_utils
from UG2.utils import image as image_utils
import copy
import json
from PIL import Image


def parse_imagenet(path, file, img_size = 16):
	data_file = os.path.join(data_folder, file)

	d = unpickle(data_file)
	x = d['data']
	y = d['labels']
	# mean_image = d['mean']

	x = x/np.float32(255)
	# mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0
	y = [i-1 for i in y]
	data_size = x.shape[0]

	# x -= mean_image

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

def parse_vatic_annotations(path, filename):
	txt_file = open(os.path.join(path, filename), "r").read().split("\n")
	bbox_list = []
	word_label = []
	frame_list = []

	for line in ann_arr:
		ann = line.split(" ")

		bbox_list.append([int(ann[1]), int(ann[2]), int(ann[3]), int(ann[4])])
		split_label = ann[9].strip("\"").split("_")
		word_label = "".join([x.title() for x in split_label])



def create_imagenet_dataset(imagenet_bbox, imagenet_labels, source_path, destination_path, file_name_prefix, size, buffer_size = 0, batch_size = 2000):
	count = 0
	data = []
	label = []

	random_index = np.arange(len(imagenet_bbox["wnids"]))

	np.random.shuffle(random_index)

	for i in range(len(imagenet_bbox["wnids"])):
		index = random_index[i]

		img_file = imagenet_bbox["wnids"][index]
		bbox = imagenet_bbox["bbox"][index]

		wnid = img_file.split("_")[0]

		img = cv2.imread(os.path.join(source_path, wnid, img_file+".JPEG"))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# bb = image_utils.calculate_bbox(bbox, img.shape[0:2], buffer_size = buffer_size)

		cropped_img = image_utils.crop_image(img, bbox, dim = size[0])
		final_img = cv2.resize(cropped_img, size)



		count += 1
		data.append(final_img)
		label.append(imagenet_labels.index(wnid))

		if count%batch_size == 0 and int(count/batch_size) > 0:
			print(count/batch_size)

			with h5py.File(os.path.join(destination_path, file_name_prefix + "_" + str(int(count/batch_size))+".h5"), "w") as file:
				file.create_dataset("data", data = np.array(data))
				file.create_dataset("label", data = np.array(label))

			data = []
			label = []

def imagenet2UG_labels(wnid = None, idx = None, path = "/data/UG2_data", return_type = "name"):
	ret = None

	with open(os.path.join(path, "UG2_labels.txt"), 'r') as file:
		mapping = json.load(file)

	if wnid is not None and idx is None :
		ret = mapping[wnid]

	elif idx is not None and wnid is None:
		with open(os.path.join(path, "imagenet_labels.txt"), 'r') as file:
			idx_map = json.load(file)

		ret = mapping[idx_map[idx]]

	if return_type == "idx":
		with open(os.path.join(path, "UG2_label_names.txt"), 'r') as file:
			UG2_names = json.load(file)


		ret = [UG2_names.index(x) for x in ret]


	return ret


def UG2imagenet_labels(name = None, idx = None, path = "/data/UG2_data", return_type = "name"):
	ret = None
	UG2_index = None

	with open(os.path.join(data_path, "UG2_to_imagenet_labels.txt"), 'r') as file:
		UG2_to_imagenet_labels = json.load(file)

	if name is not None and idx is None :
		with open(os.path.join(path, "UG2_label_names.txt"), 'r') as file:
			UG2_names = json.load(file)
		ug2_index = UG2_names.index(name)
		ret = UG2_to_imagenet_labels[UG2_index]

	elif idx is not None and wnid is None:
		ret = UG2_to_imagenet_labels[idx]

	if return_type == "name":
		with open(os.path.join(path, "imagenet_labels.txt"), 'w') as file:
			idx_map = json.load(file)

		ret = [idx_map[x] for x in ret]

	return ret


def create_imagenet_compressed(imagenet_bbox, imagenet_labels, source_path, destination_path, file_name_prefix, size, buffer_size = 0, batch_size = 2000, quality = 10):
	count = 0
	data = []
	label = []
	img_class = []

	random_index = np.arange(len(imagenet_bbox["wnids"]))

	np.random.shuffle(random_index)

	for i in range(len(imagenet_bbox["wnids"])):
		index = random_index[i]

		img_file = imagenet_bbox["wnids"][index]
		bbox = imagenet_bbox["bbox"][index]

		wnid = img_file.split("_")[0]

		img = cv2.imread(os.path.join(source_path, wnid, img_file+".JPEG"))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# bb = image_utils.calculate_bbox(bbox, img.shape[0:2], buffer_size = buffer_size)

		cropped_img = image_utils.crop_image(img, bbox, dim = size[0])
		resized_img = cv2.resize(cropped_img, size)

		label_img = np.copy(resized_img)

		pil_img = Image.fromarray(resized_img)
		pil_img.save("/home/susho/temp.jpg", "JPEG", quality=quality)


		final_img = cv2.imread("/home/susho/temp.jpg")
		final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

		label_img = np.transpose(label_img, (2, 0, 1))
		final_img = np.transpose(final_img, (2, 0, 1))


		count += 1
		data.append(final_img)
		label.append(label_img)
		img_class.append(imagenet_labels.index(wnid))

		if count%batch_size == 0 and int(count/batch_size) > 0:
			print(count/batch_size)

			with h5py.File(os.path.join(destination_path, file_name_prefix + "_" + str(int(count/batch_size))+".h5"), "w") as file:
				file.create_dataset("data", data = np.array(data))
				file.create_dataset("label", data = np.array(label))
				file.create_dataset("class", data = np.array(img_class))

			data = []
			label = []
			img_class = []






