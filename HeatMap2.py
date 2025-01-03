import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import os

from utils.config_utils import load_yaml
from models.builder import MODEL_GETTER #, BACKBONE_GETTER
from utils.costom_logger import timeLogger #, logger

warnings.simplefilter("ignore") # ignore warnings

#To transform this code in a Class - Heatmap
	
def simple_grad_cam(features, classifier, target_class): #features = features, classifier = classifier, target_class = args.target_class
	"""
	calculate gradient map. (simple version)
	"""
	features = nn.Parameter(features) #features = features	

	logits = torch.matmul(features, classifier) #torch,matmul is used to perform matrix multiplication between two tensors.	
	
	logits[0, :, :, target_class].sum().backward() #backward() is used to perform backpropagation in PyTorch. The gradients are computed for each trainable parameter in the model.
	features_grad = features.grad[0].sum(0).sum(0).unsqueeze(0).unsqueeze(0) #features_grad is used to store the gradients of the features with respect to the loss. The gradients are computed for each trainable parameter in the model.
	gramcam = F.relu(features_grad * features[0]) #F.relu is used to apply the ReLU activation function to the features_grad tensor. The ReLU activation function is used to introduce non-linearity to the model.
	gramcam = gramcam.sum(-1) #sum(-1) is used to sum the values in the last dimension of the gramcam tensor.
	gramcam = (gramcam - torch.min(gramcam)) / (torch.max(gramcam) - torch.min(gramcam)) #normalization

	return gramcam


def get_heat(model, img, original_class, data_size, threshold): #original_class is the class of the images it is extracted in main, and passed to the function get_heat
	# only need forward backbone
	#original_class = original_class #args.target_class #original_class is the class of the images it is extracted in main, and passed to the function get_heat
	with torch.no_grad(): #torch.no_grad() is used to deactivate the gradient calculation feature in PyTorch. It is used to reduce the memory usage and speed up the computations.
		outs = model.forward_backbone(img.unsqueeze(0)) #
	
	features = []
	for name in outs: #
		features.append(outs[name][0]) #append() is used to add the elements to the end of the list.

	layer_weights = [8, 4, 2, 1] #layer_weights is used to store the weights of the layers. The weights are used to calculate the weighted sum of the gradients of the features with respect to the loss.
	heatmap = np.zeros([data_size, data_size, 3]) #heatmap is used to store the heatmap. The heatmap is used to visualize the regions of the image that are important for the classification of the image.	
	for i in range(len(features)):
		f = features[i]
		f = f.cpu()
		if len(f.size()) == 2:
			S = int(f.size(0) ** 0.5)
			f = f.view(S, S, -1)

		# if you use original backbone without our module, 
		# please set classifier to your model's classifier. (e.g. model.classifier)
		gramcam = simple_grad_cam(f.unsqueeze(0), classifier=torch.ones(f.size(-1), 200)/f.size(-1), target_class=original_class) #target_class is the class of the images it is extracted in main, and passed to the function get_heat, it was passed using args.target_class
		gramcam = gramcam.detach().numpy()
		gramcam = cv2.resize(gramcam, (data_size, data_size))
		# heatmap colour : red
		heatmap[:, :, 2] += layer_weights[i] * gramcam

	heatmap = heatmap / sum(layer_weights)
	heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
	heatmap[heatmap < threshold] = 0 # threshold
	heatmap *= 255
	heatmap = heatmap.astype(np.uint8)

	return heatmap

def generate_heatmap(args, image, original_class, subdir_path, image_name_count):
	data_size = args.data_size
	threshold = args.threshold
	#print(f'Image: {image}')
	#input()
	#imagen = image #args.img #args.img _ vem do main
	#original_classs = original_class #args.target_class #original_class is the class of the images it is extracted in main, and passed to the function get_heat
	#args = parser.parse_args() #
	load_yaml(args, args.c) #args.c _ vem do main

	assert args.pretrained_heat != "" #assert is used to check if a condition is true or false. If the condition is true, the program will continue to execute normally. If the condition is false, the program will raise an AssertionError exception.

	model = MODEL_GETTER[args.model_name]( #
		use_fpn = args.use_fpn,
		fpn_size = args.fpn_size,
		use_selection = args.use_selection,
		num_classes = args.num_classes,
		num_selects = args.num_selects,
		use_combiner = args.use_combiner,
	) # about return_nodes, we use our default setting

	### load model
	checkpoint = torch.load(args.pretrained_heat, map_location=torch.device('cpu')) #checkpoint is used to store the weights of the model. The weights are used to initialize the model.
	model.load_state_dict(checkpoint['model'	]) #load_state_dict() is used to load the weights of the model. The weights are loaded from the checkpoint.
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #args.device is used to store the device on which the model is loaded. The device is used to run the model.
	model.to(args.device) #to() is used to move the model to the device.

	### read image and convert image to tensor
	img_transforms = transforms.Compose([
			transforms.Resize((510, 510), Image.BILINEAR),
			transforms.CenterCrop((args.data_size, args.data_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	])

	img = image #reading  #args.img) #args.img _ vem do main
	img = img[:, :, ::-1]

	#img = image.cpu().numpy()  # Convert the tensor back to a NumPy array
	#img = image.transpose((1, 2, 0))  # Change the order of dimensions
	#img = image[0] #Convert the values back to the 0-255 range
	#img = (img * 255).astype(np.uint8)  # Convert the values back to the 0-255 range

	#create a 3-channel image
	#img = np.stack([img.img, img], axis=-1)

	#Ensure that the image has 3 channels
	if img.shape[2] == 1:
		img = np.repeat(img, 3, axis=2)

	# to PIL.Image
	img = Image.fromarray(img)
	img = img_transforms(img)
	img = img.to(args.device)

	# get heatmap and original image
	heatmap = get_heat(model, img, original_class, data_size, threshold) #loading model, image and original_class
	
	rgb_img = image #cv2.imread(image) #args.img)
	rgb_img = cv2.resize(rgb_img, (510, 510))
	pad_size = (510 - args.data_size) // 2
	rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

	mix = rgb_img * 0.5 + heatmap * 0.5
	mix = mix.astype(np.uint8)

	#if args.save_img != "":
	#cv2.imwrite(os.path.join(subdir_path, "heatmap_" + img), heatmap) #.cpu().numpy().astype(np.uint8)) #to check if necessary use / or not
	#cv2.imwrite(os.path.join(subdir_path, "heatmap_" + img), heatmap.cpu().numpy().astype(np.uint8))
	#print(f'subdir_path: {subdir_path}')
	#input()
	#print(f'img: {img}')
	#input()

	#print(f'subdir_path: {subdir_path}')
	#input()
	## Replace 0o777 with the desired permission mode
	os.chmod(subdir_path, 0o777) #0o777 is used to set the permissions of the directory. The permissions are set to read, write, and execute for the owner, group, and others.	 
	cv2.imwrite(os.path.join(subdir_path, "heatmap_" + str(image_name_count) + ".jpg"), heatmap) #.cpu().numpy().astype(np.uint8)) #to check if necessary use / or not

	#cv2.imwrite(os.path.join(subdir_path, "/heatmap_" + img), heatmap.cpu().numpy().astype(np.uint8).transpose((1, 2, 0)))

	cv2.imwrite(os.path.join(subdir_path, "rbg_img_" + str(image_name_count) + ".jpg"), rgb_img) #args.save_img
	cv2.imwrite(os.path.join(subdir_path, "mix_" + str(image_name_count) + ".jpg"), mix) #.astype(np.uint8)) #args.save_img



# if __name__ == "__main__":

# 	parser = argparse.ArgumentParser("PIM-FGVC Heatmap Generation") #argparse is used to parse the arguments passed to the script when it is executed.	
# 	parser.add_argument("--c", default="", type=str) #c is used to store the path to the config file. The config file is used to store the hyperparameters of the model.	
# 	parser.add_argument("--img", default="", type=str)
# 	parser.add_argument("--target_class", default=0, type=int) #targer_class is used to store the target class. The target class is the class for which the heatmap is generated.	
# 	parser.add_argument("--threshold", default=0.75, type=float) #threshold is used to store the threshold value. The threshold value is used to remove the noise from the heatmap.	
# 	parser.add_argument("--save_img", default="", type=str, help="save path") #save_img is used to store the path to the directory where the heatmap is saved.
# 	parser.add_argument("--pretrained", default="", type=str) #pretrained is used to store the path to the pretrained model. The pretrained model is used to generate the heatmap.
# 	parser.add_argument("--model_name", default="swin-t", type=str, choices=["swin-t", "resnet50", "vit", "efficient"]) #model_name is used to store the name of the model. The name of the model is used to load the model.	
# 	args = parser.parse_args()

# 	assert args.c != "", "Please provide config file (.yaml)" #

# 	args = parser.parse_args() #
# 	load_yaml(args, args.c)

# 	assert args.pretrained != ""

# 	model = MODEL_GETTER[args.model_name]( #
# 		use_fpn = args.use_fpn,
# 		fpn_size = args.fpn_size,
# 		use_selection = args.use_selection,
# 		num_classes = args.num_classes,
# 		num_selects = args.num_selects,
# 		use_combiner = args.use_combiner,
# 	) # about return_nodes, we use our default setting

# 	### load model
# 	checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu')) #checkpoint
# 	model.load_state_dict(checkpoint['model'])
# 	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	model.to(args.device)

# 	### read image and convert image to tensor
# 	img_transforms = transforms.Compose([
# 			transforms.Resize((510, 510), Image.BILINEAR),
# 			transforms.CenterCrop((args.data_size, args.data_size)),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# 	])
# 	img = cv2.imread(args.img) #args.img _ vem do main
# 	img = img[:, :, ::-1] # BGR to RGB.  #Ocorreu erro de 'None Type' - acces an index or a key of a variable that is a variable that is of the None type data type - occurs when a method or a function return a None rather than the desired value  - TEM Q COLOCAR O NOME CORRETO DA IMAGEM NA NO TERMINAL QUANDO FOR CARREGAR A IMAGEM PARA FAZER O HEAT MAP - TEM Q VERIFICAR COMO FAZER TODAS AS ETAPAS EM AUTOMATICO - NO MODELO DISPONIVEL ESTA PASSO A PASSO - TREINA - VALIDA (EVAL) - HEAT MAP - INFER
	
# 	# to PIL.Image
# 	img = Image.fromarray(img)
# 	img = img_transforms(img)
# 	img = img.to(args.device)

# 	# get heatmap and original image
# 	heatmap = get_heat(model, img)
	
# 	rgb_img = cv2.imread(args.img)
# 	rgb_img = cv2.resize(rgb_img, (510, 510))
# 	pad_size = (510 - args.data_size) // 2
# 	rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

# 	mix = rgb_img * 0.5 + heatmap * 0.5
# 	mix = mix.astype(np.uint8)

# 	# cv2.namedWindow('heatmap', 0)
# 	# cv2.imshow('heatmap', heatmap)
# 	# cv2.namedWindow('rgb_img', 0)
# 	# cv2.imshow('rgb_img', rgb_img)
# 	# cv2.namedWindow('mix', 0)
# 	# cv2.imshow('mix', mix)
# 	# cv2.watiKey(0)

# 	if args.save_img != "":
# 		cv2.imwrite(args.save_img + "/heatmap.jpg", heatmap)
# 		cv2.imwrite(args.save_img + "/rbg_img.jpg", rgb_img)
# 		cv2.imwrite(args.save_img + "/mix.jpg", mix)
