#build a code to generate heat maps - based on heat.py
#We will build in a class format
#and it will be called in main.py in the end of test step

#importing the libraries
import torch 		#PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR).
import torch.nn as nn	#nn.Module is the base class for all neural network modules in PyTorch. Your models should also subclass this class.
import torch.nn.functional as F	#torch.nn.functional contains all the functions in the torch.nn library. torch.nn.functional is typically imported as F. torch.nn.functional provides some helper functions like relu, tanh, etc.
import torchvision.transforms as transforms	#The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
import cv2	#OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library.
from PIL import Image	#Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
import matplotlib.pyplot as plt	#Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
import numpy as np	#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import argparse	#The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.
import warnings	#This module defines the warning category classes. Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program.

from utils.config_utils import load_yaml	#load_yaml is used to load the yaml file. The yaml file is used to store the configuration of the model.	
from models.builder import MODEL_GETTER #, BACKBONE_GETTER
from utils.costom_logger import timeLogger #, logger

warnings.simplefilter("ignore") # ignore warnings

#To transform this code in a Class - Heatmap
class Heatmap:

	def __init__(self, args):
		self.args = args
		self.load_yaml(args) #, args.c) #load_yaml is used to load the yaml file. The yaml file is used to store the configuration of the model.
		assert self.args.pretrained_heat != "", "pretrained_heat is not defined"
		self.model = self.build_model()	#build_model is used to build the model. The model is used to classify the images.

	def load_yaml(self, args): #, config_path): #to check why was used config_path
		#implement your configuration file parsing logic here
		"""
		load yaml file
		"""
		config = load_yaml(args.c) #config_path)	#load_yaml is used to load the yaml file. The yaml file is used to store the configuration of the model.
		return config	#return config
	
	def load_model(self, model, model_path, args):
		"""
		load model
		"""
		model = MODEL_GETTER[args.model_name]( #
		use_fpn = args.use_fpn,
		fpn_size = args.fpn_size,
		use_selection = args.use_selection,
		num_classes = args.num_classes,
		num_selects = args.num_selects,
		use_combiner = args.use_combiner,
	) # about return_nodes, we use our default setting

		### load model
		checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu')) #checkpoint
		model.load_state_dict(checkpoint['model'])
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.to(args.device)

	@staticmethod
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
	
	@staticmethod
	def get_heat(model, img, original_class): #
		# only need forward backbone

		with torch.no_grad(): #torch.no_grad() is used to deactivate the gradient calculation feature in PyTorch. It is used to reduce the memory usage and speed up the computations.
			outs = model.forward_backbone(img.unsqueeze(0)) #
		
		features = []
		for name in outs: #
			features.append(outs[name][0]) #append() is used to add the elements to the end of the list.

		layer_weights = [8, 4, 2, 1] #layer_weights is used to store the weights of the layers. The weights are used to calculate the weighted sum of the gradients of the features with respect to the loss.
		heatmap = np.zeros([args.data_size, args.data_size, 3]) #heatmap is used to store the heatmap. The heatmap is used to visualize the regions of the image that are important for the classification of the image.	
		for i in range(len(features)):
			f = features[i]
			f = f.cpu()
			if len(f.size()) == 2:
				S = int(f.size(0) ** 0.5)
				f = f.view(S, S, -1)

			# if you use original backbone without our module, 
			# please set classifier to your model's classifier. (e.g. model.classifier)
			gramcam = simple_grad_cam(f.unsqueeze(0), classifier=torch.ones(f.size(-1), 200)/f.size(-1), target_class=original_class)
			gramcam = gramcam.detach().numpy()
			gramcam = cv2.resize(gramcam, (args.data_size, args.data_size))
			# heatmap colour : red
			heatmap[:, :, 2] += layer_weights[i] * gramcam

		heatmap = heatmap / sum(layer_weights)
		heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
		heatmap[heatmap < args.threshold] = 0 # threshold
		heatmap *= 255
		heatmap = heatmap.astype(np.uint8)

		return heatmap
	
	#def save_heatmap(self, heatmap, rgb_img, mix):	#
		#if args.save_img != "":
		#	cv2.imwrite(args.save_img + "/heatmap.jpg", heatmap)
		#	cv2.imwrite(args.save_img + "/rbg_img.jpg", rgb_img)
		#	cv2.imwrite(args.save_img + "/mix.jpg", mix)

	def generate_heatmap(self, model, img, args, original_class):

		img_transforms = transforms.Compose([
			transforms.Resize((510, 510), Image.BILINEAR),
			transforms.CenterCrop((args.data_size, args.data_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
			])
		
		#Ensure that necessary atrributes are present
		#assert self.args.pretrained_heat != "", "pretrained_heat is not defined"
		assert hasattr(self, 'model'), "model is not defined, Call build_model or set it as a class attribute."
		assert hasattr(args, 'device'), "args.device is not defined, Pass it as an argument or set it as a class attribute."

		#continue	
		img = cv2.imread(self.args.img) #args.img _ vem do main
		img = img[:, :, ::-1] # BGR to RGB.  #Ocorreu erro de 'None Type' - acces an index or a key of a variable that is a variable that is of the None type data type - occurs when a method or a function return a None rather than the desired value  - TEM Q COLOCAR O NOME CORRETO DA IMAGEM NA NO TERMINAL QUANDO FOR CARREGAR A IMAGEM PARA FAZER O HEAT MAP - TEM Q VERIFICAR COMO FAZER TODAS AS ETAPAS EM AUTOMATICO - NO MODELO DISPONIVEL ESTA PASSO A PASSO - TREINA - VALIDA (EVAL) - HEAT MAP - INFER
	
		# to PIL.Image
		img = Image.fromarray(img)
		img = img_transforms(img)
		img = img.to(self.args.device)

		# get heatmap and original image
		heatmap = self.get_heat(model, img, original_class)

		rgb_img = cv2.imread(args.img)
		rgb_img = cv2.resize(rgb_img, (510, 510))
		pad_size = (510 - self.args.data_size) // 2
		rgb_img = rgb_img[pad_size:-pad_size, pad_size:-pad_size]

		mix = rgb_img * 0.5 + heatmap * 0.5
		mix = mix.astype(np.uint8)

		if self.args.save_img != "":
			cv2.imwrite(self.args.save_img + "/heatmap.jpg", heatmap)
			cv2.imwrite(self.args.save_img + "/rbg_img.jpg", rgb_img)
			cv2.imwrite(self.args.save_img + "/mix.jpg", mix)

		#self.save_heatmap(heatmap, rgb_img, mix)
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser("PIM-FGVC Heatmap Generation")
	parser.add_argument("--c", default="", type=str) #c is used to store the path of the configuration file. The configuration file is used to store the configuration of the model.
	parser.add_argument("--img", default="", type=str) #img is used to store the path of the image. The image is used to generate the heatmap.
	parser.add_argument("--target_class", default=0, type=int) #target_class is used to store the target class. The target class is the class for which the heatmap is generated.
	parser.add_argument("--threshold", default=0.75, type=float) #threshold is used to store the threshold value. The threshold value is used to remove the noise from the heatmap.
	parser.add_argument("--save_img", default="", type=str, help="save path") #save_img is used to store the path to the directory where the heatmap is saved.
	parser.add_argument("--pretrained", default="", type=str) #pretrained is used to store the path to the pretrained model. The pretrained model is used to generate the heatmap.
	parser.add_argument("--model_name", default="swin-t", type=str, choices=["swin-t", "resnet50", "vit", "efficient"]) #model_name is used to store the name of the model. The name of the model is used to load the model.

	args = parser.parse_args()

	assert args.c != "", "Please provide config file (.yaml)"
	#args = parser.parse_args()
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	heatmap_generator = Heatmap(args)
	heatmap_generator.generate_heatmap()