#A script to balance the data in classification task
# To balance an unbalanced dataset - 
#Write a Code to read how many images are in each class (Folder)
#Ande make use of some techniques of dta augmentation to balance the dataset

#In few words the general steps will be:
##Make a loop to read and count the number of each class (folder)
#Make a (a kind of) ranking with the number of each class
#Apply the data augmentation techniques to the classes (folder) with less images to
#create aditional samples until the number of images in the target class matches the 
#number of images in the class with the most samples (or any desired balanced proportion).

import os
import random
import numpy as np
import shutil
import cv2
from PIL import Image
#	from keras import ImageDataGenerator
from torchvision import transforms
import torch

def augment_image(args):

	print("Data Augmentation to balance the dataset")
	#building a function to save the images after data augmentation balance
	def save_augmented_image(augmented_img, augmented_img_path):
		#ensure the output directory exists
		os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)

		#save the augmented image
		augmented_img.save(augmented_img_path)

	#Define the path to the dataset directory
	dataset_directory = args.data_path #"/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/all/"
	output_directory = args.data_path 

	#generate the output directory folder #Won't more need, because the output directory is the same of dataset directory
	#if not os.path.exists(output_directory):
		#os.makedirs(output_directory, exist_ok=True) 

	#get the number of images in each class (folder)
	class_folders = os.listdir(dataset_directory)

	#initialize the variables to keep to keep track of mim and max class size
	min_class_size = float('inf')
	max_class_size = 0

	#loop trough the class folders to find the min and max class size
	for class_folder in class_folders: #building a loop to count the classs (folder)

		class_path = os.path.join(dataset_directory, class_folder) #define the class path
		num_images = len(os.listdir(class_path)) #count the number of images in the class

		min_class_size = min(min_class_size, num_images) #update the min class size #function mim
		max_class_size = max(max_class_size, num_images) #update the max class size  #function max
		
		#save the name of folder of max_class_size
		if max_class_size == num_images:
			target_class = class_folder
		
		#define the target class size
	target_class_size = max_class_size

	#copy the target class folder to the output directory - won't more need, because the output directory is the same of dataset directory
	#shutil.copytree(os.path.join(dataset_directory, target_class), os.path.join(output_directory, target_class))

	#save the target_class folder in output directory
	#target_class_dir = os.path.join(output_directory, target_class)


	#Initalize the data augmentation generator
	#use several data augmentation techniques to generate aditional images - next step of the algorithm
	

	datagen_1 = torch.nn.Sequential(
		transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
	)

	datagen_2 = torch.nn.Sequential(
		transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),

	)

	datagen_3 = torch.nn.Sequential(
		transforms.RandomSolarize(threshold=0.1, p=0.1),
	)

	datagen_4 = torch.nn.Sequential(
		transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
	)

	datagen_5 = torch.nn.Sequential(
		transforms.RandomAutocontrast(p=0.1),
	)

	datagen_6 = torch.nn.Sequential(
		transforms.RandomEqualize(p=0.1))
	
	datagen_7 = torch.nn.Sequential(
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

	)

	#building a list of datagens, to use it in the loop of augmentation to use one of the datagens in each iteration
	datagens = [datagen_1, datagen_2, datagen_3, datagen_4, datagen_5, datagen_6, datagen_7]

	#augmented_images_as_np = []  # Initialize a list to store augmented images
	# Loop through the class folders to balance the dataset
	for class_folder in class_folders:

		#put a print to identify in what folder is the image
		print(f"Augmenting images in {class_folder} folder...")
		
		class_path = os.path.join(dataset_directory, class_folder)  # Define the class path
		num_images = len(os.listdir(class_path))  # Count the number of images in the class
		
		#num_images_updated = num_images  #to verify how to put a updater in this line, to update the number of images in the class after each augmentation
		
		# Calculate the number of augmentations needed to balance the class
		#The original formula was subtraction, but I changed to division, its need to use only the entire number of division, ok
		num_augmentations = target_class_size // num_images #num_images_updated	
		
		if num_augmentations <= args.num_augments:  # If the class is already balanced, continue, in this line num_agumentations =1 don't copy the target class folder to the output directory
			continue

		# Create a subfolder within the output directory for this class
		output_class_dir = os.path.join(output_directory, class_folder)
		os.makedirs(output_class_dir, exist_ok=True)

		#loop trough the images in the class folder to augment the images
		#perform data augmentation to generate aditional images
		
		for image_file_name in os.listdir(class_path):
			
			image_path = os.path.join(class_path, image_file_name) #define the image path
			image = Image.open(image_path) #read the image
			
			image_as_np = np.array(image)	#convert the image to numpy array
			image_as_np = image_as_np.transpose(2, 0, 1) #convert the image to numpy array

			# Generate num_augmentations additional images for each original image
            # Perform data augmentation

			for i in range(num_augmentations):
            # Choose a datagen from the list in each iteration
			    
				datagen = datagens[i % len(datagens)]
	
				augmented_image_as_np = datagen(torch.from_numpy(image_as_np))

				augmented_image = Image.fromarray(augmented_image_as_np.numpy().astype(np.uint8).transpose(1, 2, 0))
				augmented_filename = f"{i}_{image_file_name}"
				augmented_img_path = os.path.join(output_class_dir, augmented_filename) # + '.jpg')

            # Save the augmented image
				save_augmented_image(augmented_image, augmented_img_path)
