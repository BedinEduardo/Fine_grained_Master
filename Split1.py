import os     #miscellaneous operating system interfaces
import shutil   #High level file operationso
import random   #to generate image radom separation

#qdo chama o args o main transfere a informacao
#from utils.config_utils import load_yaml, build_record_folder, get_args  #importanto os argumentos

#CRAINDO A FUNCAO PARA CRIAR FOLDS - VAI SER CHAMADO DIRETO NO MAIN.PY - NA FUNCAO ENTRY __name__ = "__main__"
def create_folds(args):     #(main_data_folder, num_folds):
	
	#defining data_folder where is saved the folder with the images
	main_data_folder = './data/all'  #defining the path where are the image folder

	#args = get_args()   #recebendo os argumentos

	#Now, creating the new folders for each original folder

	for fold in range(0, args.num_folds):   #for folds into the range by 1 until the incremented value by num_folds+1, do
	
		fold_folder = f'data/folds/fold{fold}/' #defining the fold_folder variable - the path where be saved
		os.makedirs(fold_folder, exist_ok=True) #create the path
	
	#OK, WORKED UNITL HERE
	#NOW TO FINISH TE CODE TO DIVIDE DE IMAGES IN THE PATH
	
#ITERATION TROUGH THE FOLDS IN THE FOLDERS
	class_folders = os.listdir(main_data_folder)  #listdir returns a list containing the names of the entries in the directory	 given the path

	for class_folder in class_folders: #counting class_folder in the range of class_folders... - the number of folders

		class_path = os.path.join(main_data_folder, class_folder)    #os.path commom path manipulation, .join join one or more path segments intelligently - concatenation
	
		#Now, listing all image files in the class folder
		image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpeg', '.jpg'))] #para f dentro de f, n listagem do caminho se f.lower for verdadeiro .endswith returns a bollean, se existir .png ou jpeg ou .jpg... faca -  recebe o numero
	
		random.shuffle(image_files)   #shuffle the image list randomily
	
		#calculate the number of images per fold
		num_images = len(image_files)   #reading the number of files - len read and count the itens in a list - read the quantity
		images_per_fold = num_images // args.num_folds   #floor division - divisao inteira - num_images per num_folds
	
		#moving the images to the corresponding fold folders
		for fold in range(0, args.num_folds):  #for fold in range 1 until num_folds +1, do
		
			start_idx = (fold) * images_per_fold #initializing in folder 0, because the counter start in 1, count the initial image in fold
			end_idx = start_idx + images_per_fold # count the end index image to the fold
		
			fold_folder = f'data/folds/fold{fold}/{class_folder}/'  #create in specifi path of the counting folder
			os.makedirs(fold_folder, exist_ok=True)  #creating the directory
		
			for image_file in image_files[start_idx:end_idx]:  #for image_file in range of image_files(total) in range of start_idc until end_idx
			
				src_path = os.path.join(class_path, image_file)  #coloca a imagem dentro do camimho da classe
				dst_path = os.path.join(fold_folder, image_file)
			
				shutil.copy(src_path, dst_path)  #shutil high level operations - copy the files
			
	print("Data Split and folding is complete")

	os.makedirs('./results/', exist_ok=True)  #building the folder results

	os.makedirs('./data/train/', exist_ok=True) #building the folder where will be saved the train set
	os.makedirs('./data/test/', exist_ok=True)  #building the folder where will be saved the test set
	os.makedirs('./data/val/', exist_ok=True)
	os.makedirs('./data/metrics/', exist_ok=True)   #criando a pasta para salvar as metricas - indo para train_test_split para transferir para resultados e para criar a pasta novamente
			
	print("Train, Test, Val and results folds are builded.")


#ESS PARTE EH PARA CHAMAR O CODIGO
#defining data_folder where is saved the folder with the images
#main_data_folder = './data/all'  #defining the path where are the image folders

#especifying the desidered number of folds
#num_folds = args.num_folds   #This value the user can type in .yaml file

#DEIXAR COMENTADO - QDO NECESSARIO DESCOMENTAR - USAR PARA CHAMAR A FUNCAO
#create_folds(main_data_folder, num_folds)
