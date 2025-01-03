import os  #os is a miscellanoius operating system interface
import shutil #high level operations
import random

#qdo o main chama a funcao ele envia a informacao via args
#from utils.config_utils import load_yaml, build_record_folder, get_args  #importanto os argumentos

def split_data(args, fold):  #recebe argsdo main e fold que o contador 	#Criando o inicio da Funcao

	#number of folds
	num_folds = args.num_folds #the same value of the Split1.py - maybe the user could enter this value in .yaml file

	main_data_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/folds/'  #here are saved the folds divided
	#verificar se nao vai precisar voltar para dentro do 'for'
	#fold_folder = f'/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cross/data/folds/fold{fold}/'   #aqui salva o camimho da pasta da dobra {fold} eh o contador
	test_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/test/'   #salva o caminho para o set test
	train_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/train/' #salva o caminho para train set
	val_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/val/'   #salva o caminho para val_set
	metrics_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/metrics/'  #para metricas

#create the folders to save the results - As 4 linhas abaixo foram passadas para SPlit1.py gerar antes para o config_utils.py jah puxar para usar no dataset.py
#os.makedirs('./results/', exist_ok=True)  #building the folder results

#os.makedirs('./data/train/', exist_ok=True) #building the folder where will be saved the train set
#os.makedirs('./data/test/', exist_ok=True)  #building the folder where will be saved the test set
#os.makedirs('./data/val/', exist_ok=True)

	#building a train and test set with the folds
	#for fold in range (1, args.num_folds+1):   #the condition to create the train ans test folders - run the numbers of defined folers - o fold do for q vai criar a enumeracao das pastas 
	
	print(f"Continue build train, test and val folders to fold {fold}")   #condition to continue - in the code it will be replaced by the end of fold training - TIRAR ESSA APRTE AQUI QDO RODAR NO CODIGO COMPLETO
        #input()  #ok this loop worked in begin
		
	fold_folder = os.path.join(main_data_folder, f'fold{fold}')  ##aqui salva o camimho da pasta da dobra {fold} eh o contador
	
	#fold_folder = f'/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cross/data/folds/fold{fold}/'   #aqui salva o camimho da pasta da dobra {fold} eh o contador
	#test_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cross/data/test/'   #salva o caminho para o set test
	#train_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cross/data/train/' #salva o caminho para train set
	#val_set_folder = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cross/data/val/'   #salva o caminho para val_set
	#aqui vai o val_set_folder = 
	
	
	fold_folder = os.path.join(main_data_folder, f'fold{fold}')  #verificar se vai funcionar
	
	test_folders = os.listdir(fold_folder) #os.listdir returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order. recebe a lista de folds (folders) que há dentro do caminho
		#FAZENDO O FOR PARA TEST
		# Move test set folders to test_set_folder # ESSE LOOP CRIA O TEST FOLDER
	for test_class_folder in test_folders:   #counting the folds for test
		test_class_path = os.path.join(fold_folder, test_class_folder)  #Join one or more path segments intelligently. recebe a posição e o nome
			
		if os.path.isdir(test_class_path):  #path.isdir() method in Python is used to check whether the specified path is an existing directory or not.			
			test_destination_folder = os.path.join(test_set_folder, test_class_folder)  #definindo o destino - test set - contador test_class_folder
			os.makedirs(test_destination_folder, exist_ok=True)   #building the folder
			image_files = os.listdir(test_class_path)   #returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order. - image files receive this list
			
			for image_file in image_files:  #a counter to count the number of files
		        	
				src_path = os.path.join(test_class_path, image_file)  #The return value is the concatenation of path and all members of *paths, with exactly one directory separator... #origen do dado
				dst_path = os.path.join(test_destination_folder, image_file)   #destino do dado
				shutil.copy(src_path, dst_path)
	       	            
		###AGORA COPIANDO O RESTANTE DOS FOLDS PARA TREINO
		#train_folders = os.listdir(train_fold_folder)  #
	for train_fold in range(0, args.num_folds):  #contador para contar od folds
	
		if train_fold != fold: #se o train fold diferente de fold - verificar esse fold se esta declarado correto
			
			train_fold_folder = f'/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/folds/fold{train_fold}/'   #recebe o destino do train fold
			
			train_folders = os.listdir(train_fold_folder)  #Criando uma lista com a relacao anterior de 
			
			for train_class_folder in train_folders:  #contador para contar a qtd de folders
				
				train_class_path = os.path.join(train_fold_folder, train_class_folder)  #train_class_path recebe o caminho respectivo
				
				#if os.path.isdir(train_class_path):  # path.isdir() method in Python is used to check whether the specified path is an existing directory or not.  #se estiver dentro do caminho executa
					#definindo o tipo de imagem
				image_files = [f for f in os.listdir(train_class_path) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]						
													
				#transferindo as imagens para a pasta de treino
				for image_file in image_files:
					#if image_file not in val_image_files: #se nao estiver transfere
					
					src_path = os.path.join(train_class_path, image_file)
						
					dst_path = os.path.join(train_set_folder, train_class_folder, image_file)
					#train_destination_folder = os.path.join(train_set_folder, train_class_folder, f'fold{train_fold}')
						
					os.makedirs(os.path.join(train_set_folder, train_class_folder), exist_ok=True)
					#dst_path = os.path.join(train_destination_folder, image_file)
						
					shutil.copy(src_path, dst_path)
					
	# Transfer a predetermined percentage of images from train to val
	#val_percent = val_ratio  # Set your desired validation percentage
	for train_class_folder in train_folders:  #contando os folder para transferir
		train_class_path = os.path.join(train_set_folder, train_class_folder)
		image_files = os.listdir(train_class_path)  #montando uma lista com as images do diretorio
		num_images_to_move = int(len(image_files) * args.val_ratio)  #calculando a porcenteagem a ser movida
		
		val_destination_folder = os.path.join(val_set_folder, train_class_folder)  #salva o caminho de destino
		os.makedirs(val_destination_folder, exist_ok=True) #cria
		
		images_to_move = random.sample(image_files, num_images_to_move)  #seleciona as imagens a mover
		for image_file in images_to_move:  #contando as que irao ser movidas
			src_path = os.path.join(train_class_path, image_file)
			dst_path = os.path.join(val_destination_folder, image_file)
			shutil.move(src_path, dst_path)
			
	#Para gerar salvar a pasta de metricas para a dobra
	metric_folders = os.listdir(fold_folder)  #os.listdir returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order. recebe a lista de folds (folders) que há dentro do caminho
	for metric_class_folder in metric_folders:   #counting the folds for metrics
		metric_class_path = os.path.join(fold_folder, metric_class_folder)  #Join one or more path segments intelligently. recebe a posição e o nome
		
		if os.path.isdir(metric_class_path):  #path.isdir() method in Python is used to check whether the specified path is an existing directory or not.
			metric_destination_folder = os.path.join(metrics_set_folder, metric_class_folder)  #definindo o destino - test set - contador metric_class_folder
			os.makedirs(test_destination_folder, exist_ok=True)   #building the folder
			#image_files = os.listdir(test_class_path)   #Talves nao precisa dessa linha

	#Agora gerar um loop para as metricas
	#Nao precisa pois vai gerar via Main - somente transferir o valor para transfer_to_results
	
	#TENTAR POR RETORNO
	#return train_set_folder, val_set_folder, test_set_folder #Verificar se vai funcionar  #retorna o declarado no começo e atualizado o conteudo dentro
	return train_set_folder, val_set_folder, test_set_folder, test_folders, train_folders, metrics_set_folder

#args = get_args()   #recebendo os argumentos
#val_ratio = args.val_ratio  #20% para validacao - Jah eh puxado no inicio da funcao

#Descomentar qdo for usar
#split_data(args, val_ratio)   #essa parte aqui talvez mudar para qdo chamar a funcao em outro.
		
	#para transferir um % predefinido de imagens do treino para validacao
	
	
	#essa parte do codigo colocar em outra funcao		
						
#FUNCAO PARA TRANSFERIR OS RESULTADOS - IMAGENS USADAS E SUAS RESPECTIVAS PASTAS
def transfer_to_results(fold, test_folders, train_folders, val_set_folder, test_set_folder, train_set_folder, metrics_set_folder, heatmaps):

	### AGORA CRIANDO O LOOP PARA TRANSFERIR OS ARQUIVOS PARA O RESULTS - VERIFICAR AONDE SERA INSERIDO NO CODIGO ESSA PARTE
	#Comentando para rodar o teste no main.py
	print(f"Continue to transfering Files to results")   #condition to continue - in the code it will be replaced by the end of fold training
	#input()  #Tirar essa parte qdo for rodar no codigo original
	
	results_fold_path = os.path.join('./results/', f'fold{fold}')
	os.makedirs(results_fold_path, exist_ok=True)  #Criando dentro de results os folders para cada fold
	
	results_test_path = os.path.join(results_fold_path, 'test')  #criando o caminho para copiar o teste para dentro
	os.makedirs(results_test_path, exist_ok=True)
			
	for test_class_folder in test_folders:  #contador para contar a qtd de pastas de teste
		
		src_test_path = os.path.join(test_set_folder, test_class_folder) #criando caminho de 'recurso' source_path para test
		dst_test_path = os.path.join(results_test_path, test_class_folder)
		
		if os.path.exists(dst_test_path):  #checando se exite o diretorio
			shutil.rmtree(dst_test_path)  # Delete the existing directory
		
		shutil.copytree(src_test_path, dst_test_path)    #copiando para dentro do destination]				
	
	results_train_path = os.path.join(results_fold_path, 'train')
	os.makedirs(results_train_path, exist_ok=True)  #criando o diretorio para resultado
	
	for train_class_folder in train_folders:  #o mesmo para o caminho de treino
		src_train_path = os.path.join(train_set_folder, train_class_folder)
		dst_train_path = os.path.join(results_train_path, train_class_folder)
		shutil.copytree(src_train_path, dst_train_path)
        
        #para validacao
	results_val_path = os.path.join(results_fold_path, 'val')
	os.makedirs(results_val_path, exist_ok=True)
	val_folders = os.listdir(val_set_folder)	
	
	for val_class_folder in train_folders:  #o mesmo para o caminho de treino - val_class_folder eh o limite do contador
		src_val_path = os.path.join(val_set_folder, val_class_folder)
		dst_val_path = os.path.join(results_val_path, val_class_folder)
		shutil.copytree(src_val_path, dst_val_path)

	#copy heatmaps to results
	heatmaps = os.path.join('data', 'heatmaps')
	shutil.copytree(heatmaps, os.path.join("results", f"fold{fold}"), dirs_exist_ok=True)
	#building a loop to copy heatmaps to results
	#for heatmap_class_folder in heatmaps:  #o mesmo para o caminho de treino - val_class_folder eh o limite do contador
	#print(f'heatmap: {heatmap_class_folder}')
	#input()
	#src_heatmap_path = os.path.join(heatmaps, heatmap_class_folder)
	#dst_heatmap_path = os.path.join(results_val_path, heatmap_class_folder)
	#shutil.copytree(src_heatmap_path, dst_heatmap_path)	
	
	#transferindo tbm as metricas para o results
	#metrics_set_folder = './data/metrics/'  #verificar se vai aceitar e se nao vai precisar declarar o caminho inteiro
		
	#results_metrics_path = os.path.join(results_fold_path, 'metrics')  #criando o diretorio dentro do results para transferir os graficos
	#os.makedirs(results_metrics_path, exist_ok=True)  #cria se nao existir
	
	#src_metrics_path = os.path.join(metrics_set_folder, val_class_folder) #Essa gambiarra eh para nao precisar fazer um loop na funcao anterior  #recebendo o camimho do diretorio
	#dst_val_path = os.path.join(results_metrics_path) #, val_class_folder) #Essa gambiarra eh para nao precisar fazer um loop na funcao anterio
	#shutil.copytree(src_metrics_path, dst_val_path)  #copiando
	
	results_metric_path = os.path.join(results_fold_path, 'metrics')
	os.makedirs(results_metric_path, exist_ok=True)
	metric_folders = os.listdir(metrics_set_folder)
	
	for metric_class_folder in os.listdir(metrics_set_folder):
		src_metric_path = os.path.join(metrics_set_folder, metric_class_folder)  # Corrected path construction
		dst_metric_path = os.path.join(results_metric_path, metric_class_folder)

		if os.path.isdir(src_metric_path):
			shutil.copytree(src_metric_path, dst_metric_path)
		else:	# If it's a file, copy it directly
			shutil.copy2(src_metric_path, dst_metric_path)
	
	#for metric_class_folder in metrics_set_folder:  #o mesmo para o caminho de treino - val_class_folder eh o limite do contador
		#src_metric_path = os.path.join(val_set_folder, val_class_folder)
		#dst_metric_path = os.path.join(results_metric_path, metric_class_folder)
		#shutil.copytree(src_metric_path, dst_metric_path)

		#talves não precise do for	
	
        
       #limpando os diretorios originais
	shutil.rmtree(train_set_folder)
	shutil.rmtree(test_set_folder)
	shutil.rmtree(val_set_folder)
	shutil.rmtree(metrics_set_folder)
	shutil.rmtree(heatmaps)
	
	os.makedirs(train_set_folder)
	os.makedirs(test_set_folder)
	os.makedirs(val_set_folder)
	os.makedirs(metrics_set_folder)
	os.makedirs(heatmaps)
	
# Call the function with appropriate arguments  #descomentar qdo for usar
#transfer_to_results(fold, test_folders, train_folders, val_set_folder, test_set_folder, train_set_folder)
		
		
