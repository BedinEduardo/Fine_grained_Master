
#IMPORTANDO AS BIBLIOTECAS
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
import csv
import os
import shutil

import cv2

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args  #o get_args eh chamado na entrada - no Exeution Block - if __name__ = __"main"__
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics, test #
#from heat import Heatmap 		#para gerar o heatmap	

from train_test_split import split_data, transfer_to_results  #o primeiro eh para gerar as dobras e carregar as imagens e o segundo eh para transferir depois para results - ficar registrado
#from heat_map import Heatmap  #para gerar o heatmap
from HeatMap2 import generate_heatmap  #para gerar o heatmap
#import train_test_split   #para gerar e rodar o Kfolds (Nfolds)
#from Data_Balancing_1 import augment_image  #para balancear o dataset
#from covertToRgb import convert_to_rgb  #para converter as imagens para RGB

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support

warnings.simplefilter("ignore")

def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2

def set_environment(args, tlogger):  #
    
    print("Setting Environment...")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #DEVICE PLACEMENT - CPU OR GPU
    
    ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")    #CONSTRUINDO O DATALOADER - o comeco esta aqui verificar como fazer esses carregamentos por corss-validation
    
    train_loader, val_loader, test_loader = build_loader(args)  #VEM DO ARGUMENTOS - CARREGA O TRAIN_LOADER E O VAL_LOADER - com os valores para build_loader do dataset.py - terah q alterar lah a forma que carrega os sets
    
    if train_loader is None and val_loader is None: #se estiver vazio o treino e a validacao
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:  #se nao estiver vazio
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader))) #carrega os elementos do dataset e a qtd de itens nesse
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")  #terá q vir os dois juntos - definir no dataset.py como fazer - talvez mudar o yaml
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    
    if val_loader is not None:   #sempre virah com train, val and test sets - verifcicando se a validacao possui dataset
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
        
    if test_loader is not None:   #para carregar o teste - verficar aonde inserir o test
        print("     Test Samples :{} (batch: {})".format(len(test_loader.dataset), len(test_loader)))    #verificar se vai funcionar  - montar o test_loader
    else:
        print("     Without test step")
       
    tlogger.print()   #serah necessario alterar a forma como o dataset eh carregado. 
    			#1- divir em Nfolds (Split1.py - pode ser via .yaml) 
    			#2- com o train set formado dividir ele em train and test para cada fold - esta funcionando (separado do codigo geral) - valor num_folds e val_ratio setado via yaml
    			#3- o carreagamento para aqui serah da mesma forma, somente a forma que serah montado no dataset.py que vai ser diferente - verificar config_utils
    			#4- Depois que dividir os datasets em Nfolds, e gerar os sets de train, validation e test - verficar como implementar a etapa de test
    			#5- Depois das etapas anteriores geradas - fazer loop para o codigo executar os Nfolds - kdobras - conforme valor setado num_folds
    			#6- Implementar as metricas P, R, Fscor - Acuracia já tem - talvez matriz confusão - TP, FP, TN, FN talvez
    			#7- Implementar DA para Balanceamento de dataset

    ### = = = =  Model = = = =   
    tlogger.print("Building Model....")  #depois de carregar o dataset
    model = MODEL_GETTER[args.model_name](  #isso daqui vem do yaml - mas passou pelo MODEL_GETTER do builder.py - vai buscar o model_name no yaml e carrega as informacoes para o builder.py
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting
    
    if args.pretrained is not None:  #para terminar de treinar o .pt
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])   #importando os dicionarios - keys
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0  #qundo nao houver modelo pretreinado - comeca pela epoca zero

    # model = torch.nn.DataParallel(model, device_ids=None) # device_ids : None --> use all gpus.
    model.to(args.device)   #args para o device escolhido
    tlogger.print()
    
    """
    if you have multi-gpu device, you can use torch.nn.DataParallel in single-machine multi-GPU 
    situation and use torch.nn.parallel.DistributedDataParallel to use multi-process parallelism.
    more detail: https://pytorch.org/tutorials/beginner/dist_overview.html
    """
    
    if train_loader is None: #se o o carregador do treino for vazio - carrega essas informacaoes - para validar com o .pt
        return train_loader, val_loader, model, None, None, None, start_epoch   #aqui tinnha um problema de posicao - estava errado a sequancia - verificar aonde inserir o test_loader
    
    ### = = = =  Optimizer = = = =  
    tlogger.print("Building Optimizer....")   #inserir alguns outros otimizadores aqui
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)  #inserir alguns outros otimizadores aqui
    elif args.optimizer == "Adadelta":
         optimizer = torch.optim.Adadelta(model.parameters(), lr=args.max_lr)  #inserir alguns outros otimizadores aqui
    elif args.optimizer == "RMSprop":
         optimizer = torch.optim.RMSprop(model.parameters(), lr=args.max_lr)  #inserir alguns outros otimizadores aqui
    if args.pretrained is not None:   #carregao o otimizador do modelo pretreinado
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader,test_loader, model, optimizer, schedule, scaler, amp_context, start_epoch   #retornando os valores da funcao - o test_loader foi inserido, verifificar se nao via dar erro de posicao


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):  #funcao de treino - Nfolds faz no main - somente treino, aqui nao faz validacao nem teste
    #fold_metrics = 0
    optimizer.zero_grad()
    total_batchs = len(train_loader) # just for log - numero de itens que vem do train_loader
    show_progress = [x/10 for x in range(11)] # just for log - a cada 10 passos mostra na tela
    progress_i = 0
    
    #writer = SummaryWriter()  #vamos tentar essa    
    #TP, FP, FN = 0, 0, 0
    all_preds = []   #para salvar os preditos e os labels de cada step
    all_labels = []
               
    for batch_id, (ids, datas, labels) in enumerate(train_loader):  #para contar a qtd de dados para realizar o treino em funcao da qtd no train loader - talvez tera q ter um for antes para fazer o Nfolds
        all_labels += labels.tolist() #The tolist() function is used to convert a given array to an ordinary list with the same items, elements, or values.
        
        model.train()  #chama o modelo - builder.py para treinar
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'
            
            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)
            #print("OUTS:", outs)
            #print("OUTS KEYS:", outs.keys())
            #input()

            loss = 0.0
            for name in outs:
                
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit, labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0
                
                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c

                        comb_preds = outs["comb_outs"]
                        comb_preds = torch.sort(comb_preds, dim=-1, descending=True)[1]
                        comb_preds = comb_preds[:, 0]
                        all_preds += comb_preds.tolist()



                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
            
            loss /= args.update_freq
        
        """ = = = = calculate gradient = = = = """
        if args.use_amp:
            scaler.scale(loss).backward()
        
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update() # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):  #transfera pra o wandb - verficar como inserir as outras metricas
            #print("AAAAAAAAAAAAAAAAAAA")
            model.eval()  #chama a validation
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(args, msg, outs, labels, batch_size) #, epoch, fold, fold_metrics) #Aqui chama a funcao para calcular as metricas de treino a cada epoca #Original(args, msg, outs, labels, batch_size) #Adicionado 'fold_metrics = '  #vai            
            wandb.log(msg)
          
        #writer.add_pr_curve('pr_curve', labels, outs, 0)  #tentando gerar a curva PR
        #writer.close()

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1
    
    #print("ALL_PREDS", all_preds)
    #print("ALL_LABELS", all_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    return precision, recall, fscore
	#print("Calculando P, R e F")
    #print("Precision", precision)
    #print("Recall", recall)
    #print("F-score", fscore)
    #input()

##function to extract the original class of any image
def extract_original_class(args, image_name):
	#Assuming the file name is in the format: class_name_image_name
	parts = image_name.split("_") #split the image name by "_"
	#print(f'Parts: {parts}')	
	#input()
      
	#get the list of folders in spcefic path
	#folders = [f for f in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, f))]
	folders = [f.strip() for f in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, f))]
 	
    #rank the folders in alphabetical order
	sorted_folders = sorted(folders) #save the ranked folders in a list
	#print(f'Sorted folders: {sorted_folders}')
	#input()

	#now compare the image name and rank accordingly
	if len(parts) == 2:
        # If there's only one underscore, use the part after the underscore
		class_name_from_image = parts[0].strip() #parts[1]
		#print(f'Class name from image single underscore: {class_name_from_image}')
		#input()


	elif len(parts) > 2:
        # If there are multiple underscores, use the part before the last underscore
		class_name_from_image = "_".join(parts[1:-1]).strip()
		#print(f'Class name from image multiple underscores: {class_name_from_image}')
		#input()
	else:
        # If there's only one underscore or none, use the whole name as the class name
		class_name_from_image = parts[0].strip()
		#class_name_from_image = parts[0].rsplit('.', 1)[0].strip()
		#print(f'Class name from image no underscores: {class_name_from_image}')
		#input()

	if not class_name_from_image:
		print("Warning: No class name found in the image name.")
		#input()
		return None  
            
	try:
		class_index = sorted_folders.index(str(class_name_from_image)) #get the index of the class name in the list of folders
		return class_index #sorted_folders[class_index] #return the class name
		
	except ValueError: #if the class name is not found in the list of folders
		print(f"Class '{class_name_from_image}' not found in the sorted folders.")
            
		return None			
		#return parts[0]
    

def main(args, tlogger):
	"""
	save model last.pt and best.pt
	"""
    #criando um loop para contar as iteracoes dos Nfolds para gerar os train, test and val antes de mandar para treino
	# Initialize lists to store all metrics for each fold  
	all_fold_precisions = [[] for _ in range(args.num_folds)] #para treino
     
	all_fold_recalls = [[] for _ in range(args.num_folds)]
     
	all_fold_fscores = [[] for _ in range(args.num_folds)]
     
	#agora para validacao
	eval_fold_precisions = [[] for _ in range(args.num_folds)] #validacao
     
	eval_fold_recalls = [[] for _ in range(args.num_folds)]
     
	eval_fold_fscores = [[] for _ in range(args.num_folds)]

	test_fold_precision = [[] for _ in range(args.num_folds)] #test
	test_fold_recalls = [[] for _ in range(args.num_folds)]
	test_fold_fscores = [[] for _ in range(args.num_folds)]

	for fold in range(args.num_folds):  #Loop para rodar os folds - chamar os treinos e outras funcoes a cada loop de cada fold		
		
		tlogger.print(f"Fold {fold + 1}/{args.num_folds}")  #vai imprimir em qual fold esta
		
		#chamando a funcao para gerar train, test and validation
		#split_data(args) #from train_test_split.py  #roda uma vez e deveria ir para treino, mas fica rodando a qtd de vezes q esta definido no num_folds
		
		# Calling the split_data function to generate train, val, and test sets		
		#train_set_folder, val_set_folder, test_set_folder = split_data(args, fold) #args leva os argumentos, fold leva o contador
		train_set_folder, val_set_folder, test_set_folder, test_folders, train_folders, metrics_set_folder = split_data(args, fold)
    		
		train_loader, val_loader, test_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger)
		#print(model)   #deixar comentado por eqto
		#input()Heat maps generating for fold:  TP , In Progress...

		best_acc = 0.0
		best_eval_name = "null"

		if args.use_wandb:
			wandb.init(entity=args.wandb_entity,
					project=args.project_name,
					name=args.exp_name,
					config=args)
			wandb.run.summary["best_acc"] = best_acc
			wandb.run.summary["best_eval_name"] = best_eval_name
			wandb.run.summary["best_epoch"] = 0
			
			
		all_precisions, all_recalls, all_fscores = [], [], []  #montando uma lista - para contar P, R e F por epoca de treino
		
		all_eval_precisions, all_eval_recalls, all_eval_fscores = [], [], []  #para eval
		
		all_test_precisions, all_test_recalls, all_test_fscores = [], [], [] #para test
		
		all_test_class_precisions, all_test_class_recalls, all_test_class_fscores = [], [], [] #para test - por classe

		for epoch in range(start_epoch, args.max_epochs):  #Esse loop chama a funcao de treino a cada epoca - da epoca inicial a max_epochs
			"""
			Train
			"""
			if train_loader is not None:   #Aqui ele chama o treino - se a declaracao do caminho nao for vazia
				tlogger.print(f"Start Training {epoch} Epoch for Fold {fold}")   #mostra a epoca + o fold
				precision, recall, fscore = train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)  #aqui chama as metricas que foram geradas no treino
				#que o retorno do train - da pra chamar em outra funcao no eval.py
				#all_fold_metrics.append(fold_metrics)
				tlogger.print()
				print("Calculando P, R e F para Treino")
				print("Precision", precision)
				print("Recall", recall)
				print("F-score", fscore)
				#input()
				all_precisions.append(precision)  #criando lista com as metricas por epoca de treino
				all_recalls.append(recall)
				all_fscores.append(fscore)	
                #essa parte toda vamos chamar no eval.py - criar uma funcao

				#tlogger.print()

			else: #Aqui, se a declaracao do treino for vazia e tiver caminho setado no val - ele vai direto para o eval - mostra as metricas - essa parte tera q ser automatica - verifcar
				from eval import eval_and_save
				eval_and_save(args, model, val_loader)
				break			
									
			eval_freq_schedule(args, epoch)  # aqui eh o schedule 'agenda/progr' da validacao - chama argumentos e epoch

			model_to_save = model.module if hasattr(model, "module") else model
			checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
			torch.save(checkpoint, args.save_dir + "backup/last.pt")

	
			if epoch == 0 or (epoch + 1) % args.eval_freq == 0:  #para chamar a VALIDACAO
				"""
				Evaluation
				"""
				acc = -1
				
				if val_loader is not None:  #se nao for vazio - validacao
					tlogger.print(f"Start Evaluating {epoch + 1} Epoch for Fold {fold + 1}")   #mostrar a epoca + o fold
					acc, eval_name, accs, precision_eval, recall_eval, fscore_eval = evaluate(args, model, val_loader)  #essa linha chama o eval.py a funcao evaluate para fazer as metricas de validacao
					tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
					tlogger.print()
                         
					print("Calculando P, R e F para Validacao")
                    
					print("Precision Val: ", precision_eval)
                         
					print("Recall Val: ", recall_eval)
                         
					print("fscore Val: ", fscore_eval)
                    
					all_eval_precisions.append(precision_eval)
                    
					all_eval_recalls.append(recall_eval)
                    
					all_eval_fscores.append(fscore_eval)
                         
					#DEPOIS DESSA PARTE AQUI - FAZER O SALVE DAS PASTAS
	

				if args.use_wandb:
					wandb.log(accs)

				if acc > best_acc:
					best_acc = acc
					best_eval_name = eval_name
					torch.save(checkpoint, args.save_dir + "backup/best.pt") #saving the best model after each epoch
				if args.use_wandb:
					wandb.run.summary["best_acc"] = best_acc
					wandb.run.summary["best_eval_name"] = best_eval_name
					wandb.run.summary["best_epoch"] = epoch + 1
			
		#Test Step - dentro do fold mas fora do eval
		if test_loader is not None:  #se o test_loader nao for vazio - faca o teste
			tlogger.print(f"Start Testing for epoch {fold + 1}") #indicando que esta comecando o teste
			precision_test, recall_test, fscore_test, class_precisions, class_recalls, class_fscores = test(args, model, test_loader)  #Calling the test function from eval.py	 
			tlogger.print()  #para pular uma linha
                    
			print("Calculando P, R e F para Test - for each folder")
			print("Precision Test: ", precision_test)
			print("Recall Test: ", recall_test)
			print("Recall Test: ", recall_test)
				
			all_test_precisions.append(precision_test)
			all_test_recalls.append(recall_test)
			all_test_fscores.append(fscore_test)
            
			print("Calculating P, R e F for Test - for each class")
			print("Precision Test: ", class_precisions)
			print("Recall Test: ", class_recalls)
			print("Recall Test: ", class_fscores)

			all_test_class_precisions.append(class_precisions)
			all_test_class_recalls.append(class_recalls)
			all_test_class_fscores.append(class_fscores)

		else:
			print("NAO ENTROU NO TESTE")
			input()
            
		#Now lets call the heat.py to generate the heatmaps
		#first need to transfer best.pt to pretrained folder - best.pt is generated in the eval.py after each best epoch - acc metrics comparision
		#find the best.pt in the backup folder and transfer it to the pretrained folder
		#read the path way of the best.pt
		#the best.pt is saved in records + args.project_name + args.exp_name + backup + best.pt
		best_path = args.records_path + args.project_name + "/"+ args.exp_name + "/backup/best.pt" #path to the best.pt	
		#delete current best.pt in pretrained folder
		os.remove(args.pretrained_heat) # + "/" + "best.pt") #remove the current best.pt in the pretrained folder
		#now transfer the best.pt from best_path to the pretrained folder
		
		shutil.copy(best_path, args.pretrained_heat) #copy the best.pt to the pretrained folder
          
		#now calling the Heatmap class to generate the heatmaps
		
		#heatmap_args = args  #verificar se a necessidade de chamar esse arquivo aqui
		#heat_mapgenerator = Heatmap(heatmap_args)  #chamando a classe Heatmap do heat.py e passando os argumentos
		

		#building a loop to read TP, FP, TN, and FN images from the test folders and apply the heatmaps from them
		base_path = args.heat_path + "heatmaps/" #path to the heatmap folder
		#defining the path to the TP, FP, TN, FN folders
		subdirs = ['TP/', 'FP/', 'TN/', 'FN/']
		
		#building a loop to read the images from the heatmap folders and apply the heatmaps from them
		#loop trhoug the subdirs
		#image_name_count = 0 #to count the number of images in the subdir
		for subdir in subdirs: #what idention is this? - this is the subdir loop
			print('Heat maps generating for fold: ', subdir, ', In Progress...')
			#subdir_path = base_path + subdir #path to the subdir
			subdir_path = os.path.join(base_path, subdir)
			image_name_count = 0 #to count the number of images in the subdir
			#loop through the images in the subdir
			for img in os.listdir(subdir_path):
				if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".RAW"): # and image_name_count < 11:  #verificar se eh necessario colocar o RAW
					if image_name_count < args.num_heat_images: #to limit the number of images to be processed
						#extract the original class of the image
						original_class = extract_original_class(args, img) #To extract the initial part of the name that it is the class

						#full_path = os.path.join(subdir_path, img)
						#img_path = subdir_path + "/" + img #path to the image
						img_path = os.path.join(subdir_path, img) #, subdir_path) #image path

						#read the image
						image = cv2.imread(img_path) #get the image 
						#apply the heatmap	
						image_name_count += 1 #counting the number of images in the subdir  

						generate_heatmap(args, image, original_class, subdir_path, image_name_count) 
						
						#chamando a funcao para gerar o heatmap - passando os argumentos, imagem e a classe original

						#heat_mapgenerator.generate_heatmap(model, image, args, original_class) #To change the way the function is called - the model is not necessary here - the model is already loaded in the Heatmap class - the image is already read in the Heatmap class - the args is already read in the Heatmap class - the original_class is already read in the Heatmap class
					
		#print(f'Class name from image: {class_name_from_image}')
		print('Heat maps generated for fold: ', fold + 1, ', Complete!')


		#Montar as metricas de Treino	#Uma identacao fora do loop das epocas - dentro da identacao dos folds
		#No alinhamento do for das epocas 
		metrics_path = args.metrics_paths #'/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/data/metrics'   #caminho das pastas das metricas
        #Este caminho da para usar tanto no eval como no test tbm
        
		txt_file_path = os.path.join(metrics_path, 'metrics_train.txt')  #criar uma variavel para txt e outra para csv
		txt_val_file_path = os.path.join(metrics_path, 'metrics_val.txt')  #salvando nas dobras para validacao
		txt_test_file_path = os.path.join(metrics_path, 'metrics_test.txt')  #salvando para teste
          
		txt_test_class_file_path = os.path.join(metrics_path, 'metrics_test_class.txt')  #salvando para teste - por classe	
             
		csv_file_path = os.path.join(metrics_path, 'metrics_train.csv')
		csv_val_file_path = os.path.join(metrics_path, 'metrics_val.csv')  #salvando nas dobras para validacao
		csv_test_file_path = os.path.join(metrics_path, 'metrics_test.csv')
          
		csv_test_class_file_path = os.path.join(metrics_path, 'metrics_test_class.csv')  #salvando para teste - por classe
          

		#CRIANDO ARQUIVO DA DOBRA - com os valores de cada epoca - da para criar os graficos - para treino
		with open(txt_file_path, 'w') as txt_file:
			for precision, recall, fscore in zip(all_precisions, all_recalls, all_fscores):
				txt_file.write(f'Precision: {precision}, Recall: {recall}, F-score: {fscore}\n')

			# Create a .csv file and write the metrics to it
		with open(csv_file_path, 'w', newline='') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(['Precision', 'Recall', 'F-score'])

			for precision, recall, fscore in zip(all_precisions, all_recalls, all_fscores):
				writer.writerow([precision, recall, fscore])
		
		print(f"Train Metrics saved to {txt_file_path} and {csv_file_path}")
          
		##AGORA CRIANDO PARA VALIDACAO - .txt
		with open(txt_val_file_path, 'w') as txt_val_file:
			for precision_eval, recall_eval, fscore_eval in zip(all_eval_precisions, all_eval_recalls, all_eval_fscores):
				txt_val_file.write(f'Precision: {precision_eval}, Recall: {recall_eval}, F-score: {fscore_eval}\n')

		#CRIANDO .CSV PARA VALIDACAO
		with open(csv_val_file_path, 'w', newline='') as csv_val_file:
			writer_val = csv.writer(csv_val_file)  #colocando esse write_val para nao confundir com o do loop anterior
			writer_val.writerow(['Precision', 'Recall', 'F-score'])
		
			for precision_eval, recall_eval, fscore_eval in zip(all_eval_precisions, all_eval_recalls, all_eval_fscores):
				writer_val.writerow([precision_eval, recall_eval, fscore_eval])
			
		print(f'Validation Metrics saved to {txt_val_file_path} and {csv_val_file_path}')  #mostrando aonde foi salvo

		#AGORA CRIANDO PARA TESTE
		with open(txt_test_file_path, 'w') as txt_test_file:
			for precision_test, recall_test, fscore_test in zip(all_test_precisions, all_test_recalls, all_test_fscores):
				txt_test_file.write(f'Precision: {precision_test}, Recall: {recall_test}, F-score: {fscore_test}\n')
                
		#CRIANDO O .CSV PARA TESTE
		with open(csv_test_file_path, 'w', newline='') as csv_test_file:
			writer_test = csv.writer(csv_test_file)
			writer_test.writerow(['Precision', 'Recall', 'F-score'])

			for precision_test, recall_test, fscore_test in zip(all_test_precisions, all_test_recalls, all_test_fscores):
				writer_test.writerow([precision_test, recall_test, fscore_test])
        
		#build csv file for test - class
		#it is needed to add the class name to the csv file - what is necessary?
		folders = [f.strip() for f in os.listdir(args.test_root) if os.path.isdir(os.path.join(args.test_root, f))] #get the list of folders in spcefic path
		Sorted_Folderr = sorted(folders) #rank the folders in alphabetical order
		#now lets create the csv file


		with open(csv_test_class_file_path, 'w', newline='') as csv_test_class_file:
			writer_test_class = csv.writer(csv_test_class_file)
		
			# Modify the reader to include class name
			#header = ['Class_Name'] + [f'Precision_{i}' for i in range(1, args.num_classes + 1)] + [f'Recall_{i}' for i in range(1, args.num_classes + 1)] + [f'F-score_{i}' for i in range(1, args.num_classes + 1)]
			header = ["Class_Name"] + ["Precision"] + ["Recall"] + ["F-score"]     
			writer_test_class.writerow(header)

			#modify the reader to include class name
			#header = ['Class_Name', 'Precision', 'Recall', 'F-score']

			#header = ['Class_Name'] + [f'Precision_{i}' for i in range(1, args.num_classes + 1)] + [f'Recall_{i}' for i in range(1, args.num_classes + 1)] + [f'F-score_{i}' for i in range(1, args.num_classes + 1)]

			#header = ['Class_Name'] + [f'Precision_{i}' for i in range(1, args.num_classes + 1)] + [f'Recall_{i}' for i in range(1, args.num_classes + 1)] + [f'F-score_{i}' for i in range(1, args.num_classes + 1)]
		
			#header = ['Class_Name'] + ['Precision', 'Recall', 'F-score']
			#writer_test_class.writerow(header) #(['Precision', 'Recall', 'F-score'])
			
			#print(f'all_test_class_precisions: {all_test_class_precisions}')
			#print(f'all_test_class_recalls: {all_test_class_recalls}')
			#print(f'all_test_class_fscores: {all_test_class_fscores}')
			#input()

			#for class_name, class_precisions, class_recalls, class_fscores in zip(Sorted_Folderr, all_test_class_precisions, all_test_class_recalls, all_test_class_fscores):
			#	row = [class_name] + class_precisions + class_recalls + class_fscores
			#	writer_test_class.writerow(row)
			#print(Sorted_Folderr)
			#print(all_test_class_precisions)
			#print(all_test_class_recalls)
			#print(all_test_class_fscores)
			#input()
			
			for i in range(len(Sorted_Folderr)):
				#new_row = [Sorted_Folderr[i]] + [all_test_class_precisions[i]] +[ all_test_class_recalls[i]] + [all_test_class_fscores[i]]
				#writer_test_class.writerow(new_row)       
				writer_test_class.writerow([Sorted_Folderr[i], all_test_class_precisions[0][i], all_test_class_recalls[0][i], all_test_class_fscores[0][i]])
			
			#for class_name, class_precisions, class_recalls, class_fscores in zip(Sorted_Folderr, all_test_class_precisions, all_test_class_recalls, all_test_class_fscores):
				#writer_test_class.writerow([class_name] + class_precisions + class_recalls + class_fscores)

			#for class_name, precisions, recalls, fscores in zip(Sorted_Folderr, all_test_class_precisions, all_test_class_recalls, all_test_class_fscores):
				# Write class name along with metrics
				#writer_test_class.writerow([class_name, precisions[-1], recalls[-1], fscores[-1]])

			#for class_name, class_precisions, class_recalls, class_fscores in zip(Sorted_Folderr, all_test_class_precisions, all_test_class_recalls, all_test_class_fscores):   #(all_test_class_precisions, all_test_class_recalls, all_test_class_fscores):
				#writer_test_class.writerow([class_name, class_precisions[-1], class_recalls[-1], class_fscores[-1]])
        		  

		##ESSA PARTE A ABAIXO VOU DEIXAR COMENTADA - AS MEDIAS DAS METRICAS DOS FOLDS FAZER MANUALMENTE
		#QUEM PEG							ODIGO PRINCIPAL
		#NOS LOOPS ABAIXO CONTA ERRADO AS PASTAS DESCOBRIR O PQ - VOU PRECISAR DE AJUDA
		#GERANDO UM ARQUIVO GERAL COM TODAS AS METRICAS DE TODAS AS DOBRAS
		# Append the metrics for the current fold to the corresponding lists
		#all_fold_precisions[fold].extend(all_precisions)  #PEGA AS INFORMACOES de cada epoca e e armazena na varaiavel do fold
        
		#all_fold_recalls[fold].extend(all_recalls)
          
		#all_fold_fscores[fold].extend(all_fscores)
          
		#results_path = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_Cruz/results/'  #caminho do resultado #VERIFICAR COMO COLOCAR POR YAML AQUI
          
		#csv_result_file = os.path.join(results_path, 'aggregated_metrics.csv')  #criando o arquivo direto em results com todos os valores
          
		#with open(csv_result_file, 'w', newline='') as csv_file:
            
		#	writer = csv.writer(csv_file)  #ctrlz
            
		#	writer.writerow(['Fold', 'Precision', 'Recall', 'F-score'])
               
		#	for fold, (precisions, recalls, fscores) in enumerate(zip(all_fold_precisions, all_fold_recalls, all_fold_fscores), start=1):
                    
				# Check if any metrics are available for the current fold
        
		#		if not precisions or not recalls or not fscores:
                         
		#			continue  # Skip this fold if any of the lists are empty

				# Calculate aggregate statistics (e.g., mean) for each metric
								
		#		mean_precision = sum(precisions) / len(precisions)

		#		mean_recall = sum(recalls) / len(recalls)
                    
		#		mean_fscore = sum(fscores) / len(fscores)
                    
				# Write aggregTraceback (most recent call last):
  
                    
		# Create a TXT file for aggregated results
		#txt_result_file = os.path.join(results_path, 'aggregated_metrics.txt')
          
		#with open(txt_result_file, 'w') as txt_file:
            
		#	txt_file.write("Aggregated Metrics for All Folds:\n\n")
               
		#	for fold, (precisions, recalls, fscores) in enumerate(zip(all_fold_precisions, all_fold_recalls, all_fold_fscores), start=1):
        #            ,
		#		if not precisions or not recalls or not fscores:
                         
		#			continue  # Skip this fold if any of the lists are empty
                
		#		txt_file.write(f"Fold {fold}:\n")
                
		#		txt_file.write(f"Mean Precision: {sum(precisions) / len(precisions)}\n")
                
		#		txt_file.write(f"Mean Recall: {sum(recalls) / len(recalls)}\n")
                    
		#		txt_file.write(f"Mean F-score: {sum(fscores) / len(fscores)}\n")
                    #this_name
                    
	#	print(f"Aggregated metrics saved to {csv_result_file} and {txt_result_file}")      
		
		#CHAMANDO O GERA OS GRAFICOS DAS METRICAS
		#No mesmo alinhamento do 'for' dos folds - VERIFICAR SE VAI AQUI MESMO - VERIFICAR A ESTRUTURA
		#generate_metrics_train(fold_metrics, fold)	##TENTANDO CHAMAR NO TREINO - MAS NAO VAI DAR CERTO PQ SERA CHAMADO TODA EPOCA	
				
		#No mesmo alinhamento do 'for' dos folds - VERIFICAR SE VAI AQUI MESMO - VERIFICAR A ESTRUTURA
		#plot_metrics_train(epoch_metrics, fold) #Plotar o grafico e salvar o arquivo no diretorio results dentro do Nfold - Qdo acaba o 'for' do treino do fold chama a funcao
		
		heatmaps = args.heat_path + str("heatmaps")  #caminho para as pastas dos heatmaps
		#after each iteration the function transfer the images used in train, val and test steps to results folds		
		transfer_to_results(fold, test_folders, train_folders, val_set_folder, test_set_folder, train_set_folder, metrics_set_folder, heatmaps)  #agora esta criando a nomeclatura errada da pasta
		#depois do LOOP para cada fold chama a funcao e transfere - entender pq comecou a fazer o nome errado no results
	

if __name__ == "__main__":  #Exeution Block

    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    build_record_folder(args)
    tlogger.print()

	#if some image is not in RGB format, convert it to RGB   
    from covertToRgb import convert_to_rgb  #para converter as imagens para RGB
    convert_to_rgb(args)  #chama a funcao 

	#calling the dataset balance
    from Data_Balancing_1 import augment_image
    augment_image(args)
    
	#essa parte eh nova para o cross validation
    from Split1 import create_folds    #chama a funcao que cria o Nfolds
    create_folds(args)   #roda a funcao
	

    main(args, tlogger)
