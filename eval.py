import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score  #para Calcular P, R e Fscore too.
from sklearn.metrics import precision_recall_fscore_support

from ignite.metrics import Accuracy, Precision, Recall
from ignite.engine import Engine
from sklearn.metrics import confusion_matrix
from shutil import copyfile #para copiar arquivos


int #, epoch: int, fold: int, fold_metrics): #Foi adicionado epoch: int, fold: int, fold_metrics):
@torch.no_grad()
def cal_train_metrics(args, msg: dict, outs: dict, labels: torch.Tensor, batch_size: int): 
    """
    only present top-1 training accuracy
    """
    ####
    """
    Calculate training metrics including loss, accuracy, precision, recall, and F-score.
    """
    print("BBBBBBBBBBBBBBBBBBBBBBB")
    
    total_loss = 0.0  #Comecando com elas zeradas
    #precision = 0.0  # Initialize the variables before conditional blocks
    #recall = 0.0
    #f_score = 0.0


	# sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')[source])
	#Y_true = Ground truth (correct) - Y_pred = or label indicator array - estimated targets - preditos - 
    if args.use_fpn:
        for i in range(1, 5):
            print("Metrica para USE_FPN")
            acc, precision, recall, fscore = top_k_corrects(outs["layer"+str(i)].mean(1), labels, tops=[1]) # acuracia 'preditos como verdade / todos' - 'TP + TN' / labels
            acc = round(acc["top-1"]/batch_size * 100, 2)
            #print("USE FPN", acc)
          #  input()
            msg["train_acc/layer{}_acc".format(i)] = acc
            loss = F.cross_entropy(outs["layer"+str(i)].mean(1), labels)
            msg["train_loss/layer{}_loss".format(i)] = loss.item()
            total_loss += loss.item()
            
            #chamando funcao para calcular P
            #print("Calculando P, R e F")
            #print("Precision", precision)
            #print("Recall", recall)
            #print("F-score", fscore)
            #input()
            #metric = calc_P_R_Fs(outs["layer"+str(i)].mean(1), labels)  #verificar se vai dar erro
            
            
    if args.use_selection:
        
        for name in outs:
            if "select_" not in name:
                continue
            print("Metrica para USE_SELECTION")
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc, precision, recall, fscore = top_k_corrects(logit, labels_0, tops=[1])
            acc = round(acc["top-1"]/(B*S) * 100, 2)
            print("SELECT", acc)
          #  input()
            msg["train_acc/{}_acc".format(name)] = acc
            labels_0 = torch.zeros([B * S, args.num_classes]) - 1
            labels_0 = labels_0.to(args.device)
            loss = F.mse_loss(F.tanh(logit), labels_0)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()
            
                       
            #print(f'LOGIT: {logit}')
            
            
        for name in outs:
            if "drop_" not in name:
                continue
            B, S, _ = outs[name].size()
            logit = outs[name].view(-1, args.num_classes)
            labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
            acc, precision, recall, fscore = top_k_corrects(logit, labels_1, tops=[1])
            acc = round(acc["top-1"]/(B*S) * 100, 2)
            print("DROP", acc)
           # input()
            msg["train_acc/{}_acc".format(name)] = acc
            loss = F.cross_entropy(logit, labels_1)
            msg["train_loss/{}_loss".format(name)] = loss.item()
            total_loss += loss.item()
            
            
    if args.use_combiner:
        print("Metrica para USE_COMBINER")
        acc, precision, recall, fscore = top_k_corrects(outs['comb_outs'], labels, tops=[1])  #usando o top_k_corrects para buscar os preditos e as labels
        acc = round(acc["top-1"]/batch_size * 100, 2)  #Aqui acuracia
        print("COMBINER", acc)
        #input()
        msg["train_acc/combiner_acc"] = acc
        loss = F.cross_entropy(outs['comb_outs'], labels)
        msg["train_loss/combiner_loss"] = loss.item()
        total_loss += loss.item()  #Total_loss
        
                
    if "ori_out" in outs:
        acc = top_k_corrects(outs["ori_out"], labels, tops=[1])["top-1"] / batch_size
        acc = round(acc * 100, 2)
        msg["train_acc/ori_acc"] = acc
        loss = F.cross_entropy(outs["ori_out"], labels)
        msg["train_loss/ori_loss"] = loss.item()
        total_loss += loss.item()
            
    return msg #retornando as metricas


@torch.no_grad()
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):   #se o dispositivo for diferente de cpu eh uma gpu
        preds = preds.cpu()  #pega os preditos daqui
    if labels.device != torch.device('cpu'):  #pega as labels daqui
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    #print("PREDS", preds)
    #print("LABELS", labels)
    #input()
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, sorted_preds[:, 0], average="macro")
    return corrects, precision, recall, fscore

@torch.no_grad()
def top_k_corrects_II(preds: torch.Tensor, labels: torch.Tensor, tops: list[1,3,5]): #list =   #usado para eval
    """	
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'): #se o dispositivo for diferente de cpu eh uma gpu
        preds = preds.cpu()
    if labels.device != torch.device('cpu'): #pega as labels daqui
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    #Vai ficar comentado por eqto ainda
    precision_eval, recall_eval, fscore_eval, _ = precision_recall_fscore_support(labels, sorted_preds[:,0], average="macro") #chamando igual ao k_corrects
    return corrects, precision_eval, recall_eval, fscore_eval

@torch.no_grad()
def top_k_corrects_III(preds: torch.Tensor, labels: torch.Tensor, tops: list[1,3,5]): #list =   #usado para eval
    """	
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'): #se o dispositivo for diferente de cpu eh uma gpu
        preds = preds.cpu()
    if labels.device != torch.device('cpu'): #pega as labels daqui
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    #Vai ficar comentado por eqto ainda
    precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(labels, sorted_preds[:,0], average="macro") #chamando igual ao k_corrects
    return corrects, precision_test, recall_test, fscore_test

##ESSA USANDO PARA VALIDACAO
@torch.no_grad()  #Funcao para fazer as metricas na etapa de validacao
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    #corrects eh dicionario - recebe o que predizeu correto
	#total samples - eh o tamanho do lote em analise
	# This is a PyTorch tensor containing the raw model outputs or logits.
	#labels eh os valores corretor - seriam os TP
	 #A string that represents the name or identifier for the current evaluation metric being calculated.

    
    tmp_score = torch.softmax(logits, dim=-1)  #na linha abaixo inserido precision_eval, recall_eval, fscore_eval para separar o que eh eval e o q eh numero inteiro
    tmp_corrects, precision_eval, recall_eval, fscore_eval = top_k_corrects_II(tmp_score, labels, tops=[1,3]) # return top-1, top-3, top-5 accuracy 
				#tmp_score vai entrar como os preditos, labels como labels e top [1,3] como a lista dos tops results
				#esta chamando top_k_II 
    #Pega os valores que foram predizidos corretos e faz o loop para contar os corretos - TP
    ### each layer's top-1, top-3 accuracy
    #print(f'TMP_CORRECTS: {tmp_corrects}')
    for name,_ in tmp_corrects.items():  #.items - desempacotando o dicionario do tmp. corrects
        #print(f"THIS NAME: {this_name}")
        #print(f"THIS NAME TYPE: {this_name.type}")
        #print(f"NAME: {name}")
        #print(f'NAME.TYPE: {name.type}')
        #print("PERTA BOTAO")
        #input()
        eval_name = this_name + "-" + str(name)  #um esta vindo como string outro como dicionario -
        #aparentemente 'this_name' vem como str e name parece um dic
        
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        
        #print(f'NAME: {name}')
        #print("Perta Butao")
        #input()
        corrects[eval_name] += tmp_corrects[name]
        #esta tentando acessar os nomes do top 1 e top 3 e esta dizendo quee eh um dict nap int e nem slice
        total_samples[eval_name] += labels.size(0)  #o total de exemplos para calculo recebe o tamanho da qtd que tem em labels
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


###ESSA USANDO PARA TESTE
##ESSA USANDO PARA VALIDACAO
@torch.no_grad()  #Funcao para fazer as metricas na etapa de validacao
def _cal_test_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    #corrects eh dicionario - recebe o que predizeu correto
	#total samples - eh o tamanho do lote em analise
	# This is a PyTorch tensor containing the raw model outputs or logits.
	#labels eh os valores corretor - seriam os TP
	 #A string that represents the name or identifier for the current evaluation metric being calculated.

    
    tmp_score = torch.softmax(logits, dim=-1)  #na linha abaixo inserido precision_eval, recall_eval, fscore_eval para separar o que eh eval e o q eh numero inteiro
    tmp_corrects, precision_test, recall_test, fscore_test = top_k_corrects_III(tmp_score,	 labels, tops=[1,3]) # return top-1, top-3, top-5 accuracy 
				#tmp_score vai entrar como os preditos, labels como labels e top [1,3] como a lista dos tops results
				#esta chamando top_k_II 
    #Pega os valores que foram predizidos corretos e faz o loop para contar os corretos - TP
    ### each layer's top-1, top-3 accuracy
    #print(f'TMP_CORRECTS: {tmp_corrects}')
    for name,_ in tmp_corrects.items():  #.items - desempacotando o dicionario do tmp. corrects
        #print(f"THIS NAME: {this_name}")
        #print(f"THIS NAME TYPE: {this_name.type}")
        #print(f"NAME: {name}")
        #print(f'NAME.TYPE: {name.type}')
        #print("PERTA BOTAO")
        #input()
        eval_name = this_name + "-" + str(name)  #um esta vindo como string outro como dicionario -
        #aparentemente 'this_name' vem como str e name parece um dic
        
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        
        #print(f'NAME: {name}')
        #print("Perta Butao")
        #input()
        corrects[eval_name] += tmp_corrects[name]
        #esta tentando acessar os nomes do top 1 e top 3 e esta dizendo quee eh um dict nap int e nem slice
        total_samples[eval_name] += labels.size(0)  #o total de exemplos para calculo recebe o tamanho da qtd que tem em labels
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)

@torch.no_grad()
def _average_top_k_result(corrects: dict, total_samples: dict, scores: list, labels: torch.Tensor, tops: list = [1, 2, 3, 4, 5]):
    """
    scores is a list contain:
    [
        tensor1, 
        tensor2,...
    ] tensor1 and tensor2 have same size [B, num_classes]
    """
    # initial
    for t in tops:
        eval_name = "highest-{}".format(t)
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        total_samples[eval_name] += labels.size(0)

    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    
    batch_size = labels.size(0)
    scores_t = torch.cat([s.unsqueeze(1) for s in scores], dim=1) # B, 5, C

    if scores_t.device != torch.device('cpu'):
        scores_t = scores_t.cpu()

    max_scores = torch.max(scores_t, dim=-1)[0]
    # sorted_ids = torch.sort(max_scores, dim=-1, descending=True)[1] # this id represents different layers outputs, not samples

    for b in range(batch_size):
        tmp_logit = None
        ids = torch.sort(max_scores[b], dim=-1)[1] # S
        for i in range(tops[-1]):
            top_i_id = ids[i]
            if tmp_logit is None:
                tmp_logit = scores_t[b][top_i_id]
            else:
                tmp_logit += scores_t[b][top_i_id]
            # record results
            if i+1 in tops:
                if torch.max(tmp_logit, dim=-1)[1] == labels[b]:
                    eval_name = "highest-{}".format(i+1)
                    corrects[eval_name] += 1


#### === VALIDACAO === ####
def evaluate(args, model, val_loader):  #recebe argumentos, o modelo o mesmo do treino e carrega os dados de val_loader
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under 
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~ 
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """
    
    all_preds = []   #para salvar os preditos e os labels de cada step
    
    all_labels = []

    model.eval()  #chama a validacao do modelo
    corrects = {}  #salva os corretos - comeca com vazio
    total_samples = {}  #total de exemplos - comeca com vazio

    total_batchs = len(val_loader) # just for log  -  verificar como foi feito no train
    show_progress = [x/10 for x in range(11)] # just for log
    progress_i = 0

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(val_loader):
            
            all_labels +=  labels.tolist()  #somando as labels

            score_names = [] #score para nomes
            scores = [] #total de scores
            datas = datas.to(args.device)

            outs = model(datas)  #recebe as saidas

            if args.use_fpn:  #se usa fpn
                for i in range(1, 5):
                    this_name = "layer" + str(i)   #esta enviando como dicionario
                    _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)
					#chama o calculo das metricas de validacao
					#essa parte aqui substitui a parte de calculo que esta no train

            ### for research
            if args.use_selection: #se usa seletores
                for name in outs:
                    if "select_" not in name:
                        continue #
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_1, this_name)
                    #chama o calculo das metricas de validacao
                
                for name in outs:  #passa por todos os nomes que estao na saida
                    if "drop_" not in name:
                        continue
                    this_name = name
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, args.num_classes)
                    labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
                    _cal_evalute_metric(corrects, total_samples, logit, labels_0, this_name)
                    #chama o calculo das metricas de validacao

            if args.use_combiner:   #se usa combinador
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)
                #chama o calculo das metricas de validacao
				#verificar se vai dar certo - se nao talvez enviar para dentro da funcao _cal_evaluate_metrics
                comb_preds = outs["comb_outs"]
                comb_preds = torch.sort(comb_preds, dim=-1, descending=True)[1]
                comb_preds = comb_preds[:, 0]
                all_preds += comb_preds.tolist()

            if "ori_out" in outs:
                this_name = "original"
                _cal_evalute_metric(corrects, total_samples, outs["ori_out"], labels, this_name) #
                #chama o calculo das metricas de validacao - eh o mesmo padrao para todas elas
        
            _average_top_k_result(corrects, total_samples, scores, labels)
            #veriifcar essa funcao - a princio

            eval_progress = (batch_id + 1) / total_batchs
            
            if eval_progress > show_progress[progress_i]:
                print(".."+str(int(show_progress[progress_i]*100))+"%", end='', flush=True)
                progress_i += 1

        """ calculate accuracy """
        # total_samples = len(test_loader.dataset)
        
        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            
			### only compare top-1 accuracy

            if "top-1" in name or "highest" in name:
                if acc >= best_top1:
                    best_top1 = acc
                    best_top1_name = name
	#colocar + os retornos do P, R e F aqui - o calculo sera feito no loop do eval
    
    precision_eval, recall_eval, fscore_eval, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    return best_top1, best_top1_name, eval_acces, precision_eval, recall_eval, fscore_eval

#function to find the image path from the image name
def find_image_path(root_folder, image_name):
	if isinstance(image_name, torch.Tensor): #isinstance - retorna True se o objeto especificado for do tipo especificado, caso contrario False	)
		image_name = str(image_name) #.item()) #.item() - Returns the value of this tensor as a standard Python number. This only works for tensors with one element. For other cases, see tolist().
	#use glob to find the image file subfolder
	elif not isinstance(image_name, str):
		raise ValueError("image_name must be a string or torch.Tensor, Invalid type: {}".format(type(image_name)))
     
	#use os.walk to travesse all subfolders
	for folder, _, files in os.walk(root_folder): #os.walk - gera os nomes dos arquivos em uma arvore de diretorio
		if image_name in files: #if the image is found in the current folder
			image_path = os.path.join(folder, image_name)	#os.path.join - junta um ou mais caminhos de forma inteligente
			return image_path

	#pattern = os.path.join(root_folder, "**", image_name) #** - significa que vai procurar em todos os subdiretorios
	#image_way = glob.glob(pattern, recursive=True)  #glob.glob - retorna uma lista de caminhos de arquivos que correspondem ao padrao
	#if image is not found in any directory, return None
	return None #return the image path	
     

###  ==== ETAPA DE TESTE ====###
def test(args, model, test_loader):
	"""
	Run the test step using the provided model and test loader.
	Calculate accuracy metrics for the test predictions.
	"""
	
	all_preds = []   #para salvar os preditos e os labels para cada folder
	all_labels = []
     
	all_preds_class = []  #para salvar os preditos e os labels para cada classe
	all_preds_labels = []
     
	model.eval()  # Set the model to evaluation mode
	corrects = {}   #para salvar as predicoes corretas
	total_samples = {}  #total de exemplos
	
	total_batchs = len(test_loader)  # just for log
	show_progress = [x / 10 for x in range(11)]  # just for log
	progress_i = 0
    
	num_classes =  args.num_classes #len(test_loader.dataset.classes) #get the number of classes from the dataset
     
	clas_precisions = [0.0] * num_classes
	clas_recalls = [0.0] * num_classes
	clas_fscores = [0.0] * num_classes #initialize the class-wise metrics to zero
    
	image_names_mapping = [] #initialize the image names mapping list
	
	with torch.no_grad(): #no_grad()" is like a loop where every tensor inside the loop will have requires_grad set to False. It means any tensor with gradient currently attached with the current computational graph is now detached from the current graph
		""" accumulate """
		
		for root, dirs, files in os.walk(args.test_root): #os.walk - gera os nomes dos arquivos em uma arvore de diretorio
			for file in files: #for each file in the current directory	
				if file.endswith((".jpg", ".png", ".RAW")):
					image_names_mapping.append(os.path.join(root, file)) #append the image name to the list
					#print(f'IMAGE NAMES MAPPING: {image_names_mapping}')
					#input()
			
	
		for batch_id, (ids, datas, labels) in enumerate(test_loader):  #para rodar dentro dos tamanhos dos lotes
			#print(f'IDS: {ids}')
			#input() #after to test must be removed or commented =)
			score_names = []
			scores = []
			datas = datas.to(args.device)
		
			outs = model(datas)
			all_labels += labels.tolist()
	
			if args.use_fpn:  #se estiver usando fpn
				for i in range(1, 5):
					this_name = "layer" + str(i)
					_cal_test_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)  #chamando as metricas
			### for research
			if args.use_selection:  #se usar seletor
				for name in outs:
					if "select_" not in name: #se nao for vazio
						continue
                				
					this_name = name #UM PARA TRAS
					S = outs[name].size(1)
					logit = outs[name].view(-1, args.num_classes)
					labels_1 = labels.unsqueeze(1).repeat(1, S).flatten(0)
					_cal_test_metric(corrects, total_samples, logit, labels_1, this_name)
                		
				for name in outs:  #se o nome estiver na saida #UM PARA TRAS
					if "drop_" not in name:
						continue
                			
					this_name = name #NA LINHA DO IF
					S = outs[name].size(1)
					logit = outs[name].view(-1, args.num_classes)
					labels_0 = labels.unsqueeze(1).repeat(1, S).flatten(0)
					_cal_test_metric(corrects, total_samples, logit, labels_0, this_name)
		
			if args.use_combiner:
				this_name = "combiner"
				_cal_test_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)
				
				comb_preds = outs["comb_outs"].cpu().numpy() #get the combiner predictions, and move to cpu
				comb_preds_tensor = torch.tensor(comb_preds) #convert to tensor
				comb_preds_tensor = torch.sort(comb_preds_tensor, dim=-1, descending=True)[1] #sort the predictions
				comb_preds = comb_preds_tensor.cpu().numpy() #Move back to numpy array
				comb_preds = comb_preds[:, 0]
				all_preds += comb_preds.tolist() #ate essa parte eh para calcular por dobra
                
				#by code line below is to calculate the metric for each class
				precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(labels.cpu().numpy(), comb_preds, labels=list(range(num_classes)), average=None)
				#the error is coming from the above line, - TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
				
				#Now to get the image names of TP, FP, TN and FN for each class
				conf_matrix = confusion_matrix(labels.cpu().numpy(), comb_preds, labels=list(range(num_classes))) #get the confusion matrix
								
				# Update class-wise metrics
				tn_images = [] #initialize the list of TN images
				tp_images = [] #initialize the list of TP images
				fp_images = [] #initialize the list of FP images
				fn_images = [] #initialize the list of FN images
                
				for class_idx in range(num_classes):
					clas_precisions[class_idx] += precision_class[class_idx]
					clas_recalls[class_idx] += recall_class[class_idx]  #This line use the recall_class to calculate the recall for each class - 
					clas_fscores[class_idx] += fscore_class[class_idx]

					#get the TP, FP, FN and TN for the class     
					TP = conf_matrix[class_idx, class_idx] #get the TP for the class
					FP = sum(conf_matrix[:, class_idx]) - TP #get the FP for the class
					FN = sum(conf_matrix[class_idx, :]) - TP #get the FN for the class
					TN = sum(sum(conf_matrix)) - TP - FP - FN #get the TN for the class
                         
					#get the image names for TP, FP, FN and TN for the class
					#with the information below we will use it to generate heatmaps from the images classification, in heat.py
					#tp_images = [image_names[i] for i in range(len(labels)) if labels[i] == class_idx and comb_preds[i] == class_idx]
					print(f'Len IDS: {len(ids)}') #remove after test
					print(f'Len Labels: {len(labels)}')
					print(f'Comb_preds: {comb_preds}')
					#input()
					#tp_images = [image_names_mapping[ids[i].item()] for i in range(len(labels)) if labels[i] == class_idx and comb_preds[i] == class_idx] #get the image names for TP
					#tp_images = [ids[i] for i in range(len(labels)) if labels[i] == class_idx and comb_preds[i] == class_idx] #get the image names for TP
					#fp_images = [image_names_mapping[ids[i].item()] for i in range(len(labels)) if labels[i] != class_idx and comb_preds[i] == class_idx] #
					#fp_images = [ids[i] for i in range(len(labels)) if labels[i] != class_idx and comb_preds[i] == class_idx] #get the image names for FP	
					#fn_images = [image_names_mapping[ids[i].item()] for i in range(len(labels)) if labels[i] == class_idx and comb_preds[i] != class_idx] #get the image names for FN
					#fn_images = [ids[i] for i in range(len(labels)) if labels[i] == class_idx and comb_preds[i] != class_idx] #get the image names for FN	
					#tn_images = [image_names_mapping[ids[i].item()] for i in range(len(labels)) if labels[i] != class_idx and comb_preds[i] != class_idx] #get the image names for TN
					#tn_images = [ids[i] for i in range(len(labels)) if labels[i] != class_idx and comb_preds[i] != class_idx] #get the image names for TN	
					print(f'Len IDS Before Problematic Line: {len(ids)}') #remove after test - this line is to check the len of ids
					print(f'Len Labels Before Problematic Line: {len(labels)}')
					print(f'Comb_preds Before Problematic Line: {comb_preds}')
					#input()
					for i in range(min(len(labels), len(comb_preds))):
						try:
							if labels[i] == class_idx and comb_preds[i] == class_idx:
								tp_images.append(image_names_mapping[ids[i].item()])
							elif labels[i] != class_idx and comb_preds[i] == class_idx:
								fp_images.append(image_names_mapping[ids[i].item()])
							elif labels[i] == class_idx and comb_preds[i] != class_idx:
								fn_images.append(image_names_mapping[ids[i].item()])
							elif labels[i] != class_idx and comb_preds[i] != class_idx:
								tn_images.append(image_names_mapping[ids[i].item()])
						except IndexError:
							print(f"IndexError at i={i}, len(ids)={len(ids)}, len(labels)={len(labels)}, len(comb_preds)={len(comb_preds)}")
						
						#except IndexError:
						#	print(f"IndexError at i={i}, len(ids)={len(ids)}, len(labels)={len(labels)}, len(comb_preds)={len(comb_preds)}")
					#tn_images = [image_names_mapping[ids[i].item()] for i in range(min(len(labels), len(comb_preds))) if labels[i] != class_idx and comb_preds[i] != class_idx]

					#now save the images in a directory
					#create folders for TP, FP, FN and TN
					output_folder = args.heat_path #get the output folder
					output_folder_heat = os.path.join(output_folder, "heatmaps") #get the output folder for the heatmaps
					tp_folder = os.path.join(output_folder_heat, "TP") #get the output folder for the TP
					fp_folder = os.path.join(output_folder_heat, "FP")
					fn_folder = os.path.join(output_folder_heat, "FN")
					tn_folder = os.path.join(output_folder_heat, "TN")
					
					#building the folder for each class in each fold
					for folder in [output_folder_heat,tp_folder, fp_folder, fn_folder, tn_folder]: #for each folder
						if not os.path.exists(folder):
							os.makedirs(folder)

					#saving the images in the respective folder and save the original image class in the name
					#print(f'TP_Images: {tp_images}') #print the TP images names
					#input()
					#print(f'FP_Images: {fp_images}')
					#input()
					#print(f'FN_Images: {fn_images}')
					#input()
					#print(f'TN_Images: {tn_images}')
					#input()
	
					for image_path in tp_images:
						#save the original class
						#original_class = extract_original_class(image) #get the original class of the image
						#print(f'IMAGE before find_image_path: {image_path}')
						#input()
						#check if the image_path is not None and if the image folder exist
						if os.path.exists(image_path):
							save_path = os.path.join(tp_folder, os.path.basename(image_path)) #tp_folder
							copyfile(image_path, save_path) #copy the image to the respective folder
													
						else:
							print(f"Image path is None or image folder does not exist: {image_path}")
											
					for image_path in fp_images:
						#print(f'IMAGE before find_image_path: {image_path}')
						#input()
						
						#check if the image_path is not None and if the image folder exist
						if os.path.exists(image_path):
							save_path = os.path.join(fp_folder, os.path.basename(image_path))
							copyfile(image_path, save_path) #copy the image to the respective folder
							#print(f'Image copied to the respective folder: {image_path}') #remove this line after test
						
						else:
							print(f"Image path is None or image folder does not exist: {image_path}")
                    
					for image_path in fn_images:
						#print(f'IMAGE before find_image_path: {image_path}')
						#input()
						
						#check if the image_path is not None and if the image folder exist
						if os.path.exists(image_path):
							save_path = os.path.join(fn_folder, os.path.basename(image_path))
							copyfile(image_path, save_path)
							#print(f'Image copied to the respective folder: {image_path}')                              
						
						else:
							print(f"Image path is None or image folder does not exist: {image_path}")
							
					for image_path in tn_images:
						#print(f'IMAGE before find_image_path: {image_path}')
						#input()	
						
						if os.path.exists(image_path):
							save_path = os.path.join(tn_folder, os.path.basename(image_path))
							copyfile(image_path, save_path)
							#print(f'Image copied to the respective folder: {image_path}')
						
						else:
							print(f"Image path is None or image folder does not exist: {image_path}")

					#now save the images in a list
					all_preds_class.append([tp_images, fp_images, fn_images, tn_images]) #save the images names for each class
                        


				all_preds_class += comb_preds.tolist()  #salvando os preditos e os labels para cada classe
																#
			if "ori_out" in outs:
				this_name = "original"
				_cal_test_metric(corrects, total_samples, outs["ori_out"], labels, this_name)
			
			_average_top_k_result(corrects, total_samples, scores, labels)

			eval_progress = (batch_id + 1) / total_batchs
		
			if eval_progress > show_progress[progress_i]:
				print(".." + str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
				progress_i += 1
			
		""" calculate accuracy """
		best_top1 = 0.0
		best_top1_name = ""
		test_acces = {}
	
		for name in corrects:
			acc = corrects[name] / total_samples[name]
			acc = round(100 * acc, 3)
			test_acces[name] = acc
			# Only compare top-1 accuracy
		
			if "top-1" in name or "highest" in name:
				if acc >= best_top1:
					best_top1 = acc
					best_top1_name = name

	print("Test Step Completed\n")
	#return best_top1, best_top1_name, test_acces   #deixar comentado essa linha pq como eh teste nao precisa retornar nada - somente os graficos
	#calculate the average metrics for each class
	class_precisions = [prec / len(test_loader) for prec in clas_precisions]
	class_recalls = [rec / len(test_loader) for rec in clas_recalls]
	class_fscores = [fsc / len(test_loader) for fsc in clas_fscores]

	precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(all_labels, all_preds, average = "macro")
     
	return precision_test, recall_test, fscore_test, class_precisions, class_recalls, class_fscores  #retornando os valores
				
    


def evaluate_cm(args, model, test_loader):
    """
    [Notice: Costom Model]
    If you use costom model, please change fpn module return name (under
    if args.use_fpn: ...)
    [Evaluation Metrics]
    We calculate each layers accuracy, combiner accuracy and average-higest-1 ~
    average-higest-5 accuracy (average-higest-5 means average all predict scores
    as final predict)
    """

    model.eval()
    corrects = {}
    total_samples = {}
    results = []

    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(test_loader):

            score_names = []
            scores = []
            datas = datas.to(args.device)
            outs = model(datas)

            # if args.use_fpn and (0 < args.highest < 5):
            #     this_name = "layer" + str(args.highest)
            #     _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if args.use_combiner:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            # _average_top_k_result(corrects, total_samples, scores, labels)

            for i in range(scores[0].shape[0]):
                results.append([test_loader.dataset.data_infos[ids[i].item()]['path'], int(labels[i].item()),
                                int(scores[0][i].argmax().item()),
                                scores[0][i][scores[0][i].argmax().item()].item()])  # 图片路径，标签，预测标签，得分

        """ wirte xlsx"""
        writer = pd.ExcelWriter(args.save_dir + 'infer_result.xlsx')
        df = pd.DataFrame(results, columns=["id", "original_label", "predict_label", "goal"])
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.save()
        writer.close()

        """ calculate accuracy """
        best_top1 = 0.0
        best_top1_name = ""
        eval_acces = {}
        for name in corrects:
            acc = corrects[name] / total_samples[name]
            acc = round(100 * acc, 3)
            eval_acces[name] = acc
            ### only compare top-1 accuracy
            if "top-1" in name or "highest" in name:
                if acc > best_top1:
                    best_top1 = acc
                    best_top1_name = name
                    
        #output_folder = args.data_path #get the output folder
		#			output_folder_heat = os.path.join(output_folder, "heatmaps") #get the output folder for the heatmaps
		#			tp_folder = os.path.join(output_folder_heat, "TP") #get the output folder for the TP
        #    acc = round(100 * acc, 3)
         #   eval_acces[name] = acc
         #   ### only compare top-1 accuracy
        ##    if "top-1" in name or "highest" in name:
         #       if acc > best_top1:
          #          best_top1 = acc
           #         best_top1_name = name

        """ wirte xlsx"""
        results_mat = np.mat(results)
        y_actual = results_mat[:, 1].transpose().tolist()[0]
        y_actual = list(map(int, y_actual))
        y_predict = results_mat[:, 2].transpose().tolist()[0]
        y_predict = list(map(int, y_predict))

        folders = os.listdir(args.val_root)
        folders.sort()  # sort by alphabet
        print("[dataset] class:", folders)
        df_confusion = confusion_matrix(y_actual, y_predict)
        plot_confusion_matrix(df_confusion, folders, args.save_dir + "infer_cm.png", accuracy=best_top1)

    return best_top1, best_top1_name, eval_acces


@torch.no_grad()
def eval_and_save(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "eval_results.txt", "w") as ftxt:
        ftxt.write(msg)


@torch.no_grad()
def eval_and_cm(args, model, val_loader, tlogger):
    tlogger.print("Start Evaluating")
    acc, eval_name, eval_acces = evaluate_cm(args, model, val_loader)
    tlogger.print("....BEST_ACC: {} {}%".format(eval_name, acc))
    ### build records.txt
    msg = "[Evaluation Results]\n"
    msg += "Project: {}, Experiment: {}\n".format(args.project_name, args.exp_name)
    msg += "Samples: {}\n".format(len(val_loader.dataset))
    msg += "\n"
    for name in eval_acces:
        msg += "    {} {}%\n".format(name, eval_acces[name])
    msg += "\n"
    msg += "BEST_ACC: {} {}% ".format(eval_name, acc)

    with open(args.save_dir + "infer_results.txt", "w") as ftxt:
        ftxt.write(msg)


def plot_confusion_matrix(cm, label_names, save_name, title='Confusion Matrix acc = ', accuracy=0):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(len(label_names) / 2, len(label_names) / 2), dpi=100)
    np.set_printoptions(precision=2)
    # print("cm:\n",cm)

    # 统计混淆矩阵中每格的概率值
    x, y = np.meshgrid(np.arange(len(cm)), np.arange(len(cm)))
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        try:
            c = (cm[y_val][x_val] / np.sum(cm, axis=1)[y_val]) * 100
        except KeyError:
            c = 0
        if c > 0.001:
            plt.text(x_val, y_val, "%0.1f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title + str('{:.3f}'.format(accuracy)))
    plt.colorbar()
    plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(label_names))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(save_name, format='png')
    # plt.show()
    
