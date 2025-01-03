#import libraries
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
#import torch

from .randaug import RandAugment  #NESSE ARQUIVO TERA Q INSERIR O BALANCEAMENTO DE DADOS - DATA AUGMENTAION - esse arquivo ainda nao esta sendo usado no codigo - o programa faz DA automatico nesse codigo do dataset

#vai ter q alterar um pouco o build_loader para a cross_validation funcionar
def build_loader(args): #Essa funcao eh chamada no main.py tras as informacoes de args que vem da funcao get_args do config_utils.py
    
    train_set, train_loader = None, None  #comeca zerado
    if args.train_root is not None:  #args vem do main.py - se nao estiver vazia - vai ter que alterar para conseguir ler o caminho do train gerado pelo spli (cross validation)
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)  #vem da classe ImageDataset
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None  #começa zerado - checa val_root e train_root para verificar da onde o train e o val sets estao vindo - e gerar o train e val_loader
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    
    test_set, test_loader = None, None   #comecando zerado
    if args.test_root is not None: #se nao estiver vazio
    	test_set = ImageDataset(istrain=False, root=args.test_root, data_size=args.data_size, return_index=True)
    	test_loader = torch.utils.data.DataLoader(test_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    
    #aqui vai o if args.test_root is not None:
 

    return train_loader, val_loader , test_loader #essa linha retorna e vai para 0 main.py no buildind data loader
    
    #Para o split funcionar:
    #1- Um script (Split1.py) para dividir em Nfold - pode passar via yaml - testar
    #2- Depois que separar em Nfolds - Para o corrente fold gerar train, test e validation (x% do train via yaml) para (train_test_split.py) - aqui que esta o segredo - vai ter que chamar o arquivo ou colocar ele aqui dentro em forma de funcao?
    #3-Para cada experimento rodar primeiro o Split1.py - divide em Nfolds - roda somente uma vez
    #4- Depois de dividido em Nfolds - roda o train_split_folds para cada Nfold - colocar em forma de def e chamar ele toda vez... veriifcar como fazer isso
    
    
    

def get_dataset(args): #essa funcao vai para algum outro .py - busca a informacao no yaml - para ela funcionar o Split jah devera ter sido realizado
    if args.train_root is not None:  #se nao for vazio
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):  #a dataset para image data
	
	#constructor __init__, inicia o dataset baseado nos parametros que recebeu
    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain: #to insert DataAugmentation in to train data
            
            #transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)   ##ESSA LINHA AQUI PODE SER ALTERADA PARA CHAMAR O randaug.py e fazer o balanceamento conforme classe - ou utilizar o randaug.py e, outra funcao e receber aqui ja balanceado...
            #RandAugment(n=2, m=3, img_size=sub_data_size)

            #ESSA PARTE FAZ PARA TODAS AS IMAGENS QUE VAO PARA O TREINO - O BALANCEAMENTO PODE SER FEITO ANTES - PODEMOS USAR O RANDAUG.PY ANTES EM OUTRA FUNCAO
            #
            print("Data Augmentation... Resizing, RandomHorizontalFlip, RandomVerticalFlip, Pad, ToTensor, Normalize")
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),  #510,510
                        #transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        #transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.RandomVerticalFlip(),
                        transforms.Pad(10),
						transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),  #510,510
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root): #method reads the data information from the provided root directory. 
    
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)  #method returns the total number of data samples in the dataset.

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:  #method is responsible for loading and preprocessing a single data sample at the given index. 
            # return index, img, sub_imgs, label, sub_boundarys
            # It reads the image using OpenCV, applies transformations, and returns the preprocessed image and its label.
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
