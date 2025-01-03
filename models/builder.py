import torch
from typing import Union
from torchvision.models.feature_extraction import get_graph_node_names

from .pim_module import pim_module

"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script 
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a 
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""

def load_model_weights(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    #COLOCANDO ESSA LINHA ABAIXO PARA VER O Q DAH - não deu - tem q encontrar .pth qure contenham o state_dict - se não não funciona
    #load_state_dict(state_dict, strict=False)
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        #print("Satate_Dict: ", model.state_dict())
        
        if key in state['state_dict']:  #in load_model_weights - key error- verficar se o state_dict esta no modelo pre-treinado3 -'state_dict' -https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html - ele pega a informação da linha 132
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

####-------TRESNET V2 MILL =====
def build_tresnetl(pretrained: str = "/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/tresnet_m_miil_21k.pth", #Foi substituida por não ter encontrado o formato correto#resnet50_miil_21k.pth",
                   return_nodes: Union[dict, None] = None,
                   num_selects: Union[dict, None] = None, 
                   img_size: int = 448,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Conv",
                   upsample_type: str = "Bilinear",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None):
    
    import timm
    
    if return_nodes is None: #se o retorno de nos for 'vazio'
        return_nodes = {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        }
    if num_selects is None: #se o numero de selecoes for 'vazio' usa esse padrao
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = timm.create_model('tresnet_m_miil_in21k', pretrained=False, num_classes=11221)   #pq 11221 classes? como alterar no eval?... verifdicar se vai treinar com 11221, com 1000 foi
    ### original pretrained path "./models/resnet50_miil_21k.pth"   ##args'tresnet_l.miil_in1k_448'
    if pretrained != "": 
        backbone = load_model_weights(backbone, pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)



#######----------------RESNET50------------------========
def build_resnet50(pretrained: str = "/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/resnet50_miil_21k.pth", #Foi substituida por não ter encontrado o formato correto#resnet50_miil_21k.pth",
                   return_nodes: Union[dict, None] = None,
                   num_selects: Union[dict, None] = None, 
                   img_size: int = 448,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Conv",
                   upsample_type: str = "Bilinear",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None):
    
    import timm
    
    if return_nodes is None: #se o retorno de nos for 'vazio'
        return_nodes = {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4', 
        }
    if num_selects is None: #se o numero de selecoes for 'vazio' usa esse padrao
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = timm.create_model('resnet50', pretrained=False, num_classes=11221)   #pq 11221 classes? como alterar no eval?... verifdicar se vai treinar com 11221, com 1000 foi
    ### original pretrained path "./models/resnet50_miil_21k.pth"
    if pretrained != "":
        backbone = load_model_weights(backbone, pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)

#####CRIANDO UMA FUNÇÃO PARA DENSENET- copiando a estrutura da resnet50
def build_densenet201(pretrained: str = "/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/densenet201-c1103571.pth", #Foi substituida por não ter encontrado o formato correto#resnet50_miil_21k.pth",
                   return_nodes: Union[dict, None] = None,
                   num_selects: Union[dict, None] = None, 
                   img_size: int = 448,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Conv",
                   upsample_type: str = "Bilinear",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None):
    
        
    #CHECAR SE O MODELO CARREGADO TEM O DICIONARIO
    # Load the pretrained model file into a dictionary
    pretrained_dict = torch.load(pretrained)
    #CHECAR AS CHAVES SALVADAS
    #print("pretrained_dict.keys = ",pretrained_dict.keys())
    
    # Check if 'state_dict' is in the dictionary
    if 'state_dict' in pretrained_dict:
    	print("'state_dict' key found in the pretrained model.")
    
    else:
    	print("'state_dict' key not found in the pretrained model.")

    
    import timm
    
    if return_nodes is None:
        return_nodes = {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = timm.create_model('densenet201', pretrained=False, num_classes=1000)   #densenet201 - igual o nome da rede de backbone, verificar esse num_classes - EH CRIADO NO TIMM - verifdicar se esta criando o backbone corretamente - o verificar o timm - nome do arquivo.py - densenet - checar - classes =1000
    ### original pretrained path "./models/resnet50_miil_21k.pth"
    if pretrained != "":
        
        #backbone.load_state_dict(pretrained_dict) #VAI CARREGAR os dicionarios do modelo carregado... verificar se vai funcionar
        # A LINHA ABAIXO EH A ORIGINAL
        backbone = load_model_weights(backbone, pretrained) # se pretraiend tiver "" significa que foi posto um caminho - o caso o caminho da pasta - def build_densenet201 - pq erro?

    # print(backbone)
    # print(get_graph_node_names(backbone))
    
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)


#COPIANDO A FUNÇÃO E ADAPTANDO PARA MOBILINET
def build_mobilenetv3(pretrained: bool = False, #False se for usar carregar o checkpoint
                       return_nodes: Union[dict, None] = None,
                       num_selects: Union[dict, None] = None, 
                       img_size: int = 448,
                       use_fpn: bool = True,
                       fpn_size: int = 512,
                       proj_type: str = "Conv",
                       upsample_type: str = "Bilinear",
                       use_selection: bool = True,
                       num_classes: int = 200,
                       use_combiner: bool = True,
                       comb_proj_size: Union[int, None] = None):

    import torchvision.models as models

    if return_nodes is None:
        return_nodes = {
            'features.4': 'layer1',
            'features.5': 'layer2',
            'features.6': 'layer3',
            'features.7': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = models.mobilenet_v3_large(pretrained=pretrained)  #ATENTAR AO FORMATO DA REDE QUE IRÁ CHAMAR .mobilenet_v3_large - olhar na documentação do pytorch os nomes das redes
    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    ## features.1~features.7

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)
#######################

def build_efficientnet(pretrained: bool = False, #False se for usar carregar o checkpoint
                       return_nodes: Union[dict, None] = None,
                       num_selects: Union[dict, None] = None, 
                       img_size: int = 448,
                       use_fpn: bool = True,
                       fpn_size: int = 512,
                       proj_type: str = "Conv",
                       upsample_type: str = "Bilinear",
                       use_selection: bool = True,
                       num_classes: int = 200,
                       use_combiner: bool = True,
                       comb_proj_size: Union[int, None] = None):

    import torchvision.models as models

    if return_nodes is None:
        return_nodes = {
            'features.4': 'layer1',
            'features.5': 'layer2',
            'features.6': 'layer3',
            'features.7': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    
    backbone = models.efficientnet_b7(pretrained=pretrained)  #mobilenetv2_100 #verificar da onde vem esse tf_efficientnetv2_l_in21k  - efficientnet.py
    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    ## features.1~features.7

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)




def build_vit16(pretrained: str = "/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/vit_base_patch16_224_miil_21k.pth",
                return_nodes: Union[dict, None] = None,
                num_selects: Union[dict, None] = None, 
                img_size: int = 224,  #estava 448 mas no artigo diz q para transformers usar 384
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 10,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None):

    import timm
    
    backbone = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=False)  #Cria o modelo
    ### original pretrained path "./models/vit_base_patch16_224_miil_21k.pth"
    if pretrained != "":
        backbone = load_model_weights(backbone, pretrained)

    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    # 0~11 under blocks

    if return_nodes is None:
        return_nodes = {
            'blocks.8': 'layer1',
            'blocks.9': 'layer2',
            'blocks.10': 'layer3',
            'blocks.11': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    ### Vit model input can transform 224 to another, we use linear
    ### thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    import math
    from scipy import ndimage

    posemb_tok, posemb_grid = backbone.pos_embed[:, :1], backbone.pos_embed[0, 1:]
    posemb_grid = posemb_grid.detach().numpy()
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = img_size//16
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    backbone.pos_embed = torch.nn.Parameter(posemb)

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)


def build_swintransformer(pretrained: bool = True,
                          num_selects: Union[dict, None] = None, 
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          comb_proj_size: Union[int, None] = None):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()
    
    print("Building...")
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = None,
                                   img_size = img_size,
                                   use_fpn = use_fpn,
                                   fpn_size = fpn_size,
                                   proj_type = proj_type,
                                   upsample_type = upsample_type,
                                   use_selection = use_selection,
                                   num_classes = num_classes,
                                   num_selects = num_selects, 
                                   use_combiner = num_selects,
                                   comb_proj_size = comb_proj_size)



###AQUI CHAMA OS MODELOS QUE FORAM DECLARADOS DO YAML - ATENTAR PARA MARCAR E DESMARCAR TODOS QUE FOREM NECESSARIOS
if __name__ == "__main__":
    #VAMOS CRIAR UMA OPÇÃO PARA CHAMAR A DENSENET201 - ATÉ CONSEGUIR ENCONTRAR O .PTH CORRETOS DAS OUTRAS REDES...
    ###DENSENET201 - EXAMPLE - 
	#model = build_densenet201(pretrained = '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM_master/models/densenet201-c1103571.pth')
	#como é uma CNN compiar as configurações das resnet e efficientNet
	#t = torch.randn(1, 3, 448, 448)  #igual das outras CNN
	
    ### ==== resnet50 ====
	model = build_resnet50(pretrained= '/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/resnet50_miil_21k.pth') #densenet201-c1103571.pth')
	
	t = torch.randn(1, 3, 448, 448)
    
    ### ==== swin-t ====
	#model = build_swintransformer(True)   #o Original estava em False  #ATENTAR AO IDENTENTIO - DISTANCIA DOS ESPACOS ENTRE OS LACOS - TABS
	#t = torch.randn(1, 3, 384, 384)  #Valor do tensor

    ### ==== vit ====
	#model = build_vit16(pretrained='/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/vit_base_patch16_224_miil_21k.pth') #puxa a rede pretreinada para usar como Backbone
	#t = torch.randn(1, 3, 448, 448)

    ### ==== efficientNet ====
	#model = build_efficientnet(pretrained=False)
	#t = torch.randn(1, 3, 448, 448)
	
    ### ==== mobilinetV3 ====
	#model = build_mobilinetv3(pretrained=False)
	#t = torch.randn(1, 3, 448, 448)

    ###  ==== TRESNEL L 21K
	#model = build_tresnetl(pretrained='/home/EduardoB/Documentos/Codes/FGIR/FGVC_PIM/models/tresnet_m_miil_21k.pth')
	#t = torch.randn(1, 3, 448, 448)
    

	model.cuda()
    
	t = t.cuda()
	outs = model(t)
	for out in outs:
		print(type(out))
		print("    " , end="")
            
		if type(out) == dict:
			print([name for name in out])


MODEL_GETTER = {
    #criando uma opção para mobilinet - usa as mesmas config de efficient net - verificar 'factor.py' linha 50 a 55 verificar o pq
    "mobilenet":build_mobilenetv3,
    #criando uma opção para densenet - copiar o esquema para a resnet
    "densenet": build_densenet201,
    #verificar se vai funcionar - funcionou
    "resnet50":build_resnet50,
    "swint":build_swintransformer,
    "vit":build_vit16,
    "efficient":build_efficientnet,
    "tresnetl": build_tresnetl
}
