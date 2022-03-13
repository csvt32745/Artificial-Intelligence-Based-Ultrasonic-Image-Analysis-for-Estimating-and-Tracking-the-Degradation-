import segmentation_models_pytorch as smp
import torch
from network.UnetLSTM import *
from network.Vgg_FCN8s import Single_vgg_FCN8s
# from network.Unet3D  import UNet_3D_Seg, UNet_3D
from network.new_Unet3d import New_UNet3d
from network.models import *
# from UNETR.unetr import UNETR

def WHICH_MODEL(config, frame_continue_num):
    if config.which_model == "FCN":
        net = Single_vgg_FCN8s(1)
        model_name = "FCN"
        
    elif config.which_model == "UNET":
        net = smp.Unet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "Unet"+"_"+config.backbone

    elif config.which_model == "UNET++":
        net = smp.UnetPlusPlus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "UnetPlusPlus"+"_"+config.backbone
        
    elif config.which_model == "PSPNET":
        net = smp.PSPNet(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "PSPNet"+"_"+config.backbone      

    elif config.which_model == "LINKNET":
        net = smp.Linknet (
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "Linknet "+"_"+config.backbone    

    elif config.which_model == "DEEPLABV3+":
        net = smp.DeepLabV3Plus(
            encoder_name=config.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model_name = "DeepLabV3Plus"+"_"+config.backbone

    elif config.which_model == "VNET":
        net = New_UNet3d(in_dim = 3, out_dim = 1, num_filters = config.Unet_3D_channel)
        model_name = "VNet"

    elif config.which_model == "TCSNET":
        net = New_DeepLabV3Plus_LSTM(1, (3, 8), len(frame_continue_num), config.backbone)
        model_name = "TCSNet"+"_"+config.backbone

    elif config.which_model == "TEST":
        net = DeepLabV3Plus_LSTM(len(frame_continue_num), config.backbone)
        model_name = net._get_name()

    elif config.which_model == "TEST2":
        # net = DeepLabV3Plus_TokenViT(len(frame_continue_num), config.backbone)
        net = DeepLabV3Plus_TTT(len(frame_continue_num))
        model_name = net._get_name()
        # model_name = "TEST2"

    elif str(config.which_model).split('/')[0] == 'baseline':
        # net = TemporalBaseline(str(config.which_model).split('/')[1], len(frame_continue_num))
        
        # net = SwinTokener(len(frame_continue_num))
        # model_name = net._get_name()

        tag = str(config.which_model).split('/')[1]
        # tag = 'ttt_fpn'
        # tag = 'tokener'
        net = SwinTokener(len(frame_continue_num), up_conv=tag)
        model_name = net._get_name() + f"_{tag}"

        # net = Swin3D(len(frame_continue_num))
        # model_name = 'Swin3D'
    
    # elif config.which_model == "UNETR":
    #     # net = New_DeepLabV3Plus_LSTM(1, len(frame_continue_num), config.backbone)
    #     net = UNETR(
    #         in_channels=3, out_channels=1, img_size=(352, 416, len(frame_continue_num)),
    #         hidden_size=768, mlp_dim=3072, num_heads=12,
    #         # hidden_size=240, mlp_dim=512, num_heads=12,
    #         feature_size=16, conv_block=False, dropout_rate=0.5)
    #     model_name = "UNETR"
    #     

    else:
        raise NotImplementedError(f"Not Implemented model: \"{config.which_model}\"")

    print(model_name)
    if config.model_path != "":
        net.load_state_dict(torch.load(config.model_path))
        print("Pretrain model loaded!")
    return net, model_name