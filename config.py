import configargparse

def config_parser_train():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')

    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--which_model', type=str, default="TCSNET")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_model_path', type=str, default="./My_Image_Segmentation/models/")
    parser.add_argument('--save_log_path', type=str, default="./My_Image_Segmentation/log/")
    parser.add_argument('--best_score', type=float, default=0.7)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--train_data_path', type=str, default="Medical_data/train/")
    parser.add_argument('--valid_data_path', type=str, default="Medical_data/valid/")
    parser.add_argument('--test_data_path', type=str, default="Medical_data/test/")
    parser.add_argument('--backbone', type=str, default="resnet34")
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--draw_temporal', type=int, default=0)
    parser.add_argument('--draw_image_path', type=str, default="Medical_data/test_image_output/")
    parser.add_argument('--Unet_3D_channel', type=int, default= 8)
    parser.add_argument('--continue_num', nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--data_parallel', type=int, default=0)
    parser.add_argument('--w_T_LOSS', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--boundary_pixel', type=int, default=0)
    config = parser.parse_args()
    return config

def config_parser_test():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    
    parser.add_argument('--which_model', type=str, default="TCSNET")
    parser.add_argument('--backbone', type=str, default="resnet34")
    parser.add_argument('--video_path', type=str, default="input_video")
    parser.add_argument('--output_img_path', type=str, default="output_frame")
    parser.add_argument('--model_path', type=str, default="pretrained_model.pt")
    parser.add_argument('--output_path', type=str, default="output_prediction")
    parser.add_argument('--keep_image', type= int, default=1)
    parser.add_argument('--continuous', type=int, default=1)
    parser.add_argument('--distance', type=int, default=50)
    parser.add_argument('--interval_num', type=int, default=5)
    parser.add_argument('--continue_num', nargs="+", default=[-3, -2, -1, 0, 1, 2, 3])
    config = parser.parse_args()
    return config