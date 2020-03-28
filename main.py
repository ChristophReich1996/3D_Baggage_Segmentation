from argparse import ArgumentParser

# Process command line arguments
parser = ArgumentParser()

parser.add_argument('--train', type=int, default=1,
                    help='Train network (default=1 (True)')

parser.add_argument('--test', type=int, default=1,
                    help='Test network (default=1 (True)')

parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size of the training and test set (default=32)')

parser.add_argument('--lr', type=float, default=1e-03,
                    help='Main learning rate of the adam optimizer (default=1e-04)')

parser.add_argument('--gpus_to_use', type=str, default='0',
                    help='Indexes of the GPUs to be use (default=0)')

parser.add_argument('--use_data_parallel', type=int, default=0,
                    help='Use multiple GPUs (default=0 (False)')

parser.add_argument('--epochs', type=int, default=200,
                    help='Epochs to perform while training (default=100)')

parser.add_argument('--use_cat', type=int, default=1,
                    help='True if concatenation should be utilized in O-Net (default=1 (True))')

parser.add_argument('--use_cbn', type=int, default=1,
                    help='True if conditional batch normalization should be utilized in O-Net (default=1 (True))')

parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'dice', 'focal'],
                    help='Loss function to be used (default=cross_entropy (cross_entropy, dice or focal))')

parser.add_argument('--small_encoder', type=int, default=0, choices=[0, 1],
                    help='If true a smaller encoder is utilized')

parser.add_argument('--load_model', type=str, default=None,
                    help='Path to model to be loaded (default=None)')

args = parser.parse_args()

import os

# Batch size has to be a factor of the number of devices used in data parallel
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_to_use

import torch
from torch.utils.data.dataloader import DataLoader

import Models
import Datasets
from ModelWrapper import OccupancyNetworkWrapper
import Misc
import Lossfunctions

if __name__ == '__main__':
    if args.load_model is None:
        if bool(args.small_encoder):
            channels_in_encoding_blocks = [(1, 32), (32, 32), (32, 64), (64, 64), (64, 8)]
        else:
            channels_in_encoding_blocks = [(1, 64), (64, 64), (64, 128), (128, 128), (128, 8)]
        # Init model
        if bool(args.use_cat):
            model = Models.OccupancyNetwork(
                normalization_decoding='cbatchnorm' if bool(args.use_cbn) else 'batchnorm',
                channels_in_encoding_blocks=channels_in_encoding_blocks).cuda()
        else:
            model = Models.OccupancyNetworkNoCat(
                normalization_decoding='cbatchnorm' if bool(args.use_cbn) else 'batchnorm',
                channels_in_encoding_blocks=channels_in_encoding_blocks).cuda()
    else:
        model = torch.load(args.load_model).cuda()
    # Utilize data parallel
    if (args.use_data_parallel):
        model = torch.nn.DataParallel(model)
    # Print model
    print(model)
    # Print number of parameters included in the model
    print(Misc.get_number_of_network_parameters(model))
    # Init loss function
    if args.loss == 'cross_entropy':
        loss_function = torch.nn.BCELoss(reduction='mean')
    elif args.loss == 'focal':
        loss_function = Lossfunctions.FocalLoss(reduce='mean')
    else:
        loss_function = Lossfunctions.DiceLoss()
    # Construct folder name to save logs
    folder_name = 'cat_' + str(args.use_cat) + '_cbn_' + str(args.use_cbn) + '_encoder_' + str(args.small_encoder)
    # Init model wrapper
    model_wrapper = OccupancyNetworkWrapper(occupancy_network=model,
                                            occupancy_network_optimizer=torch.optim.Adam(
                                                model.parameters(), lr=args.lr),
                                            training_data=DataLoader(Datasets.WeaponDataset(
                                                target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                target_path_label='/visinf/home/vilab15/Projects/3D_baggage_segmentation/Data_len_1/',
                                                npoints=2 ** 14,
                                                side_len=8,
                                                length=2600),
                                                batch_size=args.batch_size, shuffle=True,
                                                collate_fn=Misc.many_to_one_collate_fn_sample,
                                                num_workers=args.batch_size, pin_memory=True),
                                            test_data=DataLoader(Datasets.WeaponDataset(
                                                target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                target_path_label='/visinf/home/vilab15/Projects/3D_baggage_segmentation/Data_len_1/',
                                                npoints=2 ** 18,
                                                side_len=8,
                                                length=306,  # 200,
                                                offset=2600,  # 2600,
                                                test=True,
                                                share_box=0.0),
                                                batch_size=1, shuffle=True,
                                                collate_fn=Misc.many_to_one_collate_fn_sample_down,
                                                num_workers=1, pin_memory=True,
                                            ),
                                            validation_data=DataLoader(Datasets.WeaponDataset(
                                                target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                target_path_label='/visinf/home/vilab15/Projects/3D_baggage_segmentation/Data_len_1/',
                                                npoints=2 ** 16,
                                                side_len=8,
                                                length=36,  # 200,
                                                offset=2906,  # 2600,
                                                test=True,
                                                share_box=0.0),
                                                batch_size=1, shuffle=True,
                                                collate_fn=Misc.many_to_one_collate_fn_sample_down,
                                                num_workers=1, pin_memory=True,
                                            ),
                                            loss_function=loss_function,
                                            device='cuda',
                                            data_folder=folder_name,
                                            save_data_path='Save_data_')

    if bool(args.train):
        model_wrapper.train(epochs=args.epochs)
    if bool(args.test):
        model_wrapper.test(side_len=1)
