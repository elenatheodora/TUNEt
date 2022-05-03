# imports
import time
import argparse
import os
import pandas as pd
import pickle
from glob import glob
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.waveunet import Waveunet
from utils.data_helper import Ambisonic
import utils.model_helper as model_utils
np.seterr(divide='ignore', invalid='ignore')

def main(args):
    num_channels = [args.features * i for i in range(1, args.levels + 1)] if args.feature_growth == "add" else \
        [args.features * 2 ** i for i in range(0, args.levels)]

    target_outputs = int(args.output_size * args.sr)

    model = Waveunet(args.num_in_chan, num_channels, args.num_out_chan, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    trans = transforms.Compose([transforms.ToTensor()])
    train_dir = args.data_dir + 'train.hdf'
    trainset = Ambisonic(path=train_dir, transform=trans, args=args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    val_dir = args.data_dir + 'val.hdf'
    valset = Ambisonic(path=val_dir, transform=trans, args=args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # if args.dry_run:
    #     for data, labels in train_loader:
    #         if torch.cuda.is_available():
    #             data, labels = data.cuda(), labels.cuda()
    #         print(data.shape)
    #         outputs = model(data)
    #         print(f'Output shape : {outputs.shape}')
    #         print(f'labels shape : {labels.shape}')
    #         loss = criterion(outputs, labels)
    #         print(loss)
    #         break
    #
    #     return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(args.log_dir)

    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    avg_epoch_time = 0
    train_losses = []
    val_losses = []

    print('TRAINING START')
    stop_ctr = args.patience
    if args.dry_run:
        stop_ctr = 4

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

        #loading loss pickles
        with open(args.log_dir+'train_loss.pkl', 'rb') as fp:
            train_losses = pickle.load(fp)

        with open(args.log_dir+'val_loss.pkl', 'rb') as fp:
            val_losses = pickle.load(fp)

    while state["worse_epochs"] < stop_ctr:
        start_time = time.time()
        model.train()
        train_loss = 0.0
        # st1
        for data, labels in tqdm(train_loader):
            # time-st1
            state["step"] += 1

            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #st2
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #st2

            # print statistics
            train_loss += loss.item()

            writer.add_scalar("train_loss", loss.item()/args.batch_size, state["step"])
            # time-st1

        train_losses.append(train_loss / len(train_loader))

        #validation loss
        valid_loss = 0.0
        model.eval()

        for data, labels in val_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = model(data)
            loss = criterion(target, labels)
            valid_loss += loss.item()

        writer.add_scalar("val_loss", valid_loss / len(val_loader), state["step"])

        val_losses.append(valid_loss / len(val_loader))
        print(f'Epoch {state["epochs"]}; Train Loss: {train_loss / len(train_loader)}; '
                  f'Val Loss: {valid_loss / len(val_loader)}; time : {(time.time() - start_time)}sec')

        avg_epoch_time += (time.time() - start_time)

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if valid_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = valid_loss
            state["best_checkpoint"] = checkpoint_path
            # CHECKPOINT
            print("Saving model...")
            model_utils.save_model(model, optimizer, state, checkpoint_path)

        state["epochs"] += 1

        #saving loss
        with open(args.log_dir+'val_loss.pkl', 'wb') as fp:
            pickle.dump(val_losses, fp)

        with open(args.log_dir+'train_loss.pkl', 'wb') as fp:
            pickle.dump(train_losses, fp)

    print(f"avg. epoch time is {avg_epoch_time / state['epochs']}")
    print('Finished Training')
    writer.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()

    parser.add_argument('--dry_run', action='store_true',
                        help='dry_run will run for one batch (default: False)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader worker threads (default: 8)')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Folder to write logs into')
    parser.add_argument('--data_dir', type=str, default="data/",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Folder to write checkpoints into')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--num_in_chan', type=int, default=32,
                        help="Number of input audio channels")
    parser.add_argument('--num_out_chan', type=int, default=4,
                        help="Number of output audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=1.0,
                        help="Output duration (in sec)")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()

    main(args)