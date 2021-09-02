# -*- coding: UTF-8 -*-

"""
Transfer learning for feature extracting or finetuning. 
"""
from schnablelab.apps.base import OptionParser, ActionDispatcher, put2slurm
from schnablelab.CNN.base import EarlyStopping, image_transforms, initialize_model, train_model_regression 
from schnablelab.CNN.DatasetReady import LeafcountingDataset
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
import pandas as pd

def main():
    actions = (
        ('regression', 'using pretrained model to solve regression problems'),
        ('prediction', 'make predictions using the trained model'),
        ('classification', 'using pretrained model to solve classification problems'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def classification(args):
    pass

def regression(args):
    """
    %prog regression train_csv, train_dir, model_name_prefix
    Args:
        train_csv: csv file (comma separated without header) containing all training image filenames
        train_dir: directory where training images reside
        model_name_prefix: the prefix of the output model name 
    """
    p = OptionParser(regression.__doc__)
    p.add_option('--valid_csv',
                    help='csv file for validation if available')
    p.add_option('--valid_dir',
                    help='directory where validation images reside')
    p.add_option('--inputsize', default=224, type='int',
                    help='the input size of image. At least 224 if using pretrained models')
    p.add_option('--batchsize', default=60, type='int', 
                    help='batch size')
    p.add_option('--epoch', default=500, type='int', 
                    help='number of total epochs')
    p.add_option('--patience', default=50, type='int', 
                    help='patience in early stopping')
    p.add_option('--base_mn', default='resnet18',
                    help='base model architectures: vgg16, googlenet, resnet18, resnet152...')
    p.add_option('--tl_type', default='finetuning', choices=('feature_extractor', 'finetuning'),
                    help='transfer learning type. finetuning: initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual. feature_extractor: freeze the weights for all of the network except that of the final fully connected layer. ')
    p.add_option('--pretrained_mn',
                    help='specify your own pretrained model as feature extractor')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly in the console without generating slurm job. Do not do this in HCC login node')
    p.add_slurm_opts(job_prefix=regression.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 3:
        sys.exit(not p.print_help())
    train_csv, train_dir, model_name_prefix = args
    # genearte slurm file
    if not opts.disable_slurm:
        cmd = "python -m schnablelab.CNN.TransLearning regression "\
            f"{train_csv} {train_dir} {model_name_prefix} "\
            f"--inputsize {opts.inputsize} --base_mn {opts.base_mn} --disable_slurm "
        if opts.pretrained_mn:
            cmd += f"--pretrained_mn {opts.pretrained_mn} "
        if opts.valid_csv and opts.valid_dir:
            cmd += f"--valid_csv {opts.valid_csv} --valid_dir {opts.valid_dir} "
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    logfile = model_name_prefix + '.log'
    histfile = model_name_prefix + '.hist.csv'
    
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(logfile, mode='w')
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s:%(name)s:%(funcName)s:%(levelname)s:%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    # can also create another handler for streaming. e.g. c_handler = logging.StreamHandler()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug('device: %s'%device) # creat a LogRecord and send this info to all the handlers in the logger
    logger.debug('pytorch version: %s'%torch.__version__)
    logger.debug('cuda version: %s'%torch.version.cuda)

    # prepare training and validation data
    train_dataset = LeafcountingDataset(train_csv, train_dir, image_transforms(input_size=opts.inputsize)['train'])
    train_loader = DataLoader(train_dataset, batch_size=opts.batchsize)
    dataloaders_dict = {'train': train_loader}

    if opts.valid_csv and opts.valid_dir:
        valid_dataset = LeafcountingDataset(opts.valid_csv, opts.valid_dir, image_transforms(input_size=opts.inputsize)['valid'])
        valid_loader = DataLoader(valid_dataset, batch_size=opts.batchsize)
        dataloaders_dict['valid'] = valid_loader 

    # initialize the pre-trained model
    feature_extract = True if opts.tl_type=='feature_extractor' else False
    logger.debug('feature extract: %s'%feature_extract)
    
    if opts.pretrained_mn:
        model, input_size = initialize_model(model_name=opts.base_mn, 
                                         feature_extract=True, # set param.requires_grad=True for all layers except the fully connected layer
                                         use_pretrained=False,
                                         inputsize=opts.inputsize)
        model.load_state_dict(torch.load(opts.pretrained_mn, map_location=device))
    else:
        model, input_size = initialize_model(model_name=opts.base_mn, 
                                         feature_extract=feature_extract, 
                                         inputsize=opts.inputsize)
    logger.debug(model)

    params_to_update = [param for param in model.parameters() if param.requires_grad] # trainable parameters
    sgd_optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9) # optimizer
    criterion = nn.MSELoss() # loss
    # train and validation
    inception = True if opts.base_mn=='inception' else False
    since = time.time()
    model_ft, train_hist, valid_hist = train_model_regression(model, dataloaders_dict, 
                                                            criterion, sgd_optimizer,
                                                            model_name_prefix, 
                                                            patience=opts.patience, 
                                                            num_epochs=opts.epoch, 
                                                            is_inception=inception)
    time_elapsed = time.time() - since
    logger.debug('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # save training and validation loss.
    logger.debug('saving loss history...')
    if opts.valid_csv and opts.valid_dir:
        df = pd.DataFrame(dict(zip(['training', 'validation'], [train_hist, valid_hist])))
    else: 
        df = pd.DataFrame(dict(zip(['training'], [train_hist])))
    df.to_csv(histfile, index=False)
    
    # plot training and validation loss
    logger.debug('plot loss history...')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    plt.style.use('bmh')
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = df.plot(ax=ax)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plt.tight_layout()
    plt.savefig('%s.loss.png'%model_name_prefix, dpi=200)

def prediction(args):
    """
    %prog prediction saved_model test_csv, test_dir, output
    Args:
        saved_model: saved model with either a .pt or .pth file extension
        test_csv: csv file (comma separated with header) containing all testing image filenames
        test_dir: directory where testing images are located
        output: csv file saving prediction results
    """
    p = OptionParser(prediction.__doc__)
    p.add_option('--inputsize', default=224, type='int',
                    help='the input size of image. At least 224 if using pretrained models')
    p.add_option('--batchsize', default=36, type='int', 
                    help='batch size')
    p.add_option('--base_mn', default='resnet18',
                    help='base model architectures: vgg16, googlenet, resnet18, resnet152...')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=prediction.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 4:
        sys.exit(not p.print_help())
    saved_model, test_csv, test_dir, output = args

    # genearte slurm file
    if not opts.disable_slurm:
        cmd = "python -m schnablelab.CNN.TransLearning prediction "\
            f"{saved_model} {test_csv} {test_dir} {output} "\
            f"--batchsize {opts.batchsize} --disable_slurm "
        if opts.base_mn:
            cmd += f"--base_mn {opts.base_mn} "
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('devicd: %s'%device)

    if opts.base_mn:
        model, input_size = initialize_model(model_name=opts.base_mn, 
                                            feature_extract=True, 
                                            use_pretrained=False, 
                                            inputsize=opts.inputsize)
        # turn all gradients off
        for param in model.parameters():
            param.requires_grad = False
    else:
        sys.exit('not implemented yet...')

    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()

    test_dataset = LeafcountingDataset(test_csv, test_dir, image_transforms(input_size=opts.inputsize)['valid'])
    test_loader = DataLoader(test_dataset, batch_size=opts.batchsize)

    ground_truths, predicts, filenames = [],[],[]
    for idx, (inputs, labels, fns) in enumerate(test_loader, 1): # fns is a tuple
        print('idx %s'%idx)
        inputs = inputs.to(device)
        print('type of inputs: %s'%(type(inputs)))
        outputs = model(inputs)
        ground_truths.append(labels.squeeze().numpy())
        filenames.append(np.array(fns))
        if torch.cuda.is_available():
            predicts.append(outputs.squeeze().to('cpu').numpy())
        else:
            predicts.append(outputs.squeeze().numpy())
    ground_truths = np.concatenate(ground_truths)
    predicts = np.concatenate(predicts)
    filenames = np.concatenate(filenames)
    df = pd.DataFrame(dict(zip(['fn', 'groundtruth', 'prediction'], [filenames, ground_truths, predicts])))
    df.to_csv(output, index=False)

if __name__ == "__main__":
    main()
