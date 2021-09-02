# 7/16/18
# chenyong 
# prediction

"""
make predictions using trained model
"""
import os
import math
import os.path as op
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from schnablelab.apps.base import cutlist, ActionDispatcher, OptionParser, glob
from schnablelab.apps.headers import Slurm_header, Slurm_gpu_header
from schnablelab.apps.natsort import natsorted
from glob import glob
from pathlib import Path
from subprocess import run

def main():
    actions = (
        ('Plot', 'plot training model history'),
        ('Predict', 'using trained neural network to make prediction'),
        ('Imgs2Arrs', 'convert hyperspectral images under a dir to a numpy array object'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def Imgs2Arrs(args):
    '''
    %prog hyp_dir(filepath of hyperspectral image data) 
    Returns: numpy array object with shape [x*y, z].
        x,y dims correspond to pixel coordinates for each image
        z dim corresponds to hyperspectral image wavelength.
    '''
    import cv2
    
    p = OptionParser(Imgs2Arrs.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    mydir, = args
    imgs = [i for i in os.listdir(mydir) if i.endswith('png')]
    sorted_imgs = sorted(imgs, key=lambda x: int(x.split('_')[0]))
    all_arrs = []
    for i in sorted_imgs[2:]:
        print(i)
        #img = cv2.imread('%s/%s'%(mydir, i), cv2.IMREAD_GRAYSCALE)
        img = np.array(Image.open('%s/%s'%(mydir, i)).convert('L'))
        print(img.shape)
        all_arrs.append(img)
    arrs = np.stack(all_arrs, axis=2)
    np.save('%s.npy'%mydir, arrs)


def Predict(args):
    """
    %prog model_name npy_pattern('CM*.npy')
    using your trained model to make predictions on selected npy (2d or 3d) files.
    The pred_data is a numpy array object which has the same number of columns as the training data.
    """
    from keras.models import load_model
    import scipy.misc as sm
    p = OptionParser(Predict.__doc__)
    p.add_option('--range', default='all',
        help = "specify the range of the testing images, hcc job range style")
    p.add_option('--opf', default='infer',
        help = "specify the prefix of the output file names")
    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())
    model, npy_pattern = args
    opf = model.split('/')[-1].split('.')[0] if opts.opf == 'infer' else opts.opf

    npys = glob(npy_pattern)
    if opts.range != 'all':
        start = int(opts.range.split('-')[0])
        end = int(opts.range.split('-')[1])
        npys = npys[start:end]
    print('%s npys will be predicted this time.'%len(npys))

    my_model = load_model(model)
    for npy in npys:
        print(npy)
        test_npy = np.load(npy)
        npy_shape = test_npy.shape
        np_dim = len(npy_shape)
        test_npy_2d = test_npy.reshape(npy_shape[0]*npy_shape[1], npy_shape[2]) if np_dim==3 else test_npy
        print('testing data shape:', test_npy_2d.shape)
        pre_prob = my_model.predict(test_npy_2d)
        predictions = pre_prob.argmax(axis=1) # this is a numpy array

        if np_dim == 3:
            predictions = predictions.reshape(npy_shape[0], npy_shape[1])
            df = pd.DataFrame(predictions)
            df1 = df.replace(0, 255).replace(1, 127).replace(2, 253).replace(3, 190)#0: background; 1: leaf; 2: stem; 3: panicle
            df2 = df.replace(0, 255).replace(1, 201).replace(2, 192).replace(3, 174)
            df3 = df.replace(0, 255).replace(1, 127).replace(2, 134).replace(3, 212) 
            arr = np.stack([df1.values, df2.values, df3.values], axis=2)
            opt = npy.split('/')[-1].split('.npy')[0]+'.prd'
            sm.imsave('%s.%s.png'%(opf,opt), arr)
        elif np_dim == 2:
            opt = npy.split('/')[-1].split('.npy')[0]+'.prd'
            np.savetxt('%s.%s.csv'%(opf, opt), predictions)
        else:
            sys.exit('either 2 or 3 dim numpy array!')
        print('Done!')

def Plot(args): 
    """
    %prog dir
    plot training process
    You can load the dict back using pickle.load(open('*.p', 'rb'))
    """

    p = OptionParser(Plot.__doc__)
    p.add_option("--pattern", default="History_*.p",
        help="specify the pattern of your pickle object file, remember to add quotes [default: %default]")
    p.set_slurm_opts()
    opts, args = p.parse_args(args)

    if len(args) == 0:
        sys.exit(not p.print_help())
    mydir, = args
    pickles = glob('%s/%s'%(mydir, opts.pattern)) 
    print('total %s pickle objects.'%len(pickles))
    #print(pickles)
    for p in pickles:
        fs, es = opts.pattern.split('*')
        fn = p.split(fs)[-1].split(es)[0]
        myp = pickle.load(open(p, 'rb'))
        
        mpl.rcParams['figure.figsize']=[7.5, 3.25]
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        # summarize history for accuracy
        ax1 = axes[0]
        ax1.plot(myp['acc'])
        ax1.plot(myp['val_acc'])
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.set_ylim(0,1.01)
        ax1.legend(['train', 'validation'], loc='lower right')
        max_acc = max(myp['val_acc'])
	# summarize history for loss
        ax2 = axes[1]
        ax2.plot(myp['loss'])
        ax2.plot(myp['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper right')
        plt.tight_layout()
        plt.savefig('%s_%s.png'%(max_acc,fn))    
        plt.clf()

if __name__ == "__main__":
    main()
