import numpy as np
import math
from pathlib import Path
from PIL import Image
import cv2
import os
import sys
from sys import argv
from schnablelab.apps.natsort import natsorted
from schnablelab.apps.headers import Slurm_header
from schnablelab.apps.base import ActionDispatcher, OptionParser, glob, iglob

def main():
    actions = (
        ('three2two', 'convert 3d npy to 2d'),
        ('hyp2arr', 'convert hyperspectral images to a numpy array'),
    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())


def three2two(args):
    '''
    %prog three2two fn_in out_prefix 

    convert 3d npy to 2d
    '''
    p = OptionParser(three2two.__doc__)
    p.add_option('--crops',
        help='the coordinates for croping, follow left,upper,right,lower format. 1,80,320,479')
    p.add_option("--format", default='npy', choices=('npy', 'csv'),
        help="choose the output format")
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    fn_in, out_prefix, = args
    npy = np.load(fn_in)
    if opts.crops:
        left, up, right, down = opts.crops.split(',')
        npy = npy[int(up):int(down),int(left):int(right),:]
    h,w,d = npy.shape
    print(h, w, d)
    npy_2d = npy.reshape(h*w, d)
    if opts.format=='csv':
        out_fn = "%s.2d.csv"%out_prefix
        np.savetxt(out_fn, npy_2d, delimiter=",")
    else:
        out_fn = "%s.2d.npy"%out_prefix
        np.save(out_fn, npy_2d.astype(np.float64))
    print('Done!')

def hyp2arr(args):
    '''
    %prog hyp2arr hyp_dir out_fn

    convert hyperspectral images to numpy array
    '''
    p = OptionParser(hyp2arr.__doc__)
    opts, args = p.parse_args(args)
    if len(args)==0:
        sys.exit(not p.print_help())
    hyp_dir, out_fn, = args

    discard_imgs = ['0_0_0.png', '1_0_0.png']
    dir_path = Path(hyp_dir)
    if not dir_path.exists():
        sys.exit('%s does not exist!'%hyp_dir)
    imgs = list(dir_path.glob('*.png'))
    imgs = sorted(imgs, key=lambda x: int(x.name.split('_')[0]))
    num_imgs = len(imgs)
    print('%s images found.'%num_imgs)
    img_arrs = []
    for i in imgs:
        if not i.name in discard_imgs:
            print(i)
            arr = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
            print(i.name, arr.shape)
            img_arrs.append(arr)
    img_array = np.stack(img_arrs, axis=2)
    print(img_array.shape)
    np.save(out_fn, img_array)

if __name__=='__main__':
    main()
