# 8/13/18
# chenyong
# call plant height

"""
call plant height from predicted images
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import os.path as op
import scipy.misc as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image
from math import hypot
from schnablelab.apps.natsort import natsorted
from schnablelab.apps.headers import Slurm_header
from sklearn.linear_model import LinearRegression
from schnablelab.apps.base import ActionDispatcher, OptionParser, glob
import datetime
from dateutil import parser
from pathlib import Path

def main():
    actions = (
        ('Polish', 'Polish the predicted images (hyper)'),
        ('PolishBatch', 'generate all slurm jobs of polish (hyper)'),
        ('CallHeight', 'call height from polished image (hyper)'),
        ('CallHeightBatch', 'generate all slurm jobs of plant height calling (hyper)'),
        ('CallHeightRGB', 'call height from RGB image'),
        ('CallHeightRGBBatch', 'generate all slurm jobs of plant height calling (RGB)'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def CallHeightRGB(args):
    """
    %prog image_in_dir
    using thresholding method to calculate the plant height
    """
    p = OptionParser(CallHeightRGB.__doc__)
    p.add_option("--threshold", default = '1.12',
        help='speficy the threshold cutoff') 
    p.add_option("--zoom_date",
        help='specify which date zoome level changed, yyyy-mm-dd') 
    p.add_option("--summarize_fn", default= 'Heights.csv',
        help='specify the file recording height for each sample') 
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    imgInDir, = args
    
    inPath = Path(imgInDir)
    imgs = list(inPath.glob('*png'))
    print('Total %s images'%len(imgs))
    df = pd.DataFrame(dict(zip(['fullPath'],[imgs])))
    df['fn'] = df['fullPath'].apply(lambda x: x.name)
    df['sm'] = df['fn'].apply(lambda x: x.split('_')[1])
    df['dt'] = df['fn'].apply(lambda x: x.split('_')[2].split('.')[0])
    df['dt'] = pd.to_datetime(df['dt'])
    #df['sv'] = df['fn'].apply(lambda x: x.split('_')[-1].split('.')[0])
    #df_sort = df.sort_values(['sm','dt','sv']).reset_index(drop=True)
    df_sort = df.sort_values(['sm','dt']).reset_index(drop=True)
    #print(df_sort)

    threshold = float(opts.threshold)
    print('threshold by green index value %s'%threshold)
    zoom_date = parser.parse(opts.zoom_date) 
    print('zoom change date: %s'%zoom_date)
    zoom_border_dict = {'zoom1': (60,1750,500,2250), 'zoom2': (180,1700,850,1770)}
    zoom_ratio_dict = {'zoom1': 149/1925, 'zoom2': 149/965} 

    f0 = open(opts.summarize_fn, 'w')
    f0.write('file_name\tzoome_level\theight(pixel)\theight(cm)\n')
    for idx, row in df_sort.iterrows():
        print(row['fn'])
        # read image and convert bgr to rgb
        img = cv2.imread(str(row['fullPath']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print('read image and convert bgr to rgb')
        # convert 3 2D to 1 2D with green index
        img_float = img.astype(np.float)
        img_green = (2*img_float[:,:,1])/(img_float[:,:,0]+img_float[:,:,2]) # 2*green/(red+blue)
        thresh1 = np.where(img_green>threshold, img_green, 0)
        print('convert 3 2D to 1 2D with green index')
        # remove the chamber border
        mytime = row['dt']
        zoom_level = 'zoom1' if mytime < zoom_date else 'zoom2'
        upper,bottom,left,right = zoom_border_dict[zoom_level]
        thresh1[0:upper]=0
        thresh1[bottom:]=0
        thresh1[:,0:left]=0
        thresh1[:,right:]=0
        print('remove the chamber border')
        # rescale to 255
        try:
            thresh1 = (thresh1/float(thresh1.max()))*255
        except:
            f0.write('%s\t%s\tNaN\tNaN\n'%(row['fn'], zoom_level))
            continue
        # blur the image
        blur = cv2.GaussianBlur(thresh1, (7,7), 0)
        # 2nd threshold 
        blur_int = blur.astype(np.uint8)
        ret, thresh2 = cv2.threshold(blur_int, 1, 255, cv2.THRESH_BINARY)
        # call contours
        '''there are three arguments in cv2.findContours() function, first one is source image, 
        second is contour retrieval mode, third is contour approximation method. 
        And it outputs the contours and hierarchy. contours is a Python list of all the contours in the image. 
        Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.'''
        __,contours,__ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        # call height
        min_y, max_y = [],[]
        for i in contours:
            min_y.append(np.min(i[:,:,1]))
            max_y.append(np.max(i[:,:,1]))

        if min_y and max_y:        
            y_lowest, y_highest = min(min_y), max(max_y)
            height_pixels = y_highest-y_lowest
            height_cm = height_pixels*zoom_ratio_dict[zoom_level]
            f0.write('%s\t%s\t%s\t%s\n'%(row['fn'], zoom_level, height_pixels, height_cm))
            # draw height and save results
            cv2.line(img, (500, y_lowest), (2000, y_lowest), (255,0,0), 7)    
            new_fn = row['fn'].replace('.png', '.height.png')
            new_fn_path = inPath/new_fn
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # cv2 assume your color is bgr
            cv2.imwrite(str(new_fn_path), img)
            #print('%s saved.'%new_fn)
        else:
            f0.write('%s\t%s\tNaN\tNaN\n'%(row['fn'], zoom_level))
    f0.close()
    print('Done! check %s'%opts.summarize_fn)

def CallPart(rgb_arr, part='stem'):
    crp_shape2d = rgb_arr.shape[0:2]
    if part =='stem':
        r, g, b = 251, 129, 14
    elif part == 'panicle':
        r, g, b = 126, 94, 169
    elif part == 'leaf':
        r, g, b = 0, 147, 0
    else:
        sys.exit('only support stem, panicle, and leaf')
    p1 = np.full(crp_shape2d,r)
    p2 = np.full(crp_shape2d,g)
    p3 = np.full(crp_shape2d,b)
    p123 = np.stack([p1, p2, p3], axis=2)
    pRGB = np.where(rgb_arr==p123, rgb_arr, 255)
    return pRGB

def FilterPixels(arr3d, d=0):
    rgb_img = Image.fromarray(arr3d)
    gray_img = rgb_img.convert(mode='L')
    gray_blur_arr = cv2.GaussianBlur(np.array(gray_img), (3,3), 0)
    cutoff = pd.Series(gray_blur_arr.flatten()).value_counts().index.sort_values()[d]
    arr2d = np.where(gray_blur_arr<=cutoff, 0, 255) 
    return arr2d

def gray2rgb(arr2d, part="stem"):
    cond_k = arr2d==0
    if part =='stem':
        r, g, b = 251, 129, 14
    elif part == 'panicle':
        r, g, b = 126, 94, 169
    elif part == 'leaf':
        r, g, b = 0, 147, 0
    else:
        sys.exit('only support stem, panicle, and leaf')
    pr = np.where(cond_k, r, 255)
    pg = np.where(cond_k, g, 255)
    pb = np.where(cond_k, b, 255)
    pRGB = np.stack([pr, pg, pb], axis=2)
    return pRGB

def Polish(args):
    """
    %prog image_in image_out_prefix
    Using opencv blur function to filter noise pixles for each plant component
    """
    p = OptionParser(Polish.__doc__)
    p.add_option("--crop",
        help="if you want to crop image, please specify the crop size following coordinates of upper left conner and right bottom conner.")
    p.add_option("--blur_degree", default='4',
        help="specify the degree value in GaussinBlur function. [default: %default]")
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    imgIn, imgOut = args

    img = Image.open(imgIn)
    if opts.crop:
        crp_tuple = tuple([int(i) for i in opts.crop.split()]) # crop: left, upper, right, and lower pixel coordinate
        if len(crp_tuple) != 4:
            sys.exit("please specify 'left upper right bottom'")
        else:
            img = np.array(img.crop(crp_tuple))
    else:
        img = np.array(img)
    stemRGBraw = CallPart(img, 'stem')
    stem = FilterPixels(stemRGBraw)
    stemRGB = gray2rgb(stem, 'stem')
    panicleRGBraw = CallPart(img, 'panicle')
    panicle = FilterPixels(panicleRGBraw, d=int(opts.blur_degree))
    panicleRGB = gray2rgb(panicle, 'panicle')
    leafRGBraw = CallPart(img, 'leaf')
    leaf = FilterPixels(leafRGBraw, d=int(opts.blur_degree))
    leafRGB = gray2rgb(leaf, 'leaf')
    spRGB = np.where(stemRGB==255, panicleRGB, stemRGB)
    splRGB = np.where(spRGB==255, leafRGB, spRGB)
    sm.imsave('%s.polish.png'%imgOut, splRGB)

def PolishBatch(args):
    """
    %prog imagePattern("CM*.png")
    generate polish jobs for all image files
    """
    p = OptionParser(PolishBatch.__doc__)
    p.set_slurm_opts(array=False)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    pattern, = args
    all_pngs = glob(pattern)
    for i in all_pngs:
        out_prefix = i.split('/')[-1].split('.png')[0]
        jobname = out_prefix + '.polish'
        cmd = 'python -m schnablelab.CNN.CallHeight Polish %s %s\n'%(i, out_prefix)
        header = Slurm_header%(opts.time, opts.memory, jobname, jobname, jobname)
        header += "ml anaconda\nsource activate %s\n"%opts.env
        header += cmd
        jobfile = open('%s.polish.slurm'%out_prefix, 'w')
        jobfile.write(header)
        jobfile.close()
        print('%s.slurm polish job file generated!'%jobname)

def CallHeight(args):
    """
    %prog image_in output_prefix
    call height from polished image
    """
    p = OptionParser(CallHeight.__doc__)
    p.add_option("--crop",
        help="if you want to crop image, please specify the crop size following coordinates of upper left conner and right bottom conner.")
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    imageIn, outPrefix = args

    img = Image.open(imageIn)
    if opts.crop:
        crp_tuple = tuple([int(i) for i in opts.crop.split()]) # crop: left, upper, right, and lower pixel coordinate
        if len(crp_tuple) != 4:
            sys.exit("please specify 'left upper right bottom'")
        else:
            img = np.array(img.crop(crp_tuple))
    else:
        img = np.array(img)

    # get stem and panicle pixels
    sRGB = CallPart(img, 'stem')
    sRGB_img = Image.fromarray(sRGB)
    sgray = np.array(sRGB_img.convert(mode='L'))
    pRGB = CallPart(img, 'panicle')
    pRGB_img = Image.fromarray(pRGB)
    pgray = np.array(pRGB_img.convert(mode='L'))
    spgray = np.where(sgray==255, pgray, sgray)
    xlim, ylim = spgray.shape 
    # fit model
    X, Y = np.where(spgray< 255)
    X = X*-1+xlim
    model = LinearRegression()
    model.fit(X.reshape(-1,1), Y)
    # regression line
    
    #a = X.max()
    a = 131
    b = np.abs(model.predict(0)-model.predict(a))
    c = hypot(a, b)
    f1 = open('%s.Height.csv'%outPrefix, 'w')
    f1.write('%s'%c)
    f1.close()
    # plot
    plt.switch_backend('agg')
    rcParams['figure.figsize'] = xlim*0.015, ylim*0.015
    fig, ax = plt.subplots()
    ax.scatter(X, Y, s=0.1, color='k', alpha=0.7)
    ax.plot([0, a], [model.predict(0), model.predict(a)], c='r', linewidth=1)
    ax.text(100, 50, "%.2f"%c, fontsize=12)
    ax.set_xlim([0,xlim])
    ax.set_ylim([0,ylim])
    plt.tight_layout()
    plt.savefig('%s.Height.png'%outPrefix)

def CallHeightBatch(args):
    """
    %prog imagePattern("CM*.polish.png")
    generate height call jobs for all polished image files
    """
    p = OptionParser(CallHeightBatch.__doc__)
    p.set_slurm_opts(array=False)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    pattern, = args
    all_pngs = glob(pattern)
    for i in all_pngs:
        out_prefix = i.split('/')[-1].split('.polish.png')[0]
        jobname = out_prefix + '.Height'
        cmd = 'python -m schnablelab.CNN.CallHeight CallHeight %s %s\n'%(i, out_prefix)
        header = Slurm_header%(opts.time, opts.memory, jobname, jobname, jobname)
        header += "ml anaconda\nsource activate %s\n"%opts.env
        header += cmd
        jobfile = open('%s.CallHeight.slurm'%out_prefix, 'w')
        jobfile.write(header)
        jobfile.close()
        print('%s.CallHeight.slurm call height job file generated!'%jobname)

if __name__ == "__main__":
    main()
















