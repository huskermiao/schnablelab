# -*- coding: UTF-8 -*-

"""
class and functions to deal with high throughput phenotyping data
"""
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from shutil import copyfile
from schnablelab.apps.Tools import GenDataFrameFromPath
from schnablelab.apps.base import ActionDispatcher, OptionParser, put2slurm

def main():
    actions = (
        ('ExtractRGBs', 'extract images from project folder'),
        ('Info', 'summary of image data under the project folder'),
        ('List', 'list specified image folders'),

    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

class ParseProject():
    def __init__(self, prj_dir_name, sm_idx=1, date_idx=2, time_idx=3):
        self.prj_dir_path = Path(prj_dir_name)
        try:
            df = pd.read_csv('%s.idx.csv'%self.prj_dir_path.name)
            df['fnpath'] = df['fnpath'].apply(lambda x: Path(x))
        except FileNotFoundError:
            print('project index file does not exist, creating one...')
            df = GenDataFrameFromPath(self.prj_dir_path, pattern='*')
            df = df[df['fnpath'].apply(lambda x: x.is_dir())]
            df['sm'] = df['fn'].apply(lambda x: x.split('_')[sm_idx])
            df['date'] = df['fn'].apply(lambda x: x.split('_')[date_idx])
            df['time'] = df['fn'].apply(lambda x: x.split('_')[time_idx])
            df = df.sort_values(['sm', 'date', 'time']).reset_index(drop=True)
            df.to_csv('%s.idx.csv'%self.prj_dir_path.name, index=False)
        finally:
            self.df = df
        self.sm_counts = self.df['sm'].value_counts().sort_index()
        self.date_counts = self.df['date'].value_counts().sort_index()
        self.SMs = self.df['sm'].unique()
        self.Dates = self.df['date'].unique()
    
    def Subsamples(self, samples):
        '''
        samples (list): list of samples
        '''
        for sm in samples:
            if not sm in self.SMs:
                sys.exit('%s not in the sample list'%sm)
        #print(samples)
        cond = self.df['sm'].isin(samples)
        return cond
    
    def Subdates(self, dates):
        '''
        dates (list): list of dates
        '''
        for date in dates:
            if not date in self.Dates:
                sys.exit('%s not in the date list'%date)
        #print(dates)
        cond = self.df['date'].isin(dates)
        return cond

    def RGB(self, folder_idx=None, samples=None, dates=None, angle=None, backup_angle=None):
        if folder_idx is not None:
            df = self.df.iloc[folder_idx,:]
        elif samples and not dates:
            df = self.df[self.Subsamples(samples)]
        elif not samples and dates:
            df = self.df[self.Subdates(dates)]
        elif samples and dates:
            df = self.df[self.Subsamples(samples) & self.Subdates(dates)]
        else:
            df = self.df.copy()
        #print(df[['sm', 'date', 'time']])
        pbar = tqdm(df.iterrows(), total=df.shape[0])
        for _,row in pbar:
            sm, d, hms = row['sm'], row['date'], row['time']
            if angle:
                img_fn = row['fnpath']/('Vis_SV_%s/0_0_0.png'%angle)
                if not img_fn.exists():
                    if backup_angle:
                        img_fn = row['fnpath']/('Vis_SV_%s/0_0_0.png'%backup_angle)
                    else:
                        print(f'{img_fn} does not exist in the project directory, omit!')
                        continue
            else:
                sys.exit('specif at least one vewing angle for RGB images')
            yield sm, d, hms, img_fn

def List(args):
    '''
    %prog List project_folder
    
    list specified image folders
    '''
    p = OptionParser(List.__doc__)
    p.add_option('--item_idx', default='1,2,3',
        help = 'the index of sample name, date, and time in each image directory name')
    p.add_option('--samples',
        help = 'specify samples (comma separated without space if speciy more than one samples)')
    p.add_option('--dates',
        help = 'specify dates (comma separated without space if speciy more than one dates)')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    project_folder, = args

    sm_idx, date_idx, time_idx = [int(i) for i in opts.item_idx.split(',')]
    prj = ParseProject(project_folder, sm_idx, date_idx, time_idx)
    if opts.samples and not opts.dates:
        samples = opts.samples.split(',')
        cond = prj.Subsamples(samples)
    elif not opts.samples and opts.dates:
        dates = opts.dates.split(',')
        cond = prj.Subdates(dates)
    elif not opts.samples and not opts.dates:
        print('Specify either samples or dates for showing!')
    else:
        print('provide either samples or dates for showing')
    print(prj.df[cond][['fn', 'sm', 'date', 'time']])

def Info(args):
    '''
    %prog Info project_folder

    Show summary of images under project_folder
    '''
    p = OptionParser(Info.__doc__)
    p.add_option('--item_idx', default='1,2,3',
        help = 'the index of sample name, date, and time in each image directory name')
    opts, args = p.parse_args(args)
    
    if len(args) == 0:
        sys.exit(not p.print_help())
    project_folder, = args

    sm_idx, date_idx, time_idx = [int(i) for i in opts.item_idx.split(',')]
    prj = ParseProject(project_folder, sm_idx, date_idx, time_idx)
    print('Summary of samples:')
    for i,j in prj.sm_counts.items():
        print(i, j)
    print('Summary of dates:')
    for i,j in prj.date_counts.items():
        print(i, j)
    print('Angles for RGB images:')
    for angle in prj.df.loc[0,'fnpath'].glob('Vis_*'):
        print(angle.name)

def ExtractRGBs(args):
    '''
    %prog ExtractRGBs project_folder

    extract RGB images from project folder
    '''
    p = OptionParser(ExtractRGBs.__doc__)
    p.add_option('--npy_idx',
        help = 'specify the numpy file including the indices for extraction')
    p.add_option('--item_idx', default='1,2,3',
        help = 'the index of sample name, date, and time in each image directory name')
    p.add_option('--out_dir', default='.',
        help = 'specify the output image directory')
    p.add_option('--samples',
        help = 'extract particular samples. multiple samples separated by comma without space')
    p.add_option('--dates',
        help = 'extract particular dates. multiple dates separated by comma without space.')
    p.add_option('--angle', default='108',
        help = 'which viewing angle are you going to extract?')
    p.add_option('--backup_angle',
        help = 'specify an alternative viewing angle for RGB images if the above angle does not exist.')
    p.add_option('--copy_only', default=False, action='store_true',
        help = 'only do copy without resizing and converting image format')
    p.add_option('--disable_slurm', default=False, action='store_true',
        help = 'do not convert commands to slurmm jobs')
    p.add_slurm_opts(job_prefix=ExtractRGBs.__name__)
    opts, args = p.parse_args(args)
    
    if len(args) == 0:
        sys.exit(not p.print_help())
    project_folder, = args

    out_dir = Path(opts.out_dir)
    if not out_dir.exists():
        print("The output directory '%s' does not exist, creating.."%out_dir)
        out_dir.mkdir()

    cmd = f'python -m schnablelab.ImageProcessing.HTP ExtractRGBs {project_folder} --out_dir {out_dir} --disable_slurm '
    if opts.npy_idx:
        npy_idx = np.load(opts.npy_idx)
        print(npy_idx)
        cmd += f'--npy_idx {opts.npy_idx} '
    if opts.samples:
        cmd += f'--samples {opts.samples} '
    if opts.dates:
        cmd += f'--dates {opts.dates} '
    if opts.angle:
        cmd += f'--angle {opts.angle} '
    if opts.copy_only:
        cmd += f'--copy_only '
    print(cmd)
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        return 

    opts.samples = opts.samples.split(',') if opts.samples else opts.samples
    opts.dates = opts.dates.split(',') if opts.dates else opts.dates

    sm_idx, date_idx, time_idx = [int(i) for i in opts.item_idx.split(',')]
    prj = ParseProject(project_folder, sm_idx, date_idx, time_idx)
   

    for sm, d, hms, path_img_fn in prj.RGB(folder_idx=npy_idx, samples=opts.samples, dates=opts.dates, angle=opts.angle, backup_angle=opts.backup_angle):
        angle_dir_name = path_img_fn.parts[-2]
        dest_fn = '%s_%s_%s_%s.jpg'%(sm, d, hms, angle_dir_name)
        dest = out_dir/dest_fn
        if dest.exists():
            print(f'{dest} already exists, omit!')
        else:
            if opts.copy_only:
                copyfile(path_img_fn, dest)
            else:
                Image.open(path_img_fn).convert('RGB').resize((1227, 1028)).save(dest)
            
if __name__ == '__main__':
    main()
