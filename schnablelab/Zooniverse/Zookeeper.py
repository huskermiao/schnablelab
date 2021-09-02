'''
schnablelab CLI tool
Calls Zookeeper class
'''
import os
import sys
import math
import pandas as pd
from pathlib import Path
from shutil import copyfile
from schnablelab.apps.Tools import GenDataFrameFromPath
from schnablelab.apps.base import ActionDispatcher, OptionParser, cutlist, put2slurm

def main():
    actions = (
        ('toy', 'random pick up some images for testing purporse'),
        ('divide', 'divide a large number of images to sevearl subsets'),
        ('upload', 'load images to zooniverse'),
        ('BatchUpload', 'upload multiple dirs on HCC'),
        ('export', 'Get annotation and other exports'),
        ('manifest', 'Generate a manifest for zooniverse subject set upload')
    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def BatchUpload(args):
    '''
    %prog BatchUpload dir1 dir2... project_id subject_id

    upload multiple dataset 
    '''
    p = OptionParser(BatchUpload.__doc__)
    p.add_option('--disable_slurm', default=False, action="store_true",
                help='do not convert commands to slurm job')
    p.add_slurm_opts(job_prefix=BatchUpload.__name__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    *img_dirs, p_id, s_id = args
    cmds = []
    for img_dir in img_dirs:
        cmd = f'python -m schnablelab.Zooniverse.Zookeeper upload {img_dir} {p_id} {img_dir} --subject_id {s_id}'
        cmds.append(cmd)
    cmd_sh = '%s.cmds%s.sh'%(opts.job_prefix, len(cmds))
    pd.DataFrame(cmds).to_csv(cmd_sh, index=False, header=None)
    print(f'check {cmd_sh} for all the commands!')
    if not opts.disable_slurm:
        put2slurm_dict = vars(opts)
        put2slurm(cmds, put2slurm_dict)

def toy(args):
    '''
    %prog toy input_dir output_dir n_toy_imgs

    randomly pick up n toy images from input_dir and put to out_put_dir
    '''
    p = OptionParser(toy.__doc__)
    _, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    input_dir, output_dir, n, = args
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    df = GenDataFrameFromPath(Path(input_dir), pattern='*.png')
    for i in df['fnpath'].sample(int(n)):
        fn = i.name
        print(fn)
        copyfile(i, output_dir/fn)
    print('Done! Check images in %s.'%output_dir)

def divide(args):
    '''
    %prog divide input_dir output_dir_prefix
    '''
    p = OptionParser(divide.__doc__)
    p.add_option('--pattern', default='*.jpg',
                 help='file name pattern')
    p.add_option('--nimgs_per_folder', type='int', default=700,
                 help='~ number of images (<1000) in each smaller folder')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    input_dir, out_prefix, = args

    df = GenDataFrameFromPath(Path(input_dir), pattern=opts.pattern)
    n_folders = math.ceil(df.shape[0]/opts.nimgs_per_folder)
    print('%s will be divided to %s datasets'%(df.shape[0], n_folders))
    n = 0
    for _, grp in cutlist(df['fnpath'].values, n_folders):
        n += 1
        output_folder = Path('%s_%s'%(out_prefix,n))
        print(output_folder, grp.shape[0])
        if not output_folder.exists():
            output_folder.mkdir()
        for i in grp:
            copyfile(i, output_folder/i.name)

def upload(args):
    '''
    %prog upload imgdir projid dataset_name

    - imgdir: Path to directory of the images to be uploaded
    - projid: Zooniverse project id (4 - 5 digit number)
    - dataset_name: specify a name for this dataset. the name
		will be ignored if subject_id is provided. 

    DESC:
        Uploads images from the image directory to zooniverse
        project. If there is no manifest will generate one.
    '''

    from schnablelab.Zooniverse.Zootils import upload as load

    p = OptionParser(upload.__doc__)
    p.add_option('-s', '--subject_id', default=False,
                 help='Designate a subject set id.')
    p.add_option('-q', '--quiet', action='store_true', default=False,
                 help='Silences output when uploading images to zooniverse.')
    p.add_option('-x', '--extension', default=False,
                 help='Specify the extension of the image files to be uploaded.')

    opts, args = p.parse_args(args)

    if len(args) != 3:
        p.print_help()
        exit(False)

    imgdir, projid, dataset_name, = args
    user_info = dict()
    try:
        un = os.environ['ZOO_UN']
        pw = os.environ['ZOO_PW']
    except KeyError:
        print("ZOO_UN and ZOO_PW variables were not defined!")
    else:
        print('ZOO_UN and ZOO_PW variables were detected!')
        user_info['un'] = un
        user_info['pw'] = pw

    load(imgdir, projid, dataset_name, opts, **user_info)

    return True


def export(args):
    '''
    %prog export proj_id outfile

    - proj_id: The project id of the zooniverse project

    DESC: Fetches an export from the specified zooniverse project id.
    '''

    from schnablelab.Zooniverse.Zootils import export as exp

    p = OptionParser(export.__doc__)
    p.add_option('-t', '--type', default='classifications',
                 help='Specify the type of export')

    opts, args = p.parse_args(args)

    if len(args) != 2:
        exit(not p.print_help())

    projid, outfile = args

    exp(projid, outfile, opts)

    return True


def manifest(args):
    '''
    %prog manifest image_dir

    - img_dir: The image directory in which to generate the manifest.

    DESC: Generates a manifest inside the specified image directory.
    '''
    from schnablelab.Zooniverse.Zootils import manifest as mani

    p = OptionParser(manifest.__doc__)
    opts, args = p.parse_args(args)

    if len(args) != 1:
        exit(not p.print_help())

    imgdir = args[0]

    mani(imgdir)

    return True


if __name__ == '__main__':
    main()
