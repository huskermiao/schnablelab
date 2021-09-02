# -*- coding: UTF-8 -*-

"""
Run GEMMA command or generate the coresponding slurm job file. Find details in GEMMA manual at <http://www.xzlab.org/software/GEMMAmanual.pdf>
"""

import os.path as op
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser
from schnablelab.apps.headers import Slurm_header
from schnablelab.apps.natsort import natsorted

# the location of gemma executable file
gemma = op.abspath(op.dirname(__file__))+'/../apps/gemma'

def main():
    actions = (
        ('GLM', 'Performe GWAS using general linear model'),
        ('MLM', 'Performe GWAS using mixed linear model '),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def GLM(args):
    """
    %prog GLM GenoPrefix Pheno Outdir
    RUN automated GEMMA General Linear Model
    """ 
    p = OptionParser(GLM.__doc__)
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    
    if len(args) == 0:
        sys.exit(not p.print_help())
    GenoPrefix, Pheno, Outdir = args
    meanG, annoG = GenoPrefix+'.mean', GenoPrefix+'.annotation'
    outprefix = Pheno.split('.')[0]
    cmd = '%s -g %s -p %s -a %s -lm 4 -outdir %s -o %s' \
        %(gemma, meanG, Pheno, annoG, Outdir, outprefix)
    print('The command running on the local node:\n%s'%cmd)

    h = Slurm_header
    header = h%(opts.time, opts.memory, opts.prefix, opts.prefix, opts.prefix)
    header += cmd
    f = open('%s.glm.slurm'%outprefix, 'w')
    f.write(header)
    f.close()
    print('slurm file %s.glm.slurm has been created, you can sbatch your job file.'%outprefix)


def MLM(args):
    """
    %prog MLM GenoPrefix('*.mean' and '*.annotation') Pheno Outdir
    RUN automated GEMMA Mixed Linear Model
    """ 
    p = OptionParser(MLM.__doc__)
    p.add_option('--kinship', default=False, 
        help = 'specify the relatedness matrix file name')
    p.add_option('--pca', default=False, 
        help = 'specify the principle components file name')
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    
    if len(args) == 0:
        sys.exit(not p.print_help())
    GenoPrefix, Pheno, Outdir = args
    meanG, annoG = GenoPrefix+'.mean', GenoPrefix+'.annotation'
    outprefix = '.'.join(Pheno.split('/')[-1].split('.')[0:-1])
    cmd = '%s -g %s -p %s -a %s -lmm 4 -outdir %s -o %s' \
        %(gemma, meanG, Pheno, annoG, Outdir, outprefix)
    if opts.kinship:
        cmd += ' -k %s'%opts.kinship
    if opts.pca:
        cmd += ' -c %s'%opts.pca
    print('The command running on the local node:\n%s'%cmd)

    h = Slurm_header
    header = h%(opts.time, opts.memory, opts.prefix, opts.prefix, opts.prefix)
    header += cmd
    f = open('%s.mlm.slurm'%outprefix, 'w')
    f.write(header)
    f.close()
    print('slurm file %s.mlm.slurm has been created, you can sbatch your job file.'%outprefix)

if __name__ == "__main__":
    main()
