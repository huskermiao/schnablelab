# -*- coding: UTF-8 -*-
"""
Generate MVP slurm job file. Find more details about MVP at <https://github.com/XiaoleiLiuBio/MVP>.
"""

import os.path as op
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser
from schnablelab.apps.headers import Slurm_header
from schnablelab.apps.headers import MVP_Data_header, MVP_Run_header
from schnablelab.apps.natsort import natsorted


def main():
    actions = (
        ('PrepareData', 'Prepare the data including Genotype, Map, Kinship, and PC files for running MVP.'),
        ('RunMVP', 'Run both MLM and FarmCPU models in MVP.'),

    )
    p = ActionDispatcher(actions)
    p.dispatch(globals())


def PrepareData(args):
    """
    %prog PrepareData hmp out_prefix

    Prepare the genotype, map, kinship, and covariants for running MVP
    """
    p = OptionParser(PrepareData.__doc__)
    p.set_slurm_opts()
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    hmp, op, = args  # op: output prefix
    f1 = open('%s.mvp.data.R' % (op), 'w')
    f1.write(MVP_Data_header % (hmp, op))
    f1.close()
    f2 = open('%s.mvp.data.slurm' % op, 'w')
    header = Slurm_header % (opts.time, opts.memory, op, op, op)
    header += 'module load R\n'
    header += 'R CMD BATCH %s.mvp.data.R\n' % op
    f2.write(header)
    f2.close()
    print('%s.mvp.data.R and %s.mvp.data.slurm have been created. you can submit slurm job now.' % (op, op))


def RunMVP(args):
    """
    %prog RunMVP phenotype GMKC_prefix

    run MVP given the phenotype and prefix of GenotypeMapKinshipCovariates files.
    """
    p = OptionParser(RunMVP.__doc__)
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    pheno, op, = args  # op: output prefix
    f1 = open('%s.mlm.farmcpu.R' %opts.prefix, 'w')
    f1.write(MVP_Run_header % (pheno, op, op, op, op))
    f1.close()
    f2 = open('%s.mlm.farmcpu.slurm' %opts.prefix, 'w')
    header = Slurm_header % (opts.time, opts.memory, opts.prefix,opts.prefix,opts.prefix)
    header += 'module load R\n'
    header += 'R CMD BATCH %s.mlm.farmcpu.R\n' % opts.prefix
    f2.write(header)
    f2.close()
    print('%s.mlm.farmcpu.R and %s.mlm.farmcpu.slurm have been created.' % (opts.prefix,opts.prefix))


def plot(args):
    """
    %prog plot gwas_out result_prefix

    plt MVP results using MVP.Report function.
    https://github.com/XiaoleiLiuBio/MVP
    """
    p = OptionParser(plot.__doc__)
    p.set_slurm_opts(jn=True)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    gwasfn, op, = args  # op: output prefix
    f1 = open('%s.plot.R' %op, 'w')
    cmds = '''
    library('MVP')
    myData = read.csv(%s)
    MVP.Report(myData, plot.type='m', col=c("dodgerblue4","deepskyblue"), LOG10=TRUE, ylim=NULL, th
reshold=8.9e-8, threshold.col='grey', chr.den.col=NULL, file='png', memo='MLM', dpi=300)
    '''
    f1.write(MVP_Run_header % (pheno, op, op, op, op))
    f1.close()
    f2 = open('%s.mlm.farmcpu.slurm' %opts.prefix, 'w')
    header = Slurm_header % (opts.time, opts.memory, opts.prefix,opts.prefix,opts.prefix)
    header += 'module load R\n'
    header += 'R CMD BATCH %s.mlm.farmcpu.R\n' % opts.prefix
    f2.write(header)
    f2.close()
    print('%s.mlm.farmcpu.R and %s.mlm.farmcpu.slurm have been created.' % (opts.prefix,opts.prefix))


if __name__ == '__main__':
    main()




















