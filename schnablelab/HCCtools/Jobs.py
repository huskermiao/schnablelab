# -*- coding: UTF-8 -*-

"""
create, submit, canceal jobs. 
Find more details at HCC document: 
<https://hcc-docs.unl.edu>
"""

import os.path as op
from pathlib import Path
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser, glob,iglob
from schnablelab.apps.natsort import natsorted
from subprocess import call
from subprocess import Popen
import subprocess

def main():
    actions = (
        ('submit', 'submit a batch of jobs or all of them'),
        ('quickjob', 'create a quick slurm job'),
        ('cancel', 'canceal running, pending or all jobs'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def submit(args):
    """
    %prog dir

    Submit part of job in the dir or all jobs
    """
    p = OptionParser(submit.__doc__)
    p.add_option("--pattern", default="*.slurm", 
                 help="specify the patter of your slurm job, remember to add quotes [default: %default]")
    p.add_option("--partition", default='jclarke', choices=('batch', 'jclarke', 'gpu', 'schnablelab'),
                help = "choose which partition you are going to submit [default: %default]")
    p.add_option("--range", default='all', 
                 help="how many jobs you gonna submit this time. exp: '1-10', '11-20', 'all'. 1-based coordinate")
    opts, args = p.parse_args(args)
    if len(args) != 1:
        sys.exit(not p.print_help())

    folder, = args
    partition = '-p %s'%opts.partition
    alljobs = ['sbatch %s %s'%(partition, i) for i in Path(folder).glob(opts.pattern)]
    print("Total %s jobs under '%s'"%(len(alljobs), folder))

    if opts.range == 'all':
       for i in alljobs:
          print(i)
          call(i, shell=True)
    else:
        start, end = int(opts.range.split('-')[0]), int(opts.range.split('-')[1])
        if end <= len(alljobs):
            for i in alljobs[start-1 : end]:
                print(i)
                call(i, shell=True)
            print('%s of total %s were submitted. [%s to %s] this time.' \
                %(len(alljobs[start-1 : end]), len(alljobs), start, end))
        else:
            print('jobs exceed the limit')

def cancel(args):
    """
    %prog
    
    Cancel jobs on HCC
    """
    p = OptionParser(cancel.__doc__)
    p.add_option("--status", default='running', choices=('running', 'pending'),
                 help="specify the status of the jobs you want to cancel [default: %default]")
    p.add_option("--partition", default='jclarke', choices=('gpu', 'batch', 'jclarke'),
                 help="specify the partition where jobs are runnig [default: %default]")
    opts, args = p.parse_args(args)
    if len(args) != 0:
        sys.exit(not p.print_help())
    myjobs = Popen('squeue -u cmiao', shell = True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
    running_jobs, pending_jobs, others = [], [], []
    for i in myjobs.split('\n'):
        j = i.strip().split()
        try:
            if j[4] == 'R':
                running_jobs.append(j[0])
            elif j[4] == 'PD':
                pending_jobs.append(j[0])
            else:
                others.append(j[0])
        except:
            pass
    cmd = 'scancel %s' %(' '.join(running_jobs)) \
        if opts.status == 'running' \
        else 'scancel %s' %(' '.join(pending_jobs))

    print(cmd)

def quickjob(args):
    """
    %prog job_name cmd
    generate a qucik slurm job for the cmd
    """
    if len(args) == 0:
        sys.exit('specify job_name and command')
    jobname, *cmd = args
    cmd = ' '.join(cmd)
    header = '''#!/bin/bash
#SBATCH --time=120:00:00 
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=%s
#SBATCH --error=%s.err
#SBATCH --output=%s.out

'''
    header = header%(jobname, jobname, jobname)
    header += cmd
    with open('%s.slurm'%jobname, 'w') as f:
        f.write(header)
    print('slurm file %s.slurm has been created...'%jobname)

if __name__ == "__main__":
    main()







































