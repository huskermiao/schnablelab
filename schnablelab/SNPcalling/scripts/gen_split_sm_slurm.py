slurm_header = '''#!/bin/sh
#SBATCH --time=167:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=10000       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=%s
#SBATCH --error=./%s.err
#SBATCH --output=./%s.out

%s
'''
script = '/lustre/work/schnablelab/cmiao/TimeSeriesGWAS/Genotype_GBS/GBS_Raw_Data/split_sample.py'
data_dir = '/work/schnablelab/cmiao/TimeSeriesGWAS/Genotype_GBS/GBS_Raw_Data/1906UNHX-0018/'

def genslurm(p, bsp, N):
    p, bsp, N = int(p), int(bsp), int(N)
    for i in range(1, N+1):
        print(i) 
        r1 = 'G121_P%s_Bsp_%s_R1_%s.fastq'%(p, bsp, i)
        r2 = 'G121_P%s_Bsp_%s_R2_%s.fastq'%(p, bsp, i)
        bgp = (p-1)*2+bsp
        cmd = 'python %s %s %s bgpR%s'%(script, r1, r2, bgp)
        print(cmd)
        pfx = 'p%s_bsp%s_n%s'%(p,bsp,i)
        with open('split_sm_%s.slurm'%pfx, 'w') as f:
            f.write(slurm_header%(pfx, pfx, pfx, cmd))

import sys
if __name__ == '__main__':
    if len(sys.argv)==4:
        genslurm(*sys.argv[1:])
    else:
        print('p bsp n')
