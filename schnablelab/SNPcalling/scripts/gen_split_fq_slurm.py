header='''#!/bin/sh
#SBATCH --time=167:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=10000       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=split_%s
#SBATCH --error=./split_%s.err
#SBATCH --output=./split_%s.out
#SBATCH --qos=ac_schnablelab

%s
'''

tls = [749160852, 936812020, 858563736, 761914868, 701120380, 1015314960, 847007300, 740808360]
for p in range(1, 5):
    for i in [1, 2]:
        tl_idx = (p-1)*2+i-1
        tl = tls[tl_idx]
        for r in [1, 2]:
            print('P: %s, Bsp: %s, R: %s, TL: %s'%(p, i, r, tl))
            cmd = 'python -m schnablelab.SNPcalling.Postprocess split 70 G121_P%s_Bsp_%s_R%s.fastq --total_lines %s'%(p, i, r, tl)
            print(cmd)
            with open('G121_P%s_Bsp_%s/Splitting_P%s_Bsp%s_R%s.slurm'%(p, i, p, i, r), 'w') as f:
                prf = 'p%s_bsp%s_r%s'%(p,i,r)
                f.write(header%(prf, prf, prf, cmd))
