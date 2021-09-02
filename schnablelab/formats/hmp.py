# hapmap format class
from pathlib import Path
import os.path as op

# regular instance method. pass in self(instance)
# class method used as alternative constructers. pass in cls(class)
# static method. pass nothing. why we still incorporate into our class?


class hmp:
    # class varibales. use either class.variable or instance.variable to access the classs varibales
    tassel = Path(__file__).absolute().parents[1] / 'apps/tassel-5-standalone/run_pipeline.pl'

    def __init__(self, fn):
        self.fn = fn
        self.prefix = Path(fn).stem

    def to_vcf(self):
        # return the tassel comand for the conversion
        cmd = '{tassel} -Xms512m -Xmx10G -fork1 -h {hmp} -export -exportType VCF\n'.format(self.tassel, self.fn)
        return cmd

    def to_mean_annot(self):
        # convert hmp format to bimbam format composed of a mean and an annotation file.
        self.mean = self.prefix + '.mean'
        self.annotation = self.prefix + '.annotation'
        with open(self.fn) as rf, open(self.mean, 'w') as wf1, open(self.annotation, 'w') as wf2:
            for i in rf:
                j = i.split()
                rs = j[0]
                try:
                    ref, alt = j[1].split('/')
                    key_dict = {ref + ref: '0', alt + alt: '2', ref + alt: '1', alt + ref: 1}
                except:
                    print('omit {rs}.'.format(rs))
                    continue
                newNUMs = [key_dict[k] for k in j[11:]]
                newline = '{rs},{ref},{alt},{nums}\n'.format(rs, ref, alt, ','.join(newNUMs))
            wf1.write(newline)
            chrom, pos = j[2], j[3]
            wf2.write('{rs},{pos},{chromosome}\n'.format(rs, pos, chrom))
"""
    def to_numeric_col(self):
        self.GD = self.prefix + '.GD'
        self.GM = self.prefix + '.GM'
        with open(self.fn) as rf, open(self.GD, 'w') as wf1, open(self.GM, 'w') as wf2:

    def to_numeric_row(self, kind='row'):
        self.GD = self.prefix + '.GD'
        self.GM = self.prefix + '.GM'
        with open(self.fn) as rf, open(self.GD, 'w') as wf1, open(self.GM, 'w') as wf2:
"""
