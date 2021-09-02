"""
basic support for running library as script
"""

import os
import sys
import glob
import time
import math
import pandas as pd
import os.path as op
from schnablelab.apps.Tools import eprint
from schnablelab.apps.natsort import natsorted
from schnablelab import __copyright__, __version__
from optparse import OptionParser as OptionP, OptionGroup, SUPPRESS_HELP

JSHELP = "schnablelab utility libraries v%s [%s]\n" % (__version__, __copyright__)

try:
    basestring
except NameError:
    basestring = str

class ActionDispatcher(object):
    """
    This class will be invoked
    a) when the base package is run via __main__, listing all MODULESs
    a) when a directory is run via __main__, listing all SCRIPTs
    b) when a script is run directly, listing all ACTIONs

    This is controlled through the meta variable, which is automatically
    determined in get_meta().
    """

    def __init__(self, actions):
        self.actions = actions
        if not actions:
            actions = [(None, None)]
        self.valid_actions, self.action_helps = zip(*actions)

    def get_meta(self):
        args = splitall(sys.argv[0])[-3:]
        args[-1] = args[-1].replace(".py", "")
        if args[-2] == "schnablelab":
            meta = "MODULE"
        elif args[-1] == "__main__":
            meta = "SCRIPT"
        else:
            meta = "ACTION"
        return meta, args

    def print_help(self):
        meta, args = self.get_meta()
        if meta == 'MODULE':
            del args[0]
            args[-1] = meta
        elif meta == 'SCRIPT':
            args[-1] = meta
        else:
            args[-1] += " " + meta

        help = 'Usage:\n    python -m %s\n\n\n' % ('.'.join(args))
        help += 'Available %ss:\n' % meta
        max_action_len = max(len(action) for action, ah in self.actions)
        #for action, action_help in sorted(self.actions):
        for action, action_help in self.actions:
            action = action.rjust(max_action_len + 4)
            help += " | ".join((action, action_help[0].upper() +
                                action_help[1:])) + '\n'
        help += "\n" + JSHELP
        sys.stderr.write(help)
        sys.exit(1)

    def dispatch(self, globals):
        from difflib import get_close_matches
        meta = 'ACTION'
        if len(sys.argv) == 1:
            self.print_help()
        action = sys.argv[1]
        if not action in self.valid_actions:
            eprint("[error] %s not a valid %s\n"% (action, meta))
            alt = get_close_matches(action, self.valid_actions)
            eprint(sys.stderr, "Did you mean one of these?\n\t%s\n" % (", ".join(alt)))
            self.print_help()
        globals[action](sys.argv[2:])

class OptionParser(OptionP):
    def __init__(self, doc):
        OptionP.__init__(self, doc, epilog=JSHELP)

    def parse_args(self, args=None):
        dests = set()
        ol = []
        for g in [self] + self.option_groups:
            ol += g.option_list
        for o in ol:
            if o.dest in dests:
                continue
            self.add_help_from_choices(o)
            dests.add(o.dest)
        return OptionP.parse_args(self, args)

    def add_help_from_choices(self, o):
        if o.help == SUPPRESS_HELP:
            return

        default_tag = "%default"
        assert o.help, "Option %s do not have help string" % o
        help_pf = o.help[:1].upper() + o.help[1:]
        if "[" in help_pf:
            help_pf = help_pf.rsplit("[", 1)[0]
        help_pf = help_pf.strip()

        if o.type == "choice":
            if o.default is None:
                default_tag = "guess"
            ctext = "|".join(natsorted(str(x) for x in o.choices))
            if len(ctext) > 100:
                ctext = ctext[:100] + " ... "
            choice_text = "must be one of %s" % ctext
            o.help = "%s, %s [default: %s]" % (help_pf, choice_text, default_tag)
        else:
            o.help = help_pf
            if o.default is None:
                default_tag = "disabled"
            if o.get_opt_string() not in ("--help", "--version") \
                    and o.action != "store_false":
                o.help += " [default: %s]" % default_tag

    def add_slurm_opts(self, job_prefix='myjob'):
        group = OptionGroup(self, 'Slurm job options')
        group.add_option('--time', type='int', default=120,
                            help='the maximum running time(hour)')
        group.add_option('--memory', type='int', default=10000,
                            help='the maximum running memory(Mb). crane12(35g), crane20(384g), crane36(4*256g), crane36(4*512g). k20(5000), k40(12000), p100(16000)')
        group.add_option('--job_prefix', default=job_prefix,
                            help='prefix of job name and log file')
        group.add_option('--pu_type', default='cpu', choices=('cpu', 'gpu'),
                            help='PU type')
        group.add_option('--partition', default='jclarke', choices=('jclarke', 'schnablelab', 'gpu', 'batch'),
                            help = 'partition name')
        group.add_option('--ncpus_per_node', type='int', default=1,
                            help = 'number of cpus per node')
        group.add_option('--gpu_model',
                            help = "gpu model. 'gpu_k40', 'gpu_p100|gpu_v100' (requires '--pu_type' is 'gpu')")
        group.add_option('--ncmds_per_slurm', type='int', default=1,
                            help ='number of commands added to slurm')
        self.add_option_group(group)
    
    def set_cpus(self, cpus=0):
        """
        Add --cpus options to specify how many threads to use.
        """
        from multiprocessing import cpu_count

        max_cpus = cpu_count()
        if not 0 < cpus < max_cpus:
            cpus = max_cpus
        self.add_option("--cpus", default=cpus, type="int",
                     help="Number of CPUs to use, 0=unlimited [default: %default]")

def get_module_docstring(filepath):
    "Get module-level docstring of Python module at filepath, e.g. 'path/to/file.py'."
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], basestring):
        docstring = co.co_consts[0]
    else:
        docstring = None
    return docstring

def get_today():
    """
    Returns the date in 2010-07-14 format
    """
    from datetime import date
    return str(date.today())

def splitall(path):
    allparts = []
    while True:
        path, p1 = op.split(path)
        if not p1:
            break
        allparts.append(p1)
    allparts = allparts[::-1]
    return allparts

def dmain(mainfile, type='action'):
    cwd = op.dirname(mainfile)
    pyscripts = [x for x in glob(op.join(cwd, "*", '__main__.py'))] \
        if type == "module" \
        else glob(op.join(cwd, "*.py"))
    actions = []
    for ps in sorted(pyscripts):
        action = op.basename(op.dirname(ps)) \
            if type == 'module' \
            else op.basename(ps).replace('.py', '')
        if action.startswith('_') or action=='base':
            continue
        pd = get_module_docstring(ps)
        action_help = [x.rstrip(":.,\n") for x in pd.splitlines(True)
                       if len(x.strip()) > 10 and x[0] != '%'][0] \
            if pd else "no docstring found"
        actions.append((action, action_help))
    a = ActionDispatcher(actions)
    a.print_help()

def glob(pathname, pattern=None):
    """
    Wraps around glob.glob(), but return a sorted list.
    """
    import glob as gl
    if pattern:
        pathname = op.join(pathname, pattern)
    return natsorted(gl.glob(pathname))

def iglob(pathname, patterns):
    """
    Allow multiple file formats. This is also recursive. For example:

    >>> iglob("apps", "*.py,*.pyc")
    """
    matches = []
    patterns = patterns.split(",") if "," in patterns else listify(patterns)
    for root, dirnames, filenames in os.walk(pathname):
        matching = []
        for pattern in patterns:
            matching.extend(fnmatch.filter(filenames, pattern))
        for filename in matching:
            matches.append(op.join(root, filename))
    return natsorted(matches)

def cutlist(lst, n):
    """
    cut list to different groups with equal size
    yield index range for each group and the group itself
    """
    series = pd.Series(lst)
    ctg = pd.qcut(series.index, n)
    grps = series.groupby(ctg)
    for _, grp in grps:
        idx = grp.index.tolist()
        if grp.shape[0] ==1:
            yield str(idx[0]), grp
        else:
            st, ed = idx[0], idx[-1]
            yield '%s-%s' % (st, ed), grp

Slurm_header = '''#!/bin/sh
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ncpus_per_node}
#SBATCH --time={time}:00:00          # Run time in hh:mm:ss
#SBATCH --mem={memory}       # --mem-per-cpu Maximum memory required per CPU (in megabytes)
#SBATCH --job-name={jobname}
#SBATCH --error=./{jobname}.err
#SBATCH --output=./{jobname}.out
'''

def put2slurm(cmds, slurm_dict):
    '''
    args:
        cmds (list): list of all the commands
    kwargs:
        ncmds_per_slurm (int): number of commands for each slurm, default 1
        partition: jclarke, schnablelab, gpu, default jclarke
        time: time in hour, default 120
        mem: memory in hour, default 10_000
        job_prefix: job name, default 'job'
        ncpus_per_node: numebr of CPUs requested in a node, default 1
        cmd_header: header command, ex: 'ml bcftools', default None
        gpu: if request a gpu, default False
        gpu_model: request a specified gpu model, default None, 'gpu_k40', 'gpu_p100|gpu_v100'
    '''
    if len(cmds) < slurm_dict['ncmds_per_slurm']:
        sys.exit('# of commands per slurm > # of cmds !!!')
    n_slurms = math.ceil(len(cmds)/slurm_dict['ncmds_per_slurm'])

    for range_label, grp in cutlist(cmds, n_slurms):
        jobname = '%s_%s'%(slurm_dict['job_prefix'], range_label) if len(cmds)>1 else slurm_dict['job_prefix']
        slurm_dict['jobname'] = jobname
        slurm_header = Slurm_header.format(**slurm_dict)
        if slurm_dict['pu_type']=='gpu':
            slurm_header += '#SBATCH --gres=gpu\n'
            if slurm_dict['gpu_model']:
                slurm_header += "#SBATCH --constraint='{gpu_model}'\n".format(gpu_model=slurm_dict['gpu_model'])
        slurm_header += '\n'
        if 'cmd_header' in slurm_dict:
            slurm_header += '%s\n'%slurm_dict['cmd_header']
        for cmd in grp:
            slurm_header += '%s\n'%cmd
        with open('%s.slurm'%jobname, 'w') as f:
            f.write(slurm_header)
        print('%s.slurm is ready to submit!'%jobname)
