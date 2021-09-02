"""
require personal terminal used for computation
"""
import sys
from schnablelab.apps.base import ActionDispatcher, OptionParser
from subprocess import call

def main():

    actions = (
        ('cpu', 'request cpu node'),
        ('gpu', 'request gpu node'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def cpu(args):
    """
    %prog  
    request a cpu node from hcc.
    """
    p = OptionParser(cpu.__doc__)
    p.add_option("--partition", default="jclarke", choices=('batch', 'jclarke'),
                help="which partition? [default: %default]")
    p.add_option("--memory", default="10240",
                help="specify the how much memory [default: %default]")
    p.add_option("--time", default='20',
                help = "specify the time (hour) [default: %default]")
    opts, args = p.parse_args(args)
    if len(args) == 0:
        print('add --help to see options.\n')
        cmd = 'srun --partition=%s --mem-per-cpu=%s --ntasks-per-node=6 --nodes=1 --time=%s:0:0 --pty $SHELL\n'%(opts.partition, opts.memory, opts.time)
        print(cmd)
        #call(cmd, shell=True)
    else:
        sys.exit(not p.print_help())

def gpu(args):
    """
    %prog
    request a gpu node from hcc.
    """
    p = OptionParser(gpu.__doc__)
    p.add_option("--memory", default="12000",
                help="specify the how much memory [default: %default]")
    p.add_option("--time", default='20',
                help = "specify the time (hour) [default: %default]")
    p.add_option("--model", default='gpu_k40', choices=('gpu_p100', 'gpu_k20', 'gpu_k40'),
                help = "specify gpu mode, p100:16gb, k40:12gb, k20:5bg [default: %default]")
    opts, args = p.parse_args(args)
    if len(args) == 0:
        print('add --help to see options.\n')
        cmd = 'srun --partition=schnablelab --gres=gpu --constraint=%s --mem-per-cpu=%s --ntasks-per-node=1 --nodes=1 --time=%s:0:0 --pty $SHELL\n'%(opts.model, opts.memory, opts.time)
        print(cmd)
        #call(cmd, shell=True)
    else:
        sys.exit(not p.print_help())

if __name__ == "__main__":
    main()
