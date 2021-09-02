"""
generate labels of training images
"""

import sys
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from schnablelab.apps.Tools import GenDataFrameFromPath
from schnablelab.apps.base import ActionDispatcher, OptionParser

def main():
    actions = (
        ('genlabel', 'genearte label for training image files'),
        ('extract_info', 'extract testing and prediction results from dpp log file'),
        ('statistics', 'calculate CountDiff, AbsCountDiff, MSE, Agreement, r2, p_value, and draw scatter, bar plots'),
        ('gentraining', 'create balanced training dataset for each class'),
        ('gentesting', 'create balanced testing dataset for each class'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def gentesting(args):
    """
    %prog source_imgs_dir source_imgs_csv training_imgs_csv testing_imgs_per_cls output_dir

    create the balanced testing dataset for each class
    """
    p = OptionParser(gentraining.__doc__)
    p.add_option('--header', default=None,
        help = 'spefiy if the source csv file has header')
    p.add_option('--comma_sep', default=True,
        help = 'spefiy if the csv file is separated by comma')
    p.add_option('--groupby_col', default=1,
        help = 'spefiy the groupy column. 0: 1st column; 1: 2nd column')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    source_dir, source_csv, training_csv, ipc, testing_dir = args # ipc: number of images per class.

    # read the source csv file
    if opts.header and opts.comma_sep: # without header with ,
        df0 = pd.read_csv(source_csv, header=None)
    elif (not opts.header) and opts.comma_sep: # with header with ,
        df0 = pd.read_csv(source_csv)
    elif not (opts.header and opts.comma_sep): # with header with tab/space
        df0 = pd.read_csv(source_csv, delim_whitespace=True)
    else:
        print('keke... implement this option first!')
    print('shape of source csv %s: %s'%(mycsv, df0.shape))

    

def gentraining(args):
    """
    %prog source_imgs_dir source_imgs_csv cls_range(eg: 5-10) training_imgs_per_cls output_dir

    create the balanced training dataset where each class has the same number of images
    """
    p = OptionParser(gentraining.__doc__)
    p.add_option('--header', default=None,
        help = 'spefiy if the source csv file has header')
    p.add_option('--comma_sep', default=True,
        help = 'spefiy if the csv file is separated by comma')
    p.add_option('--groupby_col', default=1,
        help = 'spefiy the groupy column. 0: 1st column; 1: 2nd column')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    source_dir, source_csv, cls_range, ipc, train_dir = args # ipc: number of images per class. 
    
    # read the source csv file
    if opts.header and opts.comma_sep: # without header with ,
        df0 = pd.read_csv(source_csv, header=None)
    elif (not opts.header) and opts.comma_sep: # with header with ,
        df0 = pd.read_csv(source_csv)
    elif not (opts.header and opts.comma_sep): # with header with tab/space
        df0 = pd.read_csv(source_csv, delim_whitespace=True)
    else:
        print('keke... implement this option first!')
    print('shape of source csv %s: %s'%(mycsv, df0.shape))

    # choose df in the class range
    gc = int(opts.groupby_col)
    fn = 0 if gc==1 else 1 # file name column
    st, ed = [int(n) for n in cls_range.split('-')]
    df1 = df0[df0.iloc[:, gc].isin(range(st,ed+1))]

    summ = df1.iloc[:, gc].value_counts().sort_index()
    print('image distribution in each class:\n', summ)
    
    # pick up the data and move to the specified training dir
    ipc = int(ipc)
    sr_dir = Path(source_dir)
    tr_dir = Path(train_dir)
    for lc, grp in df1.groupby(df1.columns[gc]):
        print('group index: %s'%lc)
        grpsub = grp.sample(ipc)
        print(grpsub.shape)
        for fn in grpsub.iloc[:,fc]:
            copy(sr_dir/fn, train_dir)
    print('Done!')    
    
def genlabel(args):
    """
    %prog train_dir label_fn

    generate my_labels.csv in the training dir
    """
    p = OptionParser(genlabel.__doc__)
    p.add_option('--header', default=False,
        help = 'if add the header to csv or not!')
    p.add_option('--source_dir', default='/work/schnablelab/cmiao/LeafCounts_JinliangData/Raw_RealImages',
        help = 'spefiy the source dir')
    p.add_option('--source_csv', default='labels_4633.csv',
        help = 'spefiy the csv file in source dir')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    train_dir, label_fn, = args

    train_path = Path(train_dir)
    train_df = GenDataFrameFromPath(train_path)
    total_n = train_df.shape[0]
    print('total %s images.'%total_n)

    all_real_dir = Path(opts.source_dir)
    all_real_df = pd.read_csv(all_real_dir/opts.source_csv).set_index('sm')

    train_df['real'] = train_df['fn'].isin(all_real_df.index)

    real_n = train_df['real'].sum()
    fake_n = total_n-real_n
    print('%s real images.'%real_n)
    print('%s fake images.'%fake_n)

    if real_n != 0:
        # asign real labels
        real_idx = train_df['fn'][train_df['real']]
        real_labels = all_real_df.loc[real_idx, 'LeafCounts'].values
        train_df.loc[train_df['real'], 'real']=real_labels

    if fake_n != 0:
        fake_df = train_df[train_df['real'].isin([False])]
        fake_labels = fake_df['fn'].apply(lambda x: int(x.split('_')[0])).values
        train_df.loc[train_df['real'].isin([False]), 'real']=fake_labels
    if opts.header:
        train_df[['fn', 'real']].to_csv(train_path/label_fn, index=False)
    else:
        train_df[['fn', 'real']].to_csv(train_path/label_fn, index=False, header=False)
    print('Done, check %s'%label_fn)

def convert2list(lines):
    vs = []
    for i in lines:
        j = i.strip().lstrip('[').rstrip(']').split()
        for it in j:
            try:
                v = float(it)
                vs.append(v)
            except:
                pass
    return vs

def extract_num(lists):
    if len(lists)==1:
        nums = [float(i) for i in lists[0].split('[')[-1].split(']')[0].split()]
    else:
        nums = []
        for line in lists:
            if '[' in line:
                ns = [float(i) for i in line.split('[')[-1].split()]
                nums.extend(ns)
            elif ']' in line:
                ns = [float(i) for i in line.split(']')[0].split()]
                nums.extend(ns)
            else:
                ns = [float(i) for i in line.split()]
                nums.extend(ns)
    return nums

def extract_info(args):
    """
    %prog log_file output_fn
    
    extract testing and prediction results from dpp log file
    """
    p = OptionParser(extract_info.__doc__)
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    logfile, opp, = args
    
    f0 = open(logfile)
    all_lines = f0.readlines()
    test_idx, predict_idx, hist_idx = 0, 0, 0
    for i,j in enumerate(all_lines):
        if 'All test labels:' in j:
            test_idx = i
        if 'All predictions:' in j:
            predict_idx = i
        if 'Histogram of ' in j:
            hist_idx = i

    test_lines = all_lines[test_idx+1: predict_idx]
    ground_truth = extract_num(test_lines)
    #print(len(ground_truth), '\n', ground_truth)

    predict_lines = all_lines[predict_idx+1: hist_idx]
    prediction = extract_num(predict_lines)
    #print(len(prediction), '\n', prediction)

    df = pd.DataFrame(dict(zip(['groundtruth', 'prediction'], [groundtruth, prediction])))
    df.to_csv(opp, index=False, sep='\t')
    print('Done! check %s'%opp)

def statistics(args):
    """
    %prog 2cols_csv cls_range(eg: 5-10) output_prefix 

    calculate CountDiff, AbsCountDiff, MSE, Agreement, r2, p_value, and scatter, bar plots
    """
    p = OptionParser(statistics.__doc__)
    p.add_option('--header', default=None,
        help = 'spefiy if csv file has header')
    p.add_option('--cutoff_agreement', default=0.5,
        help = 'spefiy the cutoff counted for agreement calculation')
    opts, args = p.parse_args(args)
    if len(args) == 0:
        sys.exit(not p.print_help())
    mycsv,cls_range, opp, = args
    
    df_compare = pd.read_csv(mycsv) if opts.header else pd.read_csv(mycsv, header=None)
    print('shape of %s: %s'%(mycsv, df_compare.shape))
    
    slope, intercept, r_value, p_value, std_err = linregress(df_compare['groundtruth'], df_compare['prediction'])
    df_compare['prediction_adjusted'] = df_compare['prediction']
    #df_compare['prediction_adjusted'] = (df_compare['prediction']-intercept)/slope

    df_compare['diff'] = df_compare['groundtruth'] - df_compare['prediction_adjusted']
    df_compare['abs_diff'] = np.abs(df_compare['diff'])

    mi, ma = df_compare['diff'].min(), df_compare['diff'].max()
    mi_int = np.ceil(mi) if mi>0 else np.floor(mi)
    ma_int = np.ceil(ma) if ma>0 else np.floor(ma)
    bins = np.arange(mi_int, ma_int+1)
    cats = pd.cut(df_compare['diff'], bins)
    ax1 = pd.value_counts(cats).sort_index().plot.bar(color='blue')
    plt.xticks(rotation=40)
    ax1.set_xlabel('Relative Count Differece')
    ax1.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('%s_diff_bar.png'%opp)
    print('diff bar/histogram plot done!')

    aggrmt = (df_compare['abs_diff']<=float(opts.cutoff_agreement)).sum()/df_compare.shape[0]
    print('agreement is defined as diff <= %s.'%opts.cutoff_agreement)
    
    
    rmse = np.sqrt(mean_squared_error(df_compare['groundtruth'], df_compare['prediction_adjusted']))

    bt = int(cls_range.split('-')[0])
    tp = int(cls_range.split('-')[1])
    x = np.array([bt-0.5,tp+0.5])
    y = slope*x+intercept

    mean, std = df_compare['diff'].mean(), df_compare['diff'].std()
    abs_mean, abs_std = df_compare['abs_diff'].mean(), df_compare['abs_diff'].std()
    txt = 'CountDiff: %.2f(%.2f)\n'%(mean, std)
    txt += 'AbsCountDiff: %.2f(%.2f)\n'%(abs_mean, abs_std)
    txt += 'r2: %.2f\n'%r_value**2
    txt += 'p value: %s\n'%p_value
    txt += 'RMSE: %.2f\n'%rmse
    txt += 'Agreement: %.2f\n'%aggrmt
    txt += 'y = %.2fx + %.2f'%(slope, intercept)
    with open('%s.statics'%opp, 'w') as f1:
        f1.write(txt)
    print('statistics done!')

    ax2 = df_compare.plot.scatter(x='groundtruth', y='prediction', alpha=0.5, figsize=(7,7), edgecolor='k')        
    #ax2.set_xlim(bt-0.9, tp+0.9)
    #ax2.set_ylim(bt-0.9, tp+0.9)
    ax2.plot(x,y, color='red', linewidth=2)
    ax2.text(x=bt, y=tp-2, s = txt, fontsize=12, color='red')
    plt.savefig('%s_scatter.png'%opp)
    print('scatter plot done!')

if __name__ == "__main__":
    main()
