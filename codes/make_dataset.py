"""
Example usage:
    1. For 1-d signal
        python make_dataset.py --data_type signal --fs 500 --upsampling_factor 1 --sig_scaling both
    2. For 2-d STFT
        python make_dataset.py --data_type stft --fs 500 --upsampling_factor 1 --sig_scaling both --stft_scaling both --nperseg 128 --t_len 500 --stretch True --stretch_size 500
    3. For 2-d CWT
        python make_dataset.py --data_type cwt --fs 500 --upsampling_factor 1 --sig_scaling both --cwt_scaling both --width_min 0.1 --width_max 30.1 --width_inc 0.1 --wavelet ricker --cwt_scaling both --length 5000 --cwt_resize True --cwt_height 300 --cwt_width 1000 
"""

import os, glob, ast, wfdb, argparse, json, tqdm
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import SignalDataset, CWTDataset, STFTDataset, get_dataset, standard_normalization, minmax_normalization, both_normalization
from train_options import get_opt  # 返回的是 config

import torch


def get_input_config():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    opt = get_opt()  # 返回的是 config
    # opt for base
    target_label = opt.target_label
    data_type = opt.data_type
    # opt for signal
    fs = opt.fs
    upsampling_factor = opt.upsampling_factor
    sig_scaling = opt.sig_scaling
    # opt for stft
    nperseg = opt.nperseg
    t_len = opt.t_len
    stft_scaling = opt.stft_scaling
    # opt for cwt
    width_min = opt.width_min
    width_max = opt.width_max
    width_inc = opt.width_inc
    wavelet_type = opt.wavelet_type
    cwt_scaling = opt.cwt_scaling
    cwt_resize = opt.cwt_resize
    cwt_height =  opt.cwt_height
    cwt_width = opt.cwt_width
        
    # importing dataset
    if data_type == 'signal':
        save_dir = f'{data_type}-{fs}-{upsampling_factor}-{opt.sig_scaling}'
        config = {'data_type': data_type, 'fs': fs, 'upsampling_factor': upsampling_factor, 'sig_scaling': opt.sig_scaling, 'save_dir': save_dir, 'base_dir': base_dir}
    elif data_type == 'stft':
        save_dir = f'{data_type}-{fs}-{upsampling_factor}-{opt.sig_scaling}-{opt.stft_scaling}-{stretch}-{stretch_size}'
        config = {'data_type': data_type, 'fs': fs, 'upsampling_factor': upsampling_factor, 'sig_scaling': opt.sig_scaling, 'stft_scaling': opt.stft_scaling, 'stretch': stretch, 'stretch_size': stretch_size, 'save_dir': save_dir, 'base_dir': base_dir}
    elif data_type == 'cwt':
        save_dir = f'{data_type}-{fs}-{upsampling_factor}-{opt.sig_scaling}-{opt.cwt_scaling}-{wavelet_type}-{width_min}-{width_max}-{width_inc}-{cwt_resize}-{cwt_height}-{cwt_width}'
        config = {'data_type': data_type, 'fs': fs, 'upsampling_factor': upsampling_factor, 'sig_scaling': opt.sig_scaling, 'cwt_scaling': opt.cwt_scaling,'width_min': width_min, 'width_max': width_max, 'width_inc': width_inc, 'wavelet_type': wavelet_type, 'cwt_resize': cwt_resize, 'cwt_height': cwt_height, 'cwt_width': cwt_width, 'save_dir': save_dir, 'base_dir': base_dir}
    config['target_label'] = target_label
    return config

def main():
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    df_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input/physionet.org/files/ptb-xl/1.0.1')

    df = pd.read_csv(os.path.join(df_dir, 'ptbxl_database.csv'))
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x)) # df['scp_codes']里面每一行对应的是一个小字典，ast.literal_eval(x)保证这个字典不是str
    df['label'] = df['scp_codes'].apply(lambda x: set(x.keys())) # 把上面每一行的小字典里面的 键 取唯一。这里是在df这个数据结构里面的最后一列新增了一列，用于存储df.scp_codes 里面的键值
    df_scp = pd.read_csv(os.path.join(df_dir, 'scp_statements.csv'))
    df_scp.index = df_scp['Unnamed: 0'].values # df_scp['Unnamed: 0'] 是一个小字典，键是行序列号（是 0 到...）。值是（0 到...）（就是这个）
    df_scp = df_scp.iloc[:,1:] # 去掉的 第一列是（0 到...）
    df_scp['diagnostic'] = [idx_code if val == 1 else None for idx_code, val in zip(df_scp.index, df_scp['diagnostic'].values)] # 'diagnostic'这一列中值为1对应的行序列号。 生成的是[0,...,33(粗略估计)]
    df_scp['all'] = df_scp.index

    config = get_input_config()
    config_dir = os.path.join(config['base_dir'], config['save_dir']) #用于配置文件(.json)的路径
    dataset = get_dataset(config, df, df_scp, df_dir)
    
    print('Configurations:')
    print(config)
    print(f'Files will be saved in {config_dir}')
    
    save_dir = os.path.join(config_dir, 'data') #用于存储数据（后来会变成.npy格式）的路径
    os.makedirs(config_dir)
    os.makedirs(save_dir)
    json.dump(config, open(os.path.join(config_dir, 'config.json'), 'w'))
    
    pbar = tqdm.tqdm(total = len(dataset), position = 0)  #进度条。total = len(dataset)指一共有多少个患者
    for d, fname in zip(dataset, dataset.files):   # dataset是所有患者的数据，dataset.files 是所有患者的数据的 路径的集合
        #eg:
        #a = [1,2,3]
        #b = [4,5,6]
        #zipped = zip(a,b) 
        #输出：
        #[(1, 4), (2, 5), (3, 6)]
        data = d[0] # dataset 有sig,还有lable.所以 d[0]指sig，d[1]指lable。因为这里的循环用zip打包了，所以这里的data指的是单个患者数据
        fname = os.path.basename(fname)
        
        #path = '/home/User/Documents/file.txt'
        #basename = os.path.basename(path)
        #print(basename)
        #得到path = 'file.txt'
        
        np.save(os.path.join(save_dir, f'{fname}.npy'), data)
        pbar.update(1)    # pbar.update(1)用来更新进度，这里的1指的是完成了一个患者的训练，让进度条加1。
    pbar.close()
    print(f'Files are saved in {config_dir}')
    
if __name__ == '__main__':
    main()
