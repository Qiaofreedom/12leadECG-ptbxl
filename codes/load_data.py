import os, glob, ast, wfdb, cv2
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MultiLabelBinarizer

import torch

import matplotlib.pyplot as plt

invalid_file_name = set(['12722'])

# helper normalization functions
def standard_normalization(sig):
    if len(sig.shape) > 1:
        return (sig - sig.mean(axis = (1,2))[:,None,None]) / sig.std(axis = (1,2))[:,None,None]
    else:
        return (sig - sig.mean()) / sig.std()

def minmax_normalization(sig):
    if len(sig.shape) > 1:
        return (sig - sig.min(axis = (1,2))[:,None,None]) / (sig.max(axis = (1,2)) - sig.min(axis = (1,2)))[:,None,None]
    else:
        return (sig - sig.min()) / (sig.max() - sig.min())

def both_normalization(sig):
    sig = standard_normalization(sig)
    sig = minmax_normalization(sig)
    return sig

class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df,
                 df_scp,
                 base_dir, 
                 target_label = 'diagnostic', 
                 fs = 100,
                 upsampling_factor = 1,
                 sig_scaling = None
                ):
        assert target_label in ['diagnostic', 'form', 'rhythm', 'diagnostic_class', 'diagnostic_subclass', 'all'], f"target_label should be one of (diagnostic, form, rhythm, diagnostic_class, diagnostic_subclass, all), but given target_label is {target_label}"
        assert fs in [100, 500], f"sampling rate should be one of (100, 500), but the given fs is {fs}"
        
        if fs == 100:
            self.target_col = 'filename_lr'
        elif fs == 500:
            self.target_col = 'filename_hr'
        df = df[df[self.target_col].apply(lambda x: x.split('/')[-1].split('_')[0] not in invalid_file_name)]  # 按照行的序号提取不同患者的数据（每个患者12个列） 组成集合
        self.files = [os.path.join(base_dir, fname) for fname in df[self.target_col].values]  # 所有患者的数据的 路径的集合 
        self.upsampling_factor = upsampling_factor
        self.fs = fs * upsampling_factor
        
        # generating labels
        dict_label = df_scp[df_scp[target_label].isnull() == False][target_label].to_dict() # 转化成 {index(是 NDT,NST_等) : 
      
        # 这里target_label指的是 'diagnostic'
        # data（df_scp）数据类型为DataFrame结构, 一般来说，to_dict()输出为 data_dict[key1][key2]；data_dict 为数据名；key1 为列属性的键值（即 'diagnostic'）；key2 为内层字典对应的键值；生成一个字典
        # 这里data_dict[key1].to_dict()(也就是dict_label)得到一个小字典。
        #{'NDT': 'STTC',
        # 'NST_': 'NST_',
        # 'DIG': 'STTC',
        # 'LNGQT': 'STTC',
        # 'NORM': 'NORM'}
              
        # df_scp = pd.read_csv(os.path.join(df_dir, 'scp_statements.csv'))  NDT,NST_等是行索引
        # df_scp.index = df_scp['Unnamed: 0'].values # df_scp['Unnamed: 0'] 是一个小字典，键是行序列号（是 0 到...）。值是（0 到...）（就是这个）
        # df_scp = df_scp.iloc[:,1:] # 去掉的 是 df_scp['Unnamed: 0'] 这一列.第一列是 NDT,NST_等。。	
        # df_scp['diagnostic'] = [idx_code if val == 1 else None for idx_code, val in zip(df_scp.index, df_scp['diagnostic'].values)] # 'diagnostic'这一列中 值 为1对应的行序列号。 生成的是[0,...,20(粗略估计)]
        # df_scp['all'] = df_scp.index
        
               
        labels = df['label'].apply(lambda x: [dict_label.get(i) for i in x if i in dict_label]).values  
        #Python 字典(Dictionary) get() 函数返回指定键的值。这里的i 就是（键）NDT,NST_等，dict_label这个字典的键指NDT,NST_等。.get(i)就是得到对应键的值。df['label']得到一个集合
        # 说的是在 df 这个字典的键'label'中存进 dict_label 字典中每个键对应的 值 组成的列表 [1（'diagnostic'对应）]。
        # df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x)) # df['scp_codes']里面每一行对应的是一个小字典，ast.literal_eval(x)保证这个字典不是str
        # df['label'] = df['scp_codes'].apply(lambda x: set(x.keys())) # 把上面每一行的小字典里面的 键 取唯一 作为 集合 返回。{a,v,b}。返回的是一个集合
        # .values 
        # 这个指令结果没有.values 时的结果
        #ecg_id
        #1        [NORM]
        #2        [NORM]
        #3        [NORM]
        #4        [NORM]
        #5        [NORM]
        #  ...  
        #Name: label, Length: 21837, dtype: object
  
        # 这个指令结果有.values 时的结果
        #array([list(['NORM']), list(['NORM']), list(['NORM']), ..., list([]),
        #list(['NORM']), list(['NORM'])], dtype=object)
        
        #df['label']结果
        #ecg_id
        #1               {SR, NORM, LVOLT}
        #2                   {SBRAD, NORM}
        #3                      {SR, NORM}
        #4                      {SR, NORM}
        
        
        self.mlb = MultiLabelBinarizer().fit(labels) # .fit()指 用labels 这个数据来 拟合 MultiLabelBinarizer() 这个模型。
        self.labels = labels
        self.labels_encoded = self.mlb.transform(labels)
        self.sig_scaling = sig_scaling
        
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        file = self.files[idx] # 单个患者
        label = self.labels_encoded[idx]
        sig = self.load_sig(file)
        return sig, label.astype(np.float32)  # torch.utils.data.Dataset构造的这个迭代器，让这个__getitem__重复len(self.files)次。最后返回的是所有患者的数据。
    def load_sig(self, file):
        sig, meta = wfdb.rdsamp(file) # 单个患者。wfdb.rdsamp 专门数据库读取数据，meta指头文件。如果是12导联，则数据的shape是[n * 12].n指这个患者数据一共有n个采样点
        if self.upsampling_factor != 1:
            sig = scipy.signal.resample(sig, int(len(sig) * self.upsampling_factor))
        sig = sig.T  # shape是[12 * n]
        if self.sig_scaling is not None:
            for i in range(len(sig)):
                sig[i] = self.sig_scaling(sig[i])
        return sig.astype(np.float32)
        
class STFTDataset(SignalDataset):
    def __init__(self, 
                 df, 
                 df_scp,
                 base_dir, 
                 target_label = 'diagnostic', 
                 fs = 500,
                 upsampling_factor = 1,
                 nperseg = 128,
                 t_len = 500,
                 stretch = False,
                 stretch_size = 500,
                 sig_scaling = None,
                 scaling = None
                ):
        super().__init__(df, df_scp, base_dir, target_label = target_label, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling)
        self.nperseg = nperseg * upsampling_factor
        self.t_len = t_len
        self.noverlap = int(self.nperseg - ((self.fs * 10) / self.t_len))
        self.scaling = scaling
    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels_encoded[idx]
        stft = self.get_stft(file)
        return stft, label.astype(np.float32)
    def get_stft(self, file):
        sig = self.load_sig(file)  # shape是[12 * n]
        output = []
        for idx, s in enumerate(sig):
            f,t,Zxx = scipy.signal.stft(s, fs = self.fs, nperseg = self.nperseg, noverlap = self.noverlap, return_onesided = True)
            Zxx = abs(Zxx) + 1e-10
            Zxx = np.log(Zxx ** 2)
            output.append(Zxx)
        output = np.array(output).astype(np.float32) # shape是[12 * n]
        if self.scaling is not None:
            output = self.scaling(output)
        return output
    
class STFTDataset_Image(STFTDataset):
    def __init__(self, 
                 df, 
                 df_scp,
                 base_dir, 
                 target_label = 'diagnostic', 
                 fs = 500,
                 upsampling_factor = 1,
                 nperseg = 128,
                 t_len = 500,
                 sig_scaling = None,
                 scaling = None
                ):
        super().__init__(df, df_scp, base_dir, target_label = target_label, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling, nperseg = nperseg, t_len = t_len, scaling = scaling)
    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels_encoded[idx]
        stft = self.get_stft(file) # shape是[12 * n]
        output = []
        for s in stft:
            fig = plt.figure()
            plt.pcolormesh(s, cmap = 'jet')
            plt.axis('off')
            plt.tight_layout(pad = 0)
            fig.canvas.draw()
            output.append(np.array(fig.canvas.renderer._renderer)[:,:,:3].transpose(2,0,1))
            plt.close()
        output = np.concatenate(output) / 255.
        
        #a=np.array([1,2,3])
        #b=np.array([11,22,33])
        #c=np.array([44,55,66])
        #np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写
        #array([ 1,  2,  3, 11, 22, 33, 44, 55, 66]) #对于一维数组拼接，axis的值不影响最后的结果
        
        return output.astype(np.float32), label.astype(np.float32)
    
class CWTDataset(SignalDataset):
    def __init__(self,
                 df,
                 df_scp,
                 base_dir,
                 target_label = 'diagnostic',
                 fs = 100,
                 upsampling_factor = 1,
                 widths = np.arange(0.1,30.1,0.1),
                 wavelet = scipy.signal.ricker,
                 length = None,
                 start = None,
                 end = None,
                 sig_scaling = None,
                 scaling = None,
                 cwt_resize = False,
                 cwt_height = 300,
                 cwt_width = 1000
                ):
        super().__init__(df, df_scp, base_dir, target_label = target_label, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling)
        self.length = length if length is not None else fs * 10 * upsampling_factor
        self.start = start if start is not None else 0
        self.end = end if end is not None else self.start + self.length
        self.resize_shape = None if cwt_resize == False else (cwt_width, cwt_height)
        self.widths = widths
        self.wavelet = wavelet
        self.scaling = scaling
    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels_encoded[idx]
        cwt = self.get_cwt(file)
        return cwt, label.astype(np.float32)
    def get_cwt(self, file):
        sig = self.load_sig(file)[:,self.start: self.end]
        output = np.empty((12, len(self.widths), len(sig[0])), dtype = np.float32)
        for idx, s in enumerate(sig):
            cwt = scipy.signal.cwt(data = s, wavelet = self.wavelet, widths = self.widths, dtype = 'float')
            output[idx] = cwt
        if self.resize_shape is not None:
            output = cv2.resize(output.transpose(1,2,0), self.resize_shape).transpose(2,0,1)
        if self.scaling is not None:
            output = self.scaling(output)
        return output

def get_dataset(config, df, df_scp, base_dir): # 这个要用到
    if isinstance(config, dict) == False:
        config = vars(config)
  
#举例：class Geeks:
    #def __init__(self, name1 = "Arun", num2 = 46, name3 = "Rishab"):
        #self.name1 = name1 
        #self.num2 = num2 
        #self.name3 = name3 
  #GeeksforGeeks = Geeks() 
  #print(vars(GeeksforGeeks))
  #输出：{'num2':46, 'name1':'Arun', 'name3':'Rishab'}

    # base configurations
    target_label = config['target_label']
    data_type = config['data_type']
    fs = config['fs']
    upsampling_factor = config['upsampling_factor']
    sig_scaling = config['sig_scaling']
    # Scaling configs - signal
    if sig_scaling == 'standard':
        sig_scaling = standard_normalization
    elif sig_scaling == 'minmax':
        sig_scaling = minmax_normalization
    elif sig_scaling == 'both':
        sig_scaling = both_normalization
        
    # make dataset
    if data_type == 'signal':
        dataset = SignalDataset(df = df, df_scp = df_scp, base_dir = base_dir, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling, target_label = target_label)
    elif data_type == 'stft':
        # stft configurations
        stft_scaling = config['stft_scaling']
        # Scaling configs - stft
        if stft_scaling == 'standard':
            stft_scaling = standard_normalization
        elif stft_scaling == 'minmax':
            stft_scaling = minmax_normalization
        elif stft_scaling == 'both':
            stft_scaling = both_normalization
        dataset = STFTDataset(df = df, df_scp = df_scp, base_dir = base_dir, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling, scaling = stft_scaling, target_label = target_label)
    elif data_type == 'stft_image':
        # stft configurations
        stft_scaling = config['stft_scaling']
        # Scaling configs - stft
        if stft_scaling == 'standard':
            stft_scaling = standard_normalization
        elif stft_scaling == 'minmax':
            stft_scaling = minmax_normalization
        elif stft_scaling == 'both':
            stft_scaling = both_normalization
        dataset = STFTDataset_Image(df = df, df_scp = df_scp, base_dir = base_dir, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling, scaling = stft_scaling, target_label = target_label)
    elif data_type == 'cwt':
        # cwt configurations
        cwt_scaling = config['cwt_scaling']
        width_min = config['width_min']
        width_max = config['width_max']
        width_inc = config['width_inc']
        widths = np.arange(width_min, width_max, width_inc)
        wavelet_type = config['wavelet_type']
        cwt_resize = config['cwt_resize']
        cwt_height = config['cwt_height']
        cwt_width = config['cwt_width']
        # Scaling configs - cwt
        if cwt_scaling == 'standard':
            cwt_scaling = standard_normalization
        elif cwt_scaling == 'minmax':
            cwt_scaling = minmax_normalization
        elif cwt_scaling == 'both':
            cwt_scaling = both_normalization
        # wavelet config - cwt
        if wavelet_type == 'ricker':
            wavelet = scipy.signal.ricker
        elif wavelet_type == 'morlet2':
            wavelet = scipy.signal.morlet2
        dataset = CWTDataset(df = df, df_scp = df_scp, base_dir = base_dir, fs = fs, upsampling_factor = upsampling_factor, sig_scaling = sig_scaling, scaling = cwt_scaling, widths = widths, wavelet = wavelet, cwt_resize = cwt_resize, cwt_height = cwt_height, cwt_width = cwt_width, target_label = target_label)
    return dataset
