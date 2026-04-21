# 库索引
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
#import camb
#from camb import model, initialpower
from scipy.integrate import quad

#import mpmath as mp
import astropy.units as u
from astropy.cosmology import z_at_value
# import pickle
from matplotlib.ticker import FuncFormatter # 图表格式化库

from typing import Tuple # 类型提示库
import pandas as pd # 使用pandas读取信息

# from gapp import dgp # GP拟合方法库

import csv
from scipy.interpolate import interp1d

from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.types import TimeSeries,FrequencySeries, Array

# 多线程运算库
from multiprocessing import Pool

# 进程展示
from tqdm import tqdm

import sys
import argparse

# from pathos.multiprocessing import ProcessingPool as Pool

# 定义全局常量
np.random.seed(2030)  # 固定随机数种子
# 光速
c_light=const.c.to(u.km/u.s).value  # 定义天文学中的光速
c_mpc_yr=const.c.to(u.Mpc/u.yr) # 将光速转化为Mpc/year 的情况
c_kms=const.c.to(u.km/u.s).value

# 万有引力常数
constG=const.G.to(u.Mpc**3/(u.Msun*u.s**2)).value # 是万有引力常数
constc=const.c.to(u.Mpc/u.s).value # 转换换单位为 Mpc³ / (Msun·s²)，方便天体物理和宇宙学单位下使用
 
#ACDM标准宇宙学模型=cosmo内储存了一些常量+cosmo可以计算很多宇宙学（依据其所储存的常量）量
cosmo=FlatLambdaCDM(H0=67.716454,Om0=0.31035540,Tcmb0=2.7255, m_nu = [0.06, 0.0, 0.0] * u.eV, Neff=3.046)

# RT修正引力理论参数
RT_Xi = 1.67
RT_n = 1.94

# print(constG)
# print(cosmo.luminosity_distance(1))
# print(cosmo.age(0))
# print(cosmo.hubble_time)
# print(cosmo.H(0).to(u.Gyr**(-1)))
# print(z_at_value(cosmo.luminosity_distance,40*u.Mpc))

# 读取CSV文件并输出为NumPy二维数组
# 输入：CSV文件路径 -> 输出：原始数据的NumPy二维数组data_array[][]
def read_csv_to_numpy(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path)  # 文件路径可以是相对路径或绝对路径
    data_array = df.to_numpy()
    return data_array # 这里返回的是一个二维Numpy数组

# 绘制二维数组[][2]数据直方图
# 输入：(（csv文件地址）二维数组data_array[..][3], 图例标签，点线样式，误差条颜色) -> 输出：无（图像）
# def plot_histogram_from_csv(file_path: str):
def plot_histogram_from_csv(file_path, z_min=None, z_max=None, bins=10):
    data = read_csv_to_numpy(file_path)  # 替换成你的文件路径

    z = data[:, 0]  # 假设 z 在第一列
    cnt = data[:, 1]  # 假设 cnt 在第二列

    # 自动设置 z 范围
    if z_min is None:
        z_min = z.min()
    if z_max is None:
        z_max = z.max()

    # 生成 bin 边界
    bin_edges = np.linspace(z_min, z_max, bins + 1)

    # 统计每个 bin 的总计数
    hist, _ = np.histogram(z, bins=bin_edges, weights=cnt)

    # 绘制直方图
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10,6))
    plt.bar(bin_centers, hist, width=width, color='skyblue', edgecolor='black')
    plt.xlim(0,0.5)
    plt.ylim(bottom=0)
    plt.xlabel('z')
    plt.ylabel('Count')
    plt.title(f'Histogram of counts vs z (bins={bins})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 绘制直方图，按 z 的区间统计，cnt 用作权重
    # plt.figure(figsize=(10,6))

    # # 这里用 bins=10，你可以根据需要调整区间数
    # plt.hist(z, bins=10, weights=cnt, color='skyblue', edgecolor='black')

    # plt.xlabel('z')
    # plt.ylabel('Count')
    # plt.title('Histogram of counts vs z')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    # 输入： (二维红移数组（csv文件）,输出文件名称，总抽取样本数) -> 输出：无（抽样红移分布数据csv表）
def stratified_sampling_from_histogram(csv_file, output_file, N_total, z_min=None, z_max=None, bins=10, seed=None):
    """
    从直方图 CSV 文件进行加权分层抽样
    csv_file: 输入 CSV 文件，包含 'z' 和 'cnt' 两列，z 为条形中值
    output_file: 输出 CSV 文件
    N_total: 总共要抽取的样本数
    z_min, z_max: 可选，定义直方图 z 范围
    bins: 分箱数（可选）
    seed: 随机种子（可选）
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 读取 CSV
    df = pd.read_csv(csv_file)
    z = df['z'].values
    cnt = df['cnt'].values

    # 设置 z 范围
    if z_min is None:
        z_min = z.min()
    if z_max is None:
        z_max = z.max()

    # 生成 bin 边界
    bin_edges = np.linspace(z_min, z_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    # 按 bin 中心找到每个 z 属于哪一个 bin
    bin_indices = np.digitize(z, bin_edges) - 1
    bin_indices[bin_indices == bins] = bins - 1  # 边界处理

    # 计算每个 bin 的总权重
    bin_weights = np.zeros(bins)
    for i, w in zip(bin_indices, cnt):
        bin_weights[i] += w

    # 根据权重分配样本数（四舍五入）
    bin_sample_counts = np.round(N_total * bin_weights / bin_weights.sum()).astype(int)

    # 在每个 bin 内随机生成样本
    samples = []
    for i, n_samples in enumerate(bin_sample_counts):
        if n_samples > 0:
            # 均匀分布在 bin 区间
            s = np.random.uniform(bin_edges[i], bin_edges[i+1], n_samples)
            samples.append(s)
    samples = np.concatenate(samples)

    # 保存到 CSV
    pd.DataFrame({'z_sampled': samples}).to_csv(output_file, index=False)
    print(f"生成 {len(samples)} 个样本，已保存到 {output_file}")

# 本地合并率密度 常量定义。三个数是你的模拟设置开关。它们决定了你的虚拟宇宙中到底“有多挤”
rm_f0=662 # 中子星多，但信号弱。
rm_g0=920 # 一个非常乐观的参数，用来产生大量数据以减小统计误差。
rm_bh0=50 # 黑洞少，但信号强。

#  不考虑时间延迟的情形 no-time-delay case 
def Rm0(z):
    if 0<=z<1:
        return 1+2*z
    if 1<=z<5:
        return 3/4*(5-z)
    else:
        return 0

def norm_Rm0(z):
    return rm_g0/Rm0(0)*Rm0(z)

#print(norm_Rm0(0))

v_norm_Rm0=np.vectorize(norm_Rm0)


def Rz0(z):
    return 4*np.pi*(cosmo.comoving_distance(z).to(u.Gpc).value)**2*c_light/(cosmo.H(z).to(u.km / (u.Gpc*u.s)).value*(1+z))*norm_Rm0(z)


##################################################################################################################################################
##################################################################################################################################################
# 考虑time-delay的情形 time-delay case  ：幂律延迟 power-law td^(-1)
## 来自于论文 1409.2462 Eq.1
def Rf(z):
    nu=0.36
    a=1.92
    b=1.5
    zm=2.6
    return nu*(a*np.exp(b*(z-zm)))/(a-b+b*np.exp(a*(z-zm)))


def Rm(z):
    rm=quad(lambda zf: 1/((cosmo.H(zf).value)*(1+zf))*Rf(zf)/(cosmo.lookback_time(zf).value-cosmo.lookback_time(z).value),
        z_at_value(cosmo.lookback_time, (cosmo.lookback_time(z).value+0.02) * u.Gyr, 0, 100), 200)[0]
    return rm


def norm_Rm(z):
    return rm_g0/Rm(0)*Rm(z)

#print(norm_Rm(0))

v_norm_Rm=np.vectorize(norm_Rm)

def Rz(z):
    rz=4*np.pi*(cosmo.comoving_distance(z).to(u.Gpc).value)**2*c_light/(cosmo.H(z).to(u.km / (u.Gpc*u.s)).value*(1+z))*norm_Rm(z)
    return rz

##################################################################################################################################################
##################################################################################################################################################
# 考虑time-delay的情形 time-delay case  ：指数衰减  exponential time-dela   1/tau*exp((tf-tm)/tau)
## 来自于论文 1403.0007 Eq.15
def psi_MD(z):
    a=2.7
    b=5.6
    c=2.9
    phi0=0.015
    return phi0*(1+z)**a/(1+((1+z)/c)**b)

##  来自论文1808.00901 Eq.2 and 5
def Rm_MD(z):
    tau=0.1
    return quad(lambda zf: 1/((cosmo.H(zf).value)*(1+zf))*psi_MD(zf)/
        tau*np.exp(-(cosmo.lookback_time(zf).value-cosmo.lookback_time(z).value)/tau),z, np.inf)[0]

def norm_Rm_MD(z):
    return rm_g0/Rm_MD(0)*Rm_MD(z)
# print(norm_Rm_MD(0))

v_norm_Rm_MD=np.vectorize(norm_Rm_MD)

def Rz_MD(z):
    rz=4*np.pi*(cosmo.comoving_distance(z).to(u.Gpc).value)**2*c_light/(cosmo.H(z).to(u.km / (u.Gpc*u.s)).value*(1+z))*norm_Rm_MD(z)
    return rz

# A=Rz_MD(0.1)
# print(A)

# 根据红移分布函数，计算红移小区间内“模拟观测到”的引力波事件数量。
# 输入：红移分布函数（Rz0(z)或Rz(z)或Rz_MD(z)）, 区间步长，区间数量 -> 输出：一维numpy整数数组（每个红移小区间内“模拟观测到”的引力波事件数量）
# def generate_zn_samples(z_distribution_func,step=0.01,n_range=20):
#     znlist0=[quad(z_distribution_func,step*i,step*(i+1))[0] for i in range(n_range)] # 积分计算每个红移区间的“理论期望数” 带小数
#     znlist1=[int(znlist0[i]) for i in range(len(znlist0))] # 对理论值进行向下取整
#     znlist1=np.array(znlist1)
#     znlist2=np.zeros(len(znlist0)) # 决定那些“小数点后的数字”是否能变成一个真实的信号：这是一种伯松抽样（Poisson Sampling）的简化实现
#     for i in range(len(znlist2)):
#         a=np.random.random()
#         if znlist0[i]-znlist1[i]>a:
#             znlist2[i]=znlist1[i]+1 # 向上取整
#         else:
#             znlist2[i]=znlist1[i] # 保持向下取整
#     znlist=np.array([int(znlist2[i]) for i in range(len(znlist2))]) # 将结果汇总并转回整数类型。
#     return znlist
def generate_zn_samples(z_distribution_func, step=0.01, n_range=20, z_start=0.0):
    """
    计算从 z_start 开始，分 n_range 个步长区间，每个区间的源数量
    """
    # 修改点：quad 的积分下限变为 z_start + step*i，上限变为 z_start + step*(i+1)
    znlist0 = [quad(z_distribution_func, z_start + step*i, z_start + step*(i+1))[0] for i in range(n_range)]
    
    znlist1 = np.array([int(x) for x in znlist0])
    znlist2 = np.zeros(len(znlist0))
    
    for i in range(len(znlist2)):
        a = np.random.random()
        # 概率抽样：处理小数点部分
        if znlist0[i] - znlist1[i] > a:
            znlist2[i] = znlist1[i] + 1
        else:
            znlist2[i] = znlist1[i]
            
    return znlist2.astype(int)
# znlist = generate_zn_samples(Rz_MD,step=0.01,n_range=20)
# print(znlist)

# 将之前generate_zn_samples算出的“每个区间有几个源”的数量，转化为每个源具体的红移坐标。
# 输入：红移分布函数, 区间步长，区间数量 -> 输出：一维numpy数组（每个元素是一个引力波事件的红移）
# def generate_z_samples(z_distribution_func,step=0.01,n_range=20):
#     znlist=generate_zn_samples(z_distribution_func,step,n_range)
#     zlist=np.array([])
#     for i, n in enumerate(znlist):
#         z=np.random.uniform(step*i,step*(i+1),n)
#         zlist=np.concatenate((zlist,z),axis=None)
#     zlist.sort()
#     print(zlist,np.sum(znlist),len(zlist))
#     return zlist
def generate_z_samples(z_distribution_func, step=0.01, n_range=20, z_start=0.0):
    """
    将区间数量转化为具体的红移坐标，支持起始点 z_start
    """
    znlist = generate_zn_samples(z_distribution_func, step, n_range, z_start)
    zlist = np.array([])
    
    for i, n in enumerate(znlist):
        # 修改点：均匀分布采样范围同样需要加上 z_start 偏移
        low = z_start + step * i
        high = z_start + step * (i + 1)
        z = np.random.uniform(low, high, n)
        zlist = np.concatenate((zlist, z), axis=None)
        
    zlist.sort()
    print(f"Z range: [{zlist.min() if len(zlist)>0 else z_start:.3f} - {zlist.max() if len(zlist)>0 else z_start:.3f}]")
    print(f"Total events: {len(zlist)}")
    return zlist

# zlist = generate_z_samples(Rz_MD,step=0.01,n_range=5)
# print(zlist)


# 根据生成的红移样本坐标zlist，去组成更完整的模拟引力波事件数据
# 输入：使用引力模型类型，生成的红移坐标 ->输出：二维numpy数组，每行是一个模拟引力波事件的参数（Mc, eta, dL, iota, theta, phi, psi, tc, phic）
def generate_gw_events(zlist,GW_type="GR"):
    m1=np.random.uniform(1,2,len(zlist)) #Msun

    m2=np.random.uniform(1,2,len(zlist))#Msun
    cosiota=np.random.uniform(0,1,len(zlist))  #http://keatonb.github.io/archivers/uniforminclination
    iota=np.arccos(cosiota)
    theta=np.random.uniform(0,np.pi,len(zlist))
    # cosphi=np.random.uniform(-1,1,len(zlist)) #https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    # phi=np.arccos(cosphi)
    phi = np.random.uniform(0,2*np.pi,len(zlist))
    
    for i in range(len(zlist)):
        s=np.random.random()
        if s>0.5:
            phi[i]=2*np.pi-phi[i]
    psi=np.random.uniform(0,np.pi*2,len(zlist))
    Mc=(m1*m2)**(3/5)/(m1+m2)**(1/5)*(1+zlist) #Msun
    eta = (m1 * m2) / (m1 + m2)**2
    # Maggiore GW I page 175
    # fmax=[2*constc**3/(6*np.sqrt(6)*2*np.pi*constG*(m1[i]+m2[i]))/(1+zlist[i]) for i in range(len(zlist))]
    # 1907.01487 Eq.2.8
    # F2_L=(1+cosiota**2)**2/4*(np.vectorize(Fp_L)(theta,phi,psi))**2+cosiota**2*(np.vectorize(Fc_L)(theta,phi,psi))**2
    #F2_T=(1+cosiota**2)**2/4*(np.vectorize(Fp_T)(theta,phi,psi))**2+coseta**2*(np.vectorize(Fc_T)(theta,phi,psi))**2
    if GW_type=="GR":
        dL=cosmo.luminosity_distance(zlist).value
    elif GW_type == "RT":
        dL=cosmo.luminosity_distance(zlist).value*(RT_Xi + (1 - RT_Xi)/((1 + zlist)**RT_n)) # RT模型的luminosity distance需要额外乘以一个(1+z)的因子
    else:
        dL=cosmo.luminosity_distance(zlist).value

    tc = np.zeros(len(zlist))    # 并合时刻令为 0
    phic = np.zeros(len(zlist))  # 并合相位通常也令为 0
    generate_gw_events = np.column_stack((zlist,Mc,eta,dL, iota, theta, phi, psi,tc,phic))

    return generate_gw_events

# gw_events = generate_gw_events(generate_z_samples(Rz_MD,step=0.01,n_range=20),GW_type="GR")
# print(gw_events)

def Sn_ETCE(f,detector="ET"):
    if detector == "ET":
        psd_ET=np.loadtxt('psd_ET.txt')
        def interp_psd_ET(f):
            return np.interp(f,psd_ET[:,0],psd_ET[:,3])
        return interp_psd_ET(f)**2
    elif detector == "CE":
        psd_CE=np.loadtxt('psd_CE.txt')
        def interp_psd_CE(f):
            return np.interp(f,psd_CE[:,0],psd_CE[:,3])
        return interp_psd_CE(f)**2
    else:
        raise ValueError(f"Unknown detector type: {detector}")
    
def bar(i, n):
    i = i + 1
    percent = i / n * 100
    print(
        '\r[' +
        '#' * int(40 * i / n) +
        ' ' * (40 - int(40 * i / n)) +
        f'] {percent:6.2f}%',
        end=''
    )

# time of frequency
def tf(f,Mc,tc):    #the circular case
    return tc-5*(8*np.pi*f)**(-8/3)*(constG*Mc)**(-5/3)*constc**5

#fsun=1/((1*u.yr).to(u.s).value)
fearth=1/((1*u.day).to(u.s).value)
def phiEA_t(t,phiEA0):
    return 2*np.pi*fearth*t+phiEA0

def get_detector_response2(detector="ET",type = None):
    if detector == "ET":
        if type == "1":
            def Fp_ET1 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET1Fp0=(-0.10478204441849033 + np.cos(2*(phi - phiEA))*(-0.45771795558150985 - 0.15257265186050328*np.cos(2*theta)) + np.cos(2*theta)*(0.10478204441849033 + 0.08574865294897005*np.sin(2*(phi - phiEA))) + 0.25724595884691015*np.sin(2*(phi - phiEA)) + (-0.18130721949311185*np.cos(phi - phiEA) + 0.13215019511808676*np.sin(phi - phiEA))*np.sin(2*theta))
                ET1Fc0=(-0.3437619782425746*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.6113445953704968*np.cos(theta)*np.sin(2*(phi - phiEA)) + (-0.26330154256052474*np.cos(phi - phiEA) - 0.3620522365797096*np.sin(phi - phiEA))*np.sin(theta))
                return ET1Fp0*np.cos(2*psi)-ET1Fc0*np.sin(2*psi)
            def Fc_ET1 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET1Fp0=(-0.10478204441849033 + np.cos(2*(phi - phiEA))*(-0.45771795558150985 - 0.15257265186050328*np.cos(2*theta)) + np.cos(2*theta)*(0.10478204441849033 + 0.08574865294897005*np.sin(2*(phi - phiEA))) + 0.25724595884691015*np.sin(2*(phi - phiEA)) + (-0.18130721949311185*np.cos(phi - phiEA) + 0.13215019511808676*np.sin(phi - phiEA))*np.sin(2*theta))
                ET1Fc0=(-0.3437619782425746*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.6113445953704968*np.cos(theta)*np.sin(2*(phi - phiEA)) + (-0.26330154256052474*np.cos(phi - phiEA) - 0.3620522365797096*np.sin(phi - phiEA))*np.sin(theta))
                return ET1Fp0*np.sin(2*psi)+ET1Fc0*np.cos(2*psi)
            return {"Fp": Fp_ET1, "Fc": Fc_ET1}
        
        elif type == "2":
            def Fp_ET2 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET2Fp0=(-0.10399155347212757 + np.cos(2*(phi - phiEA))*(-0.4585084465278725 - 0.1528361488426241*np.cos(2*theta)) + np.cos(2*theta)*(0.10399155347212757 - 0.08594049456064361*np.sin(2*(phi - phiEA))) - 0.2578214836819308*np.sin(2*(phi - phiEA)) + (-0.1810261182898547*np.cos(phi - phiEA) - 0.13165077128026226*np.sin(phi - phiEA))*np.sin(2*theta))
                ET2Fc0=(0.34376197824257443*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.6113445953704965*np.cos(theta)*np.sin(2*(phi - phiEA)) + (0.2633015425605245*np.cos(phi - phiEA) - 0.3620522365797093*np.sin(phi - phiEA))*np.sin(theta))
                return ET2Fp0*np.cos(2*psi)-ET2Fc0*np.sin(2*psi)
            def Fc_ET2 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET2Fp0=(-0.10399155347212757 + np.cos(2*(phi - phiEA))*(-0.4585084465278725 - 0.1528361488426241*np.cos(2*theta)) + np.cos(2*theta)*(0.10399155347212757 - 0.08594049456064361*np.sin(2*(phi - phiEA))) - 0.2578214836819308*np.sin(2*(phi - phiEA)) + (-0.1810261182898547*np.cos(phi - phiEA) - 0.13165077128026226*np.sin(phi - phiEA))*np.sin(2*theta))
                ET2Fc0=(0.34376197824257443*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.6113445953704965*np.cos(theta)*np.sin(2*(phi - phiEA)) + (0.2633015425605245*np.cos(phi - phiEA) - 0.3620522365797093*np.sin(phi - phiEA))*np.sin(theta))
                return ET2Fp0*np.sin(2*psi)+ET2Fc0*np.cos(2*psi)
            return {"Fp": Fp_ET2, "Fc": Fc_ET2}
        
        elif type == "3":
            def Fp_ET3 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET3Fp0=(-0.5156429673638618*np.sin(2*(phi - phiEA)) - 0.1718809891212873*np.cos(2*theta)*np.sin(2*(phi - phiEA)) - 0.2633015425605246*np.sin(phi - phiEA)*np.sin(2*theta))
                ET3Fc0=(0.6875239564851492*np.cos(2*(phi - phiEA))*np.cos(theta) + 0.5266030851210493*np.cos(phi - phiEA)*np.sin(theta))
                return ET3Fp0*np.cos(2*psi)-ET3Fc0*np.sin(2*psi)
            def Fc_ET3 (f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,(4+55/60)*np.pi/180)
                ET3Fp0=(-0.5156429673638618*np.sin(2*(phi - phiEA)) - 0.1718809891212873*np.cos(2*theta)*np.sin(2*(phi - phiEA)) - 0.2633015425605246*np.sin(phi - phiEA)*np.sin(2*theta))
                ET3Fc0=(0.6875239564851492*np.cos(2*(phi - phiEA))*np.cos(theta) + 0.5266030851210493*np.cos(phi - phiEA)*np.sin(theta))
                return ET3Fp0*np.sin(2*psi)+ET3Fc0*np.cos(2*psi)
            return {"Fp": Fp_ET3, "Fc": Fc_ET3}
        
        else:
            raise ValueError(f"Unknown ET type: {type}")
    

    elif detector == "CE":
        if type == "1":
            def Fp_CE1(f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,-(119+24/60+28/3600)*np.pi/180)
                CE1Fp0=(0.0549986361215859 + np.cos(2*(phi - phiEA))*(0.1767641096596249 + 0.05892136988654163*np.cos(2*theta)) + np.cos(2*theta)*(-0.05499863612158591 + 0.1723402035292836*np.sin(2*(phi - phiEA))) + 0.5170206105878509*np.sin(2*(phi - phiEA)) + (0.07715459212251842*np.cos(phi - phiEA) + 0.3276011924380142*np.sin(phi - phiEA))*np.sin(2*theta))
                CE1Fc0=(-0.6893608141171343*np.cos(2*(phi - phiEA))*np.cos(theta) + 0.23568547954616653*np.cos(theta)*np.sin(2*(phi - phiEA)) + (-0.6552023848760282*np.cos(phi - phiEA) + 0.15430918424503684*np.sin(phi - phiEA))*np.sin(theta))
                return CE1Fp0*np.cos(2*psi)-CE1Fc0*np.sin(2*psi)
            def Fc_CE1(f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,-(119+24/60+28/3600)*np.pi/180)
                CE1Fp0=(0.0549986361215859 + np.cos(2*(phi - phiEA))*(0.1767641096596249 + 0.05892136988654163*np.cos(2*theta)) + np.cos(2*theta)*(-0.05499863612158591 + 0.1723402035292836*np.sin(2*(phi - phiEA))) + 0.5170206105878509*np.sin(2*(phi - phiEA)) + (0.07715459212251842*np.cos(phi - phiEA) + 0.3276011924380142*np.sin(phi - phiEA))*np.sin(2*theta))
                CE1Fc0=(-0.6893608141171343*np.cos(2*(phi - phiEA))*np.cos(theta) + 0.23568547954616653*np.cos(theta)*np.sin(2*(phi - phiEA)) + (-0.6552023848760282*np.cos(phi - phiEA) + 0.15430918424503684*np.sin(phi - phiEA))*np.sin(theta))
                return CE1Fp0*np.sin(2*psi)+CE1Fc0*np.cos(2*psi)
            return {"Fp": Fp_CE1, "Fc": Fc_CE1}
        
        elif type == "2":
            def Fp_CE2(f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,-(90+46/60+27/3600)*np.pi/180)
                CE2Fp0=(-0.2249408831041809 + np.cos(2*(phi - phiEA))*(-0.3818218626770296 - 0.12727395422567658*np.cos(2*theta)) + np.cos(2*theta)*(0.2249408831041809 - 0.07471957383007324*np.sin(2*(phi - phiEA))) - 0.22415872149021976*np.sin(2*(phi - phiEA)) + (-0.177109999936927*np.cos(phi - phiEA) - 0.2530628713378165*np.sin(phi - phiEA))*np.sin(2*theta))
                CE2Fc0=(0.2988782953202931*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.5090958169027063*np.cos(theta)*np.sin(2*(phi - phiEA)) + (0.5061257426756333*np.cos(phi - phiEA) - 0.3542199998738541*np.sin(phi - phiEA))*np.sin(theta))
                return CE2Fp0*np.cos(2*psi)-CE2Fc0*np.sin(2*psi)
            def Fc_CE2(f,Mc,tc,theta,phi,psi):
                t=tf(f,Mc,tc)
                phiEA=phiEA_t(t,-(90+46/60+27/3600)*np.pi/180)
                CE2Fp0=(-0.2249408831041809 + np.cos(2*(phi - phiEA))*(-0.3818218626770296 - 0.12727395422567658*np.cos(2*theta)) + np.cos(2*theta)*(0.2249408831041809 - 0.07471957383007324*np.sin(2*(phi - phiEA))) - 0.22415872149021976*np.sin(2*(phi - phiEA)) + (-0.177109999936927*np.cos(phi - phiEA) - 0.2530628713378165*np.sin(phi - phiEA))*np.sin(2*theta))
                CE2Fc0=(0.2988782953202931*np.cos(2*(phi - phiEA))*np.cos(theta) - 0.5090958169027063*np.cos(theta)*np.sin(2*(phi - phiEA)) + (0.5061257426756333*np.cos(phi - phiEA) - 0.3542199998738541*np.sin(phi - phiEA))*np.sin(theta))
                return CE2Fp0*np.sin(2*psi)+CE2Fc0*np.cos(2*psi)
            return {"Fp": Fp_CE2, "Fc": Fc_CE2}
        
#时间延迟推导，引力波到达太阳和
# 主要输入: 物理量参数+探测器名字 -> 输出:返回意味数组
L=2.5*10**9. #meter
R=(1*u.AU).to(u.m).value
eob=L/(2*np.sqrt(3)*R)
rEA=const.R_earth.value

def delay0(f,Mc,tc,theta,phi,detector="ET",type=None):
    if detector == "ET":
        t=tf(f,Mc,tc)
        phiEA0=(4+55/60)*np.pi/180
        phiEA=phiEA_t(t,phiEA0)
        dl=rEA*(-0.7921121258037702*np.cos(theta) - 0.6103756057991113*np.cos(phi - phiEA)*np.sin(theta))
        return dl/const.c.value
    

    elif detector == "CE":
        if type == "1":
            t=tf(f,Mc,tc)
            phiEA0=-(119+24/60+28/3600)*np.pi/180
            phiEA=phiEA_t(t,phiEA0)
            dl=rEA*(-0.7248368549143048*np.cos(theta) - 0.6889205569279662*np.cos(phi - phiEA)*np.sin(theta))
            return dl/const.c.value
        
        elif type == "2":
            t=tf(f,Mc,tc)
            phiEA0=-(90+46/60+27/3600)*np.pi/180
            phiEA=phiEA_t(t,phiEA0)
            dl=rEA*(-0.5084821270261745*np.cos(theta) - 0.8610725442696088*np.cos(phi - phiEA)*np.sin(theta))
            return dl/const.c.value
    # elif detector == "LISA":
    #     resp = get_detector_response(detector)
    #     t=resp["tf"](tc,f,Mc)
    #     alpha=resp["alphat"](t)
    #     dl=-R*np.cos(alpha-phi)*np.sin(theta)+eob*(np.sqrt(3)*R*np.cos(alpha)*np.cos(theta)-1/2*R*((np.cos(2*alpha-phi)-3*np.cos(phi))*np.sin(theta)))
    #     return dl/const.c.value
    
    else :
        raise ValueError(f"Unknown detector type: {detector}")

fstep=0.01
nstart=500
# fmax=65

def hf_ETCE(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector="ET",type=None,model_type="IMRPhenomXHM"):
  def hf_ETCE_IMRPhenomXHM(Mc,eta,dL,iota,theta,phi,psi,tc,phic,detector="ET",type=None):
      #flist=flist0
      m1=(Mc*eta**(2/5)+np.sqrt((1-4*eta)*eta**(4/5)*Mc**2))/(2*eta)
      m2=(Mc*eta**(2/5)-np.sqrt((1-4*eta)*eta**(4/5)*Mc**2))/(2*eta)
      apx='IMRPhenomXHM'
      hpl, hcl = get_fd_waveform(approximant=apx,
                                #  hptilde,
                                #  ctilde,
                                mass1=m1,
                                mass2=m2,
                                # 自旋参数 (Spins) 
                                ##  chi1L,
                                ##  chi2L,
                                #  eccentricity=e0,
                                f_ref=0.,
                                f_lower=1,
                                f_final=constc**3/(6*np.sqrt(6)*np.pi*constG*(Mc/eta**(3/5))),
                                # f_final=65,                                
                                delta_f=fstep,
                                distance=dL,
                                inclination=iota,
                                coa_phase=phic,
                                )
      flist=np.array(hpl.sample_frequencies[nstart:])
      hp=np.array(hpl[nstart:])
      hc=np.array(hcl[nstart:])
      # 代入不同的 响应函数/时间延迟函数 计算最终的频域波形
      resp = get_detector_response2(detector,type=type)
      Fp = resp["Fp"]
      Fc = resp["Fc"]
      h0=hp*Fp(flist,Mc,tc,theta,phi,psi)+hc*Fc(flist,Mc,tc,theta,phi,psi)
      t0=tc+delay0(flist,Mc,tc,theta,phi,detector=detector,type=type)
      h=h0*np.exp(-1j*2*np.pi*flist*t0) 
      return flist,h
  def hf_ETCE_EccentricFD(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector="ET",type=None):
      #flist=flist0
      m1=(Mc*eta**(2/5)+np.sqrt((1-4*eta)*eta**(4/5)*Mc**2))/(2*eta)
      m2=(Mc*eta**(2/5)-np.sqrt((1-4*eta)*eta**(4/5)*Mc**2))/(2*eta)
      apx='EccentricFD'
      hpl, hcl = get_fd_waveform(approximant=apx,
  #                                        mchirp=Mc,
  #                                        eta=eta,
                                          mass1=m1,
                                          mass2=m2,
  #                                        lambda1=0,
  #                                        lambda2=0,
                                          delta_f=fstep,
                                          eccentricity=e0,
                                          distance=dL,
                                          inclination=iota,
                                          coa_phase=phic,
                                          long_asc_nodes=beta,
                                          mean_per_ano=ellmode,
                                          f_ref=0.,
                                          f_lower=5,
                                          f_final=constc**3/(6*np.sqrt(6)*np.pi*constG*(Mc/eta**(3/5))),
                                          #sample_points=Array(flist)
                                )
      flist=np.array(hpl.sample_frequencies[nstart:])
      hp=np.array(hpl[nstart:])
      hc=np.array(hcl[nstart:])
      # 代入不同的 响应函数/时间延迟函数 计算最终的频域波形
      resp = get_detector_response2(detector,type=type)
      Fp = resp["Fp"]
      Fc = resp["Fc"]
      h0=hp*Fp(flist*(2/ellmode),Mc,tc,theta,phi,psi)+hc*Fc(flist*(2/ellmode),Mc,tc,theta,phi,psi)
      t0=tc+delay0(flist*(2/ellmode),Mc,tc,theta,phi,detector=detector,type=type)
      h=h0*np.exp(-1j*2*np.pi*flist*t0) 
      return flist,h
  # 这里的逻辑：根据输入的 model_type 返回对应函数的计算结果
  if model_type == "IMRPhenomXHM":
      return hf_ETCE_IMRPhenomXHM(Mc,eta,dL,iota,theta,phi,psi,tc,phic,detector=detector,type=type)
  elif model_type == "EccentricFD":
      return hf_ETCE_EccentricFD(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=ellmode,detector=detector,type=type)
  else:
      raise ValueError("Unknown model type specified.")
  
# (Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector="ET",type=None,model_type="IMRPhenomXHM")
# (Mc,eta,dL,iota,theta,phi,psi,tc,phic,detector="ET",type=None)
# (Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector="ET",type=None)
# l2 waveform
def hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
  flist=hf_ETCE(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector=detector,type=type,model_type=model_type)[0]
  hlist=hf_ETCE(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,ellmode=2,detector=detector,type=type,model_type=model_type)[1]
  return flist,hlist

# 计算l2形信噪比SNR
def rho2_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    flist=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector=detector,type=type,model_type=model_type)[0]
    hlist=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector=detector,type=type,model_type=model_type)[1]
    rho2i0=4*abs(hlist)**2/Sn_ETCE(flist,detector)
    rho2i1=rho2i0[:-1]
    rho2i2=rho2i0[1:]
    rho2i=(rho2i1+rho2i2)/2*(flist[1]-flist[0])
    return np.sum(rho2i)

# 输入：物理量参数列表 -> 输出: SNR筛选后的GW事件信息(z,snr,Mc,eta,dL,iota,theta,phi,psi,tc,phic)
# def GW_SNR(gw_events,detector="ET",model_type="IMRPhenomXHM"):
#     print(f"Calculating SNR for {len(gw_events)} GW events in {detector} with model {model_type}...")
#     snr=[]
#     z=gw_events[:,0]
#     Mc=gw_events[:,1]
#     eta=gw_events[:,2]
#     dL=gw_events[:,3]
#     iota=gw_events[:,4]
#     theta=gw_events[:,5]
#     phi=gw_events[:,6]
#     psi=gw_events[:,7]
#     tc = gw_events[:,8]
#     phic = gw_events[:,9]
#     if detector == "ET":
#         for i in range(len(z)):
#             # Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None
#             rho2_ET1=rho2_ETCE_l2(Mc[i],eta[i],dL[i],iota[i],theta[i],phi[i],psi[i],tc[i],phic[i],e0=0,beta=0,detector=detector,type="1",model_type=model_type)
#             rho2_ET2=rho2_ETCE_l2(Mc[i],eta[i],dL[i],iota[i],theta[i],phi[i],psi[i],tc[i],phic[i],e0=0,beta=0,detector=detector,type="2",model_type=model_type)
#             rho2_ET3=rho2_ETCE_l2(Mc[i],eta[i],dL[i],iota[i],theta[i],phi[i],psi[i],tc[i],phic[i],e0=0,beta=0,detector=detector,type="3",model_type=model_type)
#             rho=np.sqrt(rho2_ET1+rho2_ET2+rho2_ET3)
#             bar(i, len(z))
#             snr.append(rho)
#     elif detector == "CE":
#         for i in range(len(z)):
#             rho2_CE1=rho2_ETCE_l2(Mc[i],eta[i],dL[i],iota[i],theta[i],phi[i],psi[i],tc[i],phic[i],e0=0,beta=0,detector=detector,type="1",model_type=model_type)
#             rho2_CE2=rho2_ETCE_l2(Mc[i],eta[i],dL[i],iota[i],theta[i],phi[i],psi[i],tc[i],phic[i],e0=0,beta=0,detector=detector,type="2",model_type=model_type)
#             rho=np.sqrt(rho2_CE1+rho2_CE2)
#             bar(i, len(z))
#             snr.append(rho)
#     GWlist=np.column_stack((z,snr,Mc,eta,dL,iota,theta,phi,psi,tc,phic))
#     GWlist_12=GWlist[GWlist[:,1]>12] #生成包含信噪比的GW事件列表，并且筛选信噪比大于一定值的事件
#     return GWlist_12

def SNR_worker(i, event_data, detector, model_type):
    """
    负责处理第 i 条数据的计算工人
    """
    # 解包数据 (对应你原来的索引)
    Mc, eta, dL, iota, theta, phi, psi, tc, phic = event_data
    
    if detector == "ET":
        r1 = rho2_ETCE_l2(Mc, eta, dL, iota, theta, phi, psi, tc, phic, 0, 0, detector, "1", model_type)
        r2 = rho2_ETCE_l2(Mc, eta, dL, iota, theta, phi, psi, tc, phic, 0, 0, detector, "2", model_type)
        r3 = rho2_ETCE_l2(Mc, eta, dL, iota, theta, phi, psi, tc, phic, 0, 0, detector, "3", model_type)
        rho = np.sqrt(r1 + r2 + r3)
    elif detector == "CE":
        r1 = rho2_ETCE_l2(Mc, eta, dL, iota, theta, phi, psi, tc, phic, 0, 0, detector, "1", model_type)
        r2 = rho2_ETCE_l2(Mc, eta, dL, iota, theta, phi, psi, tc, phic, 0, 0, detector, "2", model_type)
        rho = np.sqrt(r1 + r2)
    else:
        rho = 0
        
    return i, rho  # 返回索引是为了防止并行计算导致顺序错乱

def GW_SNR_parallel(gw_events, detector="ET", model_type="IMRPhenomXHM", n_processes=8):
    print(f"Parallel calculating SNR for {len(gw_events)} events with {n_processes} cores...")
    
    # 1. 准备容器
    num_events = len(gw_events)
    snr_results = [0 for _ in range(num_events)]
    
    # 2. 启动进程池
    pool = Pool(processes=n_processes)
    
    # 3. 提交任务
    # 注意：我们只传递该任务需要的列数据 (Mc 到 phic 是索引 1 到 9)
    recs = []
    for i in range(num_events):
        event_slice = gw_events[i, 1:10] # 提取计算所需的参数
        res = pool.apply_async(SNR_worker, (i, event_slice, detector, model_type))
        recs.append(res)
    
    # 4. 收集结果并显示进度条
    for r in tqdm(recs, desc="Computing SNR"):
        idx, rho = r.get()
        snr_results[idx] = rho
        
    # 5. 善后
    pool.close()
    pool.join()
    
    # 6. 后处理（和你原来的逻辑一致）
    # 拼接 z, snr, 和其他的参数
    # gw_events[:, 0] 是 z, gw_events[:, 1:] 是后面的参数
    GWlist = np.column_stack((gw_events[:, 0], snr_results, gw_events[:, 1:]))
    
    # 筛选 SNR > 12
    GWlist_12 = GWlist[GWlist[:, 1] > 12]
    return GWlist_12

# GW=generate_gw_events(generate_z_samples(Rz_MD,step=0.01,n_range=5))
# GWlist_ET=GW_SNR(GW,detector="ET",model_type="EccentricFD")
# print(len(GW),len(GWlist_ET))

# 探测器的最小可探测光子通量：表示单位时间内、单位面积上接收到的最小光子数。
# （低于这个值的信号将无法被探测到）（被认为是噪音之类的）
# 对应电磁信号探测器的 触发灵敏度
#单位are ph, s, cm, keV

# Fermi-GBM的参数
pflux_min=0.2 #(u.ph/u.s/u.cm**2) in 50-300 keV band for Fermi-GBM

# 探测器的能量响应范围：表示探测器能够有效探测的电磁信号的能量范围。
def E_min(detector="Fermi-GBM"):
    if detector == "Fermi-GBM":
        return 50 # keV
    else:
        raise ValueError(f"Unknown detector type: {detector}")

def E_max(detector="Fermi-GBM"):
    if detector == "Fermi-GBM":
        return 300 # keV
    else:
        raise ValueError(f"Unknown detector type: {detector}")
    
# 定义 GRB 能谱模型 (Band Function) :这是一个分段函数，描述了光子数通量与能量的关系。
def N_GRB(E):
    alpha=-0.5
    beta=-2.25
    Ep=800 #keV
    E0=Ep/(alpha+2)
    Eb=(alpha-beta)*E0
    if E<=Eb:
        N=(E/100)**alpha*np.exp(-E/E0)
    else:
        N=(Eb/100)**(alpha-beta)*np.exp(beta-alpha)*(E/100)**beta
    return N

# k因子修正项计算：表示在考虑红移效应后，探测器能够探测到的 GRB 信号的能量范围与原始能量范围之间的比例关系。
def k_GRB(z,detector="Fermi-GBM"):
    k=quad(N_GRB,E_min(detector=detector),E_max(detector=detector))[0]/quad(N_GRB,E_min(detector=detector)*(1+z),E_max(detector=detector)*(1+z))[0]
    return k

# 能量归一化修正因子计算：表示在考虑 GRB 能谱模型后，探测器能够探测到的 GRB 信号的能量归一化修正因子。
# 计算能段转换因子。将 50-300 keV（探测器能量窗口） 的光子数通量转换为 1-10000 keV 范围内的能量通量。
def C_det_GRB(detector="Fermi-GBM"):
    C = quad(lambda E: E*N_GRB(E), 1, 10000)[0]/quad(N_GRB,E_min(detector=detector),E_max(detector=detector))[0]
    return C

# print(C_det_GRB(detector="Fermi-GBM"))

# GRB 的光度函数（双幂律模型） 描述了宇宙中产生不同亮度（L）GRB 的概率分布
# 因为 phi_grb 本身只是一个比例关系，它的积分值不等于 1。
def phi_grb(L):
    alpha=-1.95 #这里的设定有不同的地方
    beta=-3
    Ls=2*10**52 #(erg s^-1)
    if L < Ls:
        phi=(L/Ls)**alpha
    else:
        phi=(L/Ls)**beta
    return phi

# 生成概率密度函数：norm_phi_grb(L) 
# 对 phi_grb(L) 进行归一化处理后的概率密度函数
# 表示在给定亮度 L 的情况下，产生 GRB 的概率密度。
# 10**49, 10**54 是这类GRB天体物理研究中公认的有效光度区间。
def norm_phi_grb(L):
    norm_grb=quad(phi_grb, 10**49, 10**54)[0]
    return phi_grb(L)/norm_grb

# 给定一个观测阈值 Lmin，计算有多大比例的 GRB 亮度会超过这个阈值，从而被仪器探测到。
def prob(L_min):
    if L_min<=10**49: #在之前的步骤中，GRB光度函数的下限被设定为 10**49 （
        p=1 # 表示探测器的灵敏度极高，甚至能看到比模型中定义的“最暗 GRB”还要暗的信号，那么在这个模型下，所有的 GRB 都必然能被探测到）
    elif L_min>=10**54: 
        p=0 # 表示探测器的阈值非常高，甚至超过了宇宙中可能存在的最亮 GRB 的光度，那么探测器将永远无法捕捉到任何信号，所以概率 p=0。
    else:
        p=1-quad(norm_phi_grb,10**49,L_min)[0] 
        # 得到的就是光度落在 [Lmin,10**54 ] 之间的概率（即“足够亮能被看见”的比例）
    return p

# 采样与条件筛选
# def sample_GRB(i,gw_events,detector="Fermi-GBM"):
#     zlist_t12=gw_events[:,0]
#     iota_t12=gw_events[:,5]
#     L_min0=pflux_min*4*np.pi*(cosmo.luminosity_distance(zlist_t12[i]).to(u.cm).value)**2*k_GRB(z=zlist_t12[i],detector=detector)*C_det_GRB(detector=detector)/(1+zlist_t12[i])
#     L_min1=L_min0/np.exp(-0.5*(iota_t12[i]*u.rad.to(u.deg))**2/4.7**2)
#     L_min=((L_min1*(u.keV/u.s)).to(u.erg/u.s)).value
#     p=prob(L_min)
#     a=np.random.random()
#     aa=np.random.random()
#     # if p>a and aa<4/30:
#     if p>a:
#         y=1
#     else:
#         y=0
#     GRBlisti=np.append(gw_events[i],[p,y])
#     # print(f"Event {i}: iota_deg={iota_t12[i]*u.rad.to(u.deg):.2f}, L_min={L_min:.2e}, p={p:.4f},a={a:.2e},aa={aa:.2e},y={y:.2e}")
#     return i, GRBlisti
# # 多核
# def worker(i,gw_list):
#     return sample_GRB(i, gw_list, detector="Fermi-GBM")

# ==========================================
# 1. Worker 函数 (必须放在全局)
# ==========================================
def GRB_worker(i, single_event, detector):
    # 这里的变量（cosmo, pflux_min 等）需要确保在全局已导入或定义
    z_val = single_event[0]
    iota_val = single_event[5] 
    
    # --- 计算逻辑 ---
    dist_cm = cosmo.luminosity_distance(z_val).to(u.cm).value
    L_min0 = pflux_min * 4 * np.pi * (dist_cm**2) * \
             k_GRB(z=z_val, detector=detector) * \
             C_det_GRB(detector=detector) / (1 + z_val)
    
    iota_deg = iota_val * (180.0 / np.pi) 
    L_min1 = L_min0 / np.exp(-0.5 * (iota_deg)**2 / 4.7**2)
    L_min = ((L_min1 * (u.keV / u.s)).to(u.erg / u.s)).value
    
    p = prob(L_min)

    a=np.random.random()
    aa=np.random.random()
    # y = 1 if p > np.random.random() else 0
    if p>a and aa<4/30:
    # if p>a:
        y=1
    else:
        y=0
    
    # 返回索引 i 用于排序，返回结果用于组合
    return i, np.append(single_event, [p, y])

# ==========================================
# 2. 封装函数 (你想要的简洁调用接口)
# ==========================================
def GW_GRB_parallel(gw_events, detector="Fermi-GBM", n_processes=8):
    print(f"Parallel processing {len(gw_events)} GRB samples on {n_processes} cores...")
    
    num_events = len(gw_events)
    results = [None] * num_events
    
    with Pool(processes=n_processes) as pool:
        # 异步提交任务
        recs = [pool.apply_async(GRB_worker, (i, gw_events[i], detector)) 
                for i in range(num_events)]
        
        # 收集结果
        for r in tqdm(recs, desc="GRB Sampling"):
            idx, data = r.get()
            results[idx] = data
            
    return np.array(results)



#fisher matrix微分，注意不同质量范围的源步长选取可能不同
def ph0(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dMc=Mc*10**-11
    result_plus = hf(Mc+dMc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc-dMc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dMc)

def ph1(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    deta=10**-11  
    result_plus = hf(Mc,eta+deta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta-deta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*deta)

def ph2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    ddL=10**-6
    result_plus = hf(Mc,eta,dL+ddL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL-ddL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*ddL)

def ph3(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    diota=10**-8
    result_plus = hf(Mc,eta,dL,iota+diota,theta,phi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota-diota,theta,phi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*diota)/(-np.sin(iota))

def ph4(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dtheta=10**-9
    result_plus = hf(Mc,eta,dL,iota,theta+dtheta,phi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota,theta-dtheta,phi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dtheta)

def ph5(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dphi=10**-9
    result_plus = hf(Mc,eta,dL,iota,theta,phi+dphi,psi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota,theta,phi-dphi,psi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dphi)

def ph6(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dpsi=10**-9
    result_plus = hf(Mc,eta,dL,iota,theta,phi,psi+dpsi,tc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota,theta,phi,psi-dpsi,tc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dpsi)

def ph7(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dtc=10**-6  
    result_plus = hf(Mc,eta,dL,iota,theta,phi,psi,tc+dtc,phic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota,theta,phi,psi,tc-dtc,phic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dtc)

def ph8(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    dphic=10**-6
    result_plus = hf(Mc,eta,dL,iota,theta,phi,psi,tc,phic+dphic,e0,kappa,lam)
    result_minus = hf(Mc,eta,dL,iota,theta,phi,psi,tc,phic-dphic,e0,kappa,lam)
    dh = result_plus['h'] - result_minus['h']
    return dh/(2*dphic)

# 偏心率部分
# def ph9(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
#     de0=10**-10
#     dh=hf(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0+de0,kappa,lam)[1]-hf(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0-de0,kappa,lam)[1]
#     return dh/(2*de0)

phmat=np.array([ph0,ph1,ph2,ph3,ph4,ph5,ph6,ph7,ph8])

##########################################################################################################################
##########################################################################################################################

# The fisher matrix

def ph0_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dMc=10**-9
    dh=hf_ETCE_l2(Mc+dMc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dMc
    
def ph1_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    deta=10**-9
    dh=hf_ETCE_l2(Mc,eta+deta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/deta

def ph2_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    ddL=10**-6
    dh=hf_ETCE_l2(Mc,eta,dL+ddL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    #return -hf_cbc(Mc,eta,dL+ddL,iota,theta,phi,psi,tc,phic)[1]/dL
    return dh/ddL

def ph3_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    diota=10**-7
    dh=hf_ETCE_l2(Mc,eta,dL,iota+diota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/diota/(-np.sin(iota))

def ph4_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dtheta=10**-9
    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta+dtheta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dtheta

def ph5_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dphi=10**-9
    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi+dphi,psi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dphi

def ph6_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dpsi=10**-9
    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi+dpsi,tc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dpsi

def ph7_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dtc=10**-9
    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc+dtc,phic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dtc

def ph8_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    dphic=10**-5
    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic+dphic,e0,beta,detector,type,model_type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[1]
    return dh/dphic

#def ph9_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None):
#    de0=10**-10
#    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0+de0,beta,detector,type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type)[1]
#    return dh/de0

#def ph10_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None):
#    dbeta=10**-9
#    dh=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type)[1]-hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type)[1]
#    return dh/dbeta

phmat_ETCE_l2=np.array([ph0_ETCE_l2,ph1_ETCE_l2,ph2_ETCE_l2,ph3_ETCE_l2,ph4_ETCE_l2,ph5_ETCE_l2,ph6_ETCE_l2,ph7_ETCE_l2,ph8_ETCE_l2])

#fisher矩阵元计算（通过指定矩阵元的行列指标ij）
# 输出： 结果：返回费雪矩阵的一个矩阵元 Γij
def Gamma(i,j,Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    wave =hf(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam,detector="LISA")
    flist=wave['freq']
    ri=phmat[i](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam).real
    rj=phmat[j](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam).real
    imgi=phmat[i](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam).imag
    imgj=phmat[j](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam).imag
    gammaijbin0=4*(ri*rj+imgi*imgj)/Sn(flist) # 对应引力波物理中的标准内积公式（Fisher矩阵元的计算公式）
    gammaijbin1=gammaijbin0[:-1]
    gammaijbin2=gammaijbin0[1:]
    gammaijbin=(gammaijbin1+gammaijbin2)/2*(flist[1]-flist[0])
    return np.sum(gammaijbin) # 返回第 ij 个矩阵元的数值

def Gamma_ETCE_l2(i,j,Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
    flist=hf_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)[0]
    ri=phmat_ETCE_l2[i](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type).real
    rj=phmat_ETCE_l2[j](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type).real
    imgi=phmat_ETCE_l2[i](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type).imag
    imgj=phmat_ETCE_l2[j](Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type).imag
    gammaijbin0=4*(ri*rj+imgi*imgj)/Sn_ETCE(flist,detector)
    gammaijbin1=gammaijbin0[:-1]
    gammaijbin2=gammaijbin0[1:]
    gammaijbin=(gammaijbin1+gammaijbin2)/2*(flist[1]-flist[0])
    return np.sum(gammaijbin)
# 注意 这个FIsher矩阵是针对Lisa的
# FM 函数（Fisher Matrix）是Gamma函数“总结者”。它的任务是将刚才计算的所有单个 Gamma 矩阵元组装成一个完整的费雪信息矩阵，并考虑了探测器的双通道观测效应。
# 初始化一个 9×9 的全零矩阵（对应 9 个物理参数 M eta d_L等。Fisher矩阵是一个对称矩阵。代码只循环计算了下三角部分
# 输出：一个 Fisher矩阵的二维 NumPy 数组 (ndarray)[][]。
def FM(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam):
    fisher=np.zeros((9,9))
    for j in range(9):
        for k in range(j+1):
            fisher[j,k]=Gamma(j,k,Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam)+Gamma(j,k,Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,kappa,lam+np.pi/4) #这里是LISA等效两个探测器（在lam上相差45度），所以是两个探测器的fisher矩阵之和
    for j in range(9):
        for k in range(j+1,9):
            fisher[j,k]=fisher[k,j]
    return fisher

def FM_ETCE_l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",model_type="EccentricFD"):
    def FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector="ET",type=None,model_type="IMRPhenomXHM"):
        fisher=np.zeros((9,9))
        for j in range(9):
            for k in range(j+1):
                fisher[j,k]=Gamma_ETCE_l2(j,k,Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type,model_type)
        for j in range(9):
            for k in range(j+1,9):
                fisher[j,k]=fisher[k,j]
        return fisher
    if detector == "ET":
        fisher_ET1=FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type="1",model_type=model_type)
        fisher_ET2=FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type="2",model_type=model_type)
        fisher_ET3=FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type="3",model_type=model_type)
        fisher=fisher_ET1+fisher_ET2+fisher_ET3
        return fisher
    elif detector == "CE":
        fisher_CE1=FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type="1",model_type=model_type)
        fisher_CE2=FM_ETCE_0l2(Mc,eta,dL,iota,theta,phi,psi,tc,phic,e0,beta,detector,type="2",model_type=model_type)
        fisher=fisher_CE1+fisher_CE2
        return fisher
    else :
        raise ValueError(f"Unknown detector type: {detector}")

# Fisher信息矩阵先验：Fisher矩阵 描述的是探测器从信号中获取的信息。但有时候，探测器对某些参数不敏感（比如角度）（如果我们不加约束，数学上可能会认为某个角度的误差是正无穷。但实际上我们知道，任何角度（如经纬度）的最大误差不会超过 2π。），导致矩阵中某些行或列接近于 0。
# np.diag函数 创建了一个对角矩阵，对角线上的每一个值对应一个参数的先验权重
fm_p_e0=np.diag([0,0.25**-2,0,1**-2,np.pi**-2,np.pi**-2,np.pi**-2,0,np.pi**-2]) # 这里面一共有9个值（所以对应9X9的对角矩阵）

# 输入 SNR筛选后的GW事件信息(z,snr,Mc,eta,dL,iota,theta,phi,psi,tc,phic)-> 输出："z,dl,ddl"
# def Delta_dl(GW_With_SNR,detector="ET",model_type="EccentricFD"):
#     ddl = []
#     for i in range(len(GW_With_SNR)):
#         fmatrix = FM_ETCE_l2(GW_With_SNR[i,2],GW_With_SNR[i,3],GW_With_SNR[i,4],GW_With_SNR[i,5],GW_With_SNR[i,6],GW_With_SNR[i,7],GW_With_SNR[i,8],GW_With_SNR[i,9],GW_With_SNR[i,10],e0=0,beta=0,detector=detector,model_type=model_type) + fm_p_e0
#         covi = np.linalg.inv(fmatrix)
#         ddl.append(np.sqrt(covi[2,2]))
#         bar(i,len(GW_With_SNR))
#     result_data = np.column_stack((GW_With_SNR[:,0],GW_With_SNR[:,4],ddl))
#     header = "z,dl,ddl"
#     np.savetxt('GW_GRB_results.csv', result_data, delimiter=',', header=header, comments='', fmt='%.6f')
#     return result_data
def Delta_dl_worker(i, event_data, detector, model_type, fm_p_e0):
    """
    单个事件的 Fisher 矩阵计算工人
    event_data 对应 GW_With_SNR[i]
    """
    # 提取参数 (根据你之前的索引: Mc 是 2, eta 是 3, dL 是 4...)
    # 索引：2:Mc, 3:eta, 4:dL, 5:iota, 6:theta, 7:phi, 8:psi, 9:tc, 10:phic
    args = event_data[2:11] 
    
    # 计算 Fisher 矩阵
    # *args 会自动解包这 9 个参数传给 FM_ETCE_l2
    fmatrix = FM_ETCE_l2(*args, e0=0, beta=0, detector=detector, model_type=model_type) + fm_p_e0
    
    try:
        # 计算协方差矩阵（逆矩阵）
        covi = np.linalg.inv(fmatrix)
        # 提取 dL 的不确定度 (对应矩阵中的索引 2,2)
        ddl_val = np.sqrt(covi[2,2])
    except np.linalg.LinAlgError:
        # 如果矩阵奇异无法求逆，填入 NaN 或无穷大
        ddl_val = np.nan
        
    return i, ddl_val
def Delta_dl_parallel(GW_With_SNR, detector="ET", model_type="EccentricFD", n_processes=8):
    """
    多线程 Fisher 矩阵分析函数
    """
    print(f"Parallel Fisher Analysis for {len(GW_With_SNR)} events on {n_processes} cores...")
    
    num_events = len(GW_With_SNR)
    ddl_results = [0 for _ in range(num_events)]
    
    # 预先获取全局变量 fm_p_e0 (假设它已在全局定义)
    with Pool(processes=n_processes) as pool:
        recs = [pool.apply_async(Delta_dl_worker, (i, GW_With_SNR[i], detector, model_type, fm_p_e0)) 
                for i in range(num_events)]
        
        for r in tqdm(recs, desc="Fisher Analysis"):
            idx, val = r.get()
            ddl_results[idx] = val

    # 结果拼接: z (索引0), dL (索引4), ddl
    result_data = np.column_stack((GW_With_SNR[:, 0], GW_With_SNR[:, 4], ddl_results))
    
    # 保存数据
    header = "z,dl,ddl"
    np.savetxt('GW_GRB_results.csv', result_data, delimiter=',', header=header, comments='', fmt='%.6f')
    
    return result_data


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################





# 判断是否有电磁对应体
# 单核
# for i in range(len(GWlist_ET)):
#     # 直接调用函数，它会立即运行并返回结果
#     bar(i,len(GWlist_ET))
#     res = sample_GRB(i, GWlist_ET, "Fermi-GBM")
    
#     # 根据函数返回的 (i, GRBlisti) 进行赋值
#     idx = res[0]
#     data = res[1]
#     GW_grb_list0[idx] = data



def main(z_start, n_range, step=0.01):
    # 生成模拟GW事件
    # 10000个事件（0.01，30）需要 20min
    GW=generate_gw_events(generate_z_samples(Rz_MD,step=step,n_range=n_range,z_start=z_start))

    # 判断GW事件是否能被指定探测器探测到，并计算其信噪比（GW探测器筛选）
    GWlist_ET=GW_SNR_parallel(GW,detector="ET",model_type="IMRPhenomXHM",n_processes=16)
    # print(len(GW),len(GWlist_ET))

    # 创建一个空列表，用于存储每个GW事件是否具有电磁对应体（GRB探测器筛选）的结果。初始值为0，表示默认没有对应体。
    GW_grb_list0=GW_GRB_parallel(GWlist_ET, detector="Fermi-GBM", n_processes=16)

    # 3. 转换为 numpy 数组
    GW_grb_list0 = np.array(GW_grb_list0)

    GW_GRB_list=GW_grb_list0[GW_grb_list0[:,-1]==1]

    print(f"{len(GW)} GW events generated")
    print(f"The GW events pass SNR test: {len(GWlist_ET)}")
    # print(f"GW events with GRB counterparts: {len(GW_GRB_list)}")

    output_name = f'z_{z_start}_to_{n_range*step+z_start}.csv'
    # 结果拼接: z (索引0), dL (索引4), ddl
    result_data = np.column_stack((GW_GRB_list[:, 0], GW_GRB_list[:, 4]))
    # 保存数据
    header = "z,dl"
    np.savetxt(output_name, result_data, delimiter=',', header=header, comments='', fmt='%.6f')

    # 计算每个合格的GW事件的光度距离测量不确定度Delta_dL（Fisher矩阵分析） 得到最终所需要的结果
    # Delta_dl_parallel(GW_GRB_list,detector="ET",model_type="IMRPhenomXHM",n_processes=8)

if __name__ == "__main__":
    # --- 使用 argparse 解析命令行输入 ---
    parser = argparse.ArgumentParser(description="GW-GRB Redshift Interval Simulator")
    
    # 添加参数：起始红移，区间数量
    parser.add_argument('--z_start', type=float, default=0.0, help='Starting redshift')
    parser.add_argument('--n_range', type=int, default=100, help='Number of bins (each bin size is step)')
    parser.add_argument('--step', type=float, default=0.01, help='Step size for each bin')

    args = parser.parse_args()

    # 将解析到的参数传入 main
    main(z_start=args.z_start, n_range=args.n_range, step=args.step)