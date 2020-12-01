import numpy as np
import pandas as pd
import math


def Newey_West(ret, q = 2, tao = 252):
    """
    Newey_West方差调整
    时序上存在相关性时，使用Newey_West调整协方差估计
    factor_ret: DataFrame, 行为时间，列为因子收益
    q: 假设因子收益为q阶MA过程
    tao: 算协方差时的半衰期
    """
    from functools import reduce
    from statsmodels.stats.weightstats import DescrStatsW 
    
    T = ret.shape[0]           # 时序长度
    K = ret.shape[1]           # 因子数
    if T <= q or T <= K:
        raise Exception("T <= q or T <= K")
         
    names = ret.columns    
    weights = 0.5**(np.arange(T-1, -1, -1)/tao)   # 指数衰减权重
    weights = weights / sum(weights)
    
    w_stats = DescrStatsW(ret, weights)
    ret = ret - w_stats.mean
    
    ret = np.matrix(ret.values)
    Gamma0 = [weights[t] * ret[t].T  @ ret[t] for t in range(T)]
    Gamma0 = reduce(np.add, Gamma0)

    V = Gamma0             # 调整后的协方差矩阵
    for i in range(1,q+1):
        Gammai = [weights[i+t] * ret[t].T  @ ret[i+t] for t in range(T-i)]
        Gammai = reduce(np.add, Gammai)
        V = V + (1 - i/(1+q)) * (Gammai + Gammai.T)

    return(pd.DataFrame(V, columns = names, index = names))


def progressbar(cur, total, txt):

    """显示进度条"""

    percent = '{:.2%}'.format(cur / total)
    print("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent) + txt, end='')
