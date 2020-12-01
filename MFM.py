import pandas as pd
import numpy as np
from Cross_Regression import Cross_Regression
from utils import Newey_West, progressbar



class MFM():
    '''
    data: DataFrame
    column1: date
    colunm2: stocknames
    colunm3: capital
    column4: ret
    style_factors: DataFrame
    industry_factors: DataFrame
    '''
    
    def __init__(self, data, P, Q):
        self.Q = Q                                                           #风格因子数
        self.P = P                                                           #行业因子数
        self.dates = pd.to_datetime(data.date.values)                        #日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   #排序后的日期
        self.T = len(self.sorted_dates)                                      #期数
        self.data = data                                                     #数据
        self.columns = ['country']                                           #因子名
        self.columns.extend((list(data.columns[4:])))
        
        self.last_capital = None                                             #最后一期的市值 
        self.factor_ret = None                                               #因子收益
        self.specific_ret = None                                             #特异性收益
        self.R2 = None                                                       #R2
        
        self.Newey_West_cov = None                        #逐时间点进行Newey West调整后的因子协方差矩阵
        self.eigen_risk_adj_cov = None                    #逐时间点进行Eigenfactor Risk调整后的因子协方差矩阵
        self.vol_regime_adj_cov = None                    #逐时间点进行Volatility Regime调整后的因子协方差矩阵

    def reg_by_time(self):
        '''
        逐时间点进行横截面多因子回归
        '''
        factor_ret = []
        R2 = []
        specific_ret = []
        
        print('===================================逐时间点进行横截面多因子回归===================================')       
        for t in range(self.T):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            cs = Cross_Regression(data_by_time.iloc[:,:4], data_by_time.iloc[:,-self.Q:], data_by_time.iloc[:,4:(4+self.P)])
            factor_ret_t, specific_ret_t, _ , R2_t = cs.reg()
            
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
            R2.append(R2_t)
            self.last_capital = cs.capital
         
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        
        self.factor_ret = factor_ret                                               #因子收益
        self.specific_ret = specific_ret                                           #特异性收益
        self.R2 = R2                                                               #R2
        return((factor_ret, specific_ret, R2))

    def Newey_West_by_time(self, q = 2, tao = 252):
        '''
        逐时间点计算协方差并进行Newey West调整
        q: 假设因子收益为q阶MA过程
        tao: 算协方差时的半衰期
        '''
        
        if self.factor_ret is None:
            raise Exception('please run reg_by_time to get factor returns first')
            
        Newey_West_cov = []
        print('\n\n===================================逐时间点进行Newey West调整=================================')    
        for t in range(1,self.T+1):
            try:
                Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao))
            except:
                Newey_West_cov.append(pd.DataFrame())
            
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.Newey_West_cov = Newey_West_cov
        return(Newey_West_cov)
    


