import numpy as np
import pandas as pd


def style_factor_norm(factors, capital):
    """
    使用市值进行标准化
    """
    from statsmodels.stats.weightstats import DescrStatsW
    w_stats = DescrStatsW(factors, weights = capital)
    w_mu = w_stats.mean                   # 加权平均
    w_std = np.std(factors)        # 等权标准差
    return((factors - w_mu) / w_std)


class Cross_Regression():
    """
    base_data: DataFrame
    column1: date
    colunm2: stocknames
    colunm3: capital
    column4: ret
    style_factors: DataFrame
    industry_factors: DataFrame
    """
    
    def __init__(self, base_data, style_factors=pd.DataFrame(), industry_factors=pd.DataFrame()):
        self.date = list(base_data.date)[0]                   # 日期
        self.stocknames = list(base_data.stocknames)          # 股票名
        self.capital = base_data.capital.values               # 市值
        self.ret = base_data.ret.values                       # t+1期收益率
        self.style_factors_names = list(style_factors.columns)         # 风格因子名
        self.industry_factors_names = list(industry_factors.columns)   # 行业因子名
        
        self.N = base_data.shape[0]                                                 # 股票数
        self.Q = style_factors.shape[1]                                             # 风格因子数
        self.P = industry_factors.shape[1]                                          # 行业因子数
        self.style_factors = style_factor_norm(style_factors.values, self.capital)  # 风格因子值
        self.industry_factors = industry_factors.values                             # 行业因子值
        self.country_factors = np.array(self.N * [[1]])                             # 国家因子
        
        self.W = np.sqrt(self.capital) / sum(np.sqrt(self.capital))   # 加权最小二乘法的权重
        
        print('\rCross Section Regression, ' + 'Date: ' + self.date  + ', ' + \
              str(self.N) + ' Stocks, ' + str(self.P) + ' Industry Facotrs, ' +  str(self.Q) + ' Style Facotrs', end = '')
    

    
    def reg(self):
        """
        多因子模型求解
        """
        
        W = np.diag(self.W)
        
        if self.P>0:
            # 各个行业的总市值
            industry_capital = np.array([sum(self.industry_factors[:,i] * self.capital) for i in range(self.P)])
            
            # 为处理行业共线性而引入的行业中性限制对应的约束矩阵R
            R = np.eye(1 + self.P + self.Q)    
            R[self.P, 1:(1+self.P)] = -industry_capital / industry_capital[-1]
            R = np.delete(R, self.P, axis =1)
            
            # 求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))  
            factors_tran = factors @ R
            pure_factor_portfolio_weight = R @ np.linalg.inv(factors_tran.T @ W @ factors_tran) @ factors_tran.T @ W  #纯因子组合权重
            
            """
            国家因子：第一行表示国家因子纯因子组合在各个因子上的暴露，
                     国家因子纯因子组合在国家因子上暴露为1，在风格因子上暴露也为0
                     但是在行业因子上不为0（暴露恰好为行业市值权重）
                     而在行业中性限制下国家因子纯因子组合在行业因子暴露上获得的收益恰好为0
                     从而国家因子纯因子组合的收益就是国家因子的收益 f_c
            
            行业因子：第2行-第2+P行表示行业纯因子组合在各个因子上的暴露
                     行业因子纯因子组合在国家因子上暴露为0，在风格因子上暴露也为0
                     但是在各个行业因子上不为0
                     这里所谓的行业纯因子组合是指 做多行业纯因子，同时做空国家纯因子组合
                     为了得到真正的行业纯因子，应该是第一行+第2行，获得的收益是国家因子收益+该行业的纯因子收益
                     这里算出来的行业因子纯因子组合的收益应该理解为行业因子的收益与国家因子的相对收益
            
            风格因子：在风格因子上暴露为1，其他因子上暴露为0，收益为风格纯因子的收益
            """
            
        else:
            # 求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.style_factors]))
            pure_factor_portfolio_weight = np.linalg.inv(factors.T @ W @ factors) @ factors.T @ W    # 纯因子组合权重

        
        factor_ret = pure_factor_portfolio_weight @ self.ret                        # 纯因子收益
        factor_ret = np.array(factor_ret)[0]
        
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     # 纯因子组合在各个因子上的暴露
        specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]               # 个股特异性收益
        R2 = 1 - np.var(specific_ret) / np.var(self.ret)                            # R square
        
        return((factor_ret, specific_ret, pure_factor_portfolio_exposure, R2))

