import pandas as pd
import numpy as np
import pandas_datareader as pdr


'''Derive Spread without DOM data -- (works with historical, unlike DOM which is live only)'''

def getBeta(series,sl):
    hl=series[['High','Low']].values
    hl=np.log(hl[:,0]/hl[:,1])**2
    hl=pd.Series(hl,index=series.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()


def getGamma(series):
    #h2= pandas.stats.moments.rolling_max(series['High'],window=2)
    h2 = series.High.rolling(window=2).max()
    #l2=pd.stats.moments.rolling_min(series['Low'],window=2) #Deprecated
    l2 = series.Low.rolling(window=2).min()
    gamma=np.log(h2.values/l2.values)**2
    gamma=pd.Series(gamma,index=h2.index)
    return gamma.dropna()

def getAlpha(beta,gamma):
    den = 3 -2*2**.5
    alpha = (2**.5-1)*(beta**.5)/den
    alpha -= (gamma/den)**.5
    alpha[alpha<0] = 0 #Set neg alphas to 0
    return alpha.dropna()

def corwinSchultz(series,sl=1):
    # Note: S<0 iif alpha<0
    beta=getBeta(series,sl)
    gamma=getGamma(series)
    alpha=getAlpha(beta,gamma)
    spread=2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    startTime=pd.Series(series.index[0:spread.shape[0]],index=spread.index)
    spread=pd.concat([spread,startTime],axis=1)
    spread.columns=['Spread','Start_Time'] # 1st loc used to compute beta
    return spread

def calcSpread(ticker,type='mean',top=50):
    import pandas_datareader as pdr
    sec = pdr.DataReader(ticker,data_source='yahoo',start='2019-01-01')
    sp = corwinSchultz(sec)
    sprd = sp.sort_values(by='Spread',ascending=False)
    if type == 'mean': return ticker,sprd.Spread.head(top).mean()
    if type == 'median': return sprd.Spread.head(top).median()
    if type == 'max': return sp.Spread.max()
    return 0




############################################################

'''Individual Spread tests (single instrument at a time)'''
import datetime
from ib_insync import *


df = ticker

def get_symbol_ticks(ticker,start = ''):
    end = datetime.datetime.now()
    ticks = ib.reqHistoricalTicks(ticker,start,end,1000,'BID_ASK',useRth=True)
    df = util.df(ticks)
    return df


def get_z_spread(df,stdevs):
    if ('ask','bid') not in df.columns:
        try:
            df = df.rename(columns = {'High':'ask','Low':'bid'})
            #df = df.rename(columns = {'Ask':'ask','Bid':'bid'})
        except Exception:
            print('Error -- check column names')
    df['spread'] = df['ask'] - df['bid']
    df['spread_20'] = df['spread'].rolling(window=20).mean()
    df['sigma_20'] = df['spread'].rolling(window=20).std()
    df['sprd_up'] = df['spread_20'] + (df['sigma_20'] * stdevs)
    df['sprd_dn'] = df['spread_20'] - (df['sigma_20'] * stdevs)
    df['zscore'] = (df['spread'] - df.spread_20) / df.sigma_20
    df['z_spread'] = np.where((df.zscore > stdevs),'Hi',np.where((df.zscore < -stdevs),'Lo','Avg'))

    ZSpread = df['z_spread'].iloc[-1]
    ZScore = df['zscore'].iloc[-1]
    return ZSpread,  ZScore




'''Run ZSpread ZScores on MULTIPLE instruments.'''

def get_symbol_tickers(tickers,start=''):
    '''Confirm append works ! (may need concat)'''
    DF = pd.DataFrame()
    end = datetime.datetime.now()
    for tick in tickers:
        temp = ib.reqHistoricalTicks(tick,start,end,1000,'BID_ASK',useRth=False)
        df = util.df(temp)
        df['symbol'] = tick
        DF.append(df)
        print(f'{tick} added to DF')
    return DF


def get_z_spread_multi(tickers,stdevs):
    #df.set_index(['symbol','date']) #Maybe groupby() cleaner?
    ZSpreads = {}
    ZScores = {}
    for tick in tickers:
        #df = temp_ticks(tick) -- to test w.out IB.
        df = get_symbol_ticks(tick)
        zsp, zsc = get_z_spread(df,stdevs)
        ZSpreads[tick] = zsp
        ZScores[tick] = zsc
    return ZSpreads, ZScores






################################


'''Added ATR quickly -- TAlib may be cleaner'''

def get_z_spread_slip(df,stdevs,denom):
    if ('ask','bid') not in df.columns:
        try:
            df = df.rename(columns = {'High':'ask','Low':'bid'})
            #df = df.rename(columns = {'Ask':'ask','Bid':'bid'}) #Change col names to correct case if needed
        except Exception:
            print('Error -- check column names')
    df['spread'] = df['ask'] - df['bid']
    df['spread_20'] = df['spread'].rolling(window=20).mean()
    df['sigma_20'] = df['spread'].rolling(window=20).std()
    df['sprd_up'] = df['spread_20'] + (df['sigma_20'] * stdevs)
    df['sprd_dn'] = df['spread_20'] - (df['sigma_20'] * stdevs)
    df['zscore'] = (df['spread'] - df.spread_20) / df.sigma_20
    df['z_spread'] = np.where((df.zscore > stdevs),'Hi',np.where((df.zscore < -stdevs),'Lo','Avg'))

    ZSpread = df['z_spread'].iloc[-1]
    ZScore = df['zscore'].iloc[-1]

    if ('High','Low') not in df.columns:
        df = df.rename(columns={'ask':'High','bid':'Low'})
    rng = df['High'] - df['Low']
    df['ATR'] = rng.rolling(window=14).mean()
    df['z_slip'] = (df['ATR'] + df['spread'])/denom
    ZSlip = df['z_slip'].iloc[-1]

    return ZSlip, ZSpread,  ZScore






def get_z_slip_multi(tickers,stdevs=1,denom=1.5):
    ZSlips = {}
    Z = {}
    for tick in tickers:
        df = get_symbol_ticks(tick)
        #df = temp_ticks(tick) -- Used YF initially in testing... not needed when IB works.
        zsl, zsp, zsc = get_z_spread_slip(df,stdevs,denom)
        ZSlips[tick] = zsl
        Z[tick] = [zsl, zsp, zsc]
    return Z




if __name__ == "__main__":

    calcSpread('tsla')

    #%timeit -n10 corwinSchultz(g) #50us


    ############################################################

    get_z_spread(df,1)

    #print(df)
    #ZSP = {}
    #for ticker in tickers:
        #df.loc[ticker,'zscore'] = df['z_spread'] = np.where((df.zscore > 1),'Hi',np.where((df.zscore < -1),'Lo','Avg'))

    #get_symbol_tickers(['aapl','amzn'])

    get_z_spread_multi(['AAPL','AMZN'],1)

    get_z_spread_slip(df,1,2)

    get_z_slip_multi(['AAPL','AMZN','TSLA'],2,2)
