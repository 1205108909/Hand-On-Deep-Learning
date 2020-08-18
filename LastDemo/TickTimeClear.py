#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : TickTimeClear.py 
@Time : 2020/8/14 17:00 
"""

import os
import h5py
import pandas as pd
import numpy as np
from DataService.JYDataLoader import JYDataLoader
from datetime import datetime
import csv
import warnings
warnings.filterwarnings("ignore")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def get_standand_time():
    tick_time = pd.date_range(f'{tradingDay} 9:30:00', f'{tradingDay} 15:00:00', freq='3s')
    tick_time = tick_time[(tick_time <= datetime.strptime(f'{tradingDay} 11:30:00', '%Y%m%d %H:%M:%S')) | (
            tick_time >= datetime.strptime(f'{tradingDay} 13:00:00', '%Y%m%d %H:%M:%S'))]
    series = pd.Series(tick_time)
    series = series.map(lambda x: x.hour * 10000000 + x.minute * 100000 + x.second * 1000)
    return pd.DataFrame({'Time': series})



def read_tick(f, symbol):
    """
    read tick data
    :param symbol: '600000.sh' str
    :param tradingday: '20170104' str
    :return: pd.DataFrame
    """
    if symbol not in f.keys():
        return None
    print(symbol)
    time = f[symbol]['Time']
    if len(time) == 0:
        print(f'{symbol} tick is null')
        return pd.DataFrame()
    price = f[symbol]['Price']
    volume = f[symbol]['Volume']
    turnover = f[symbol]['Turnover']
    matchItem = f[symbol]['MatchItem']
    bsflag = f[symbol]['BSFlag']
    accVolume = f[symbol]['AccVolume']
    accTurnover = f[symbol]['AccTurnover']
    askAvgPrice = f[symbol]['AskAvgPrice']
    bidAvgPrice = f[symbol]['BidAvgPrice']
    totalAskVolume = f[symbol]['TotalAskVolume']
    totalBidVolume = f[symbol]['TotalBidVolume']
    open_p = f[symbol]['Open']
    high = f[symbol]['High']
    low = f[symbol]['Low']
    preClose = f[symbol]['PreClose']

    tick = pd.DataFrame(
        {'Time': time, 'Price': price, 'Volume': volume, 'Turnover': turnover, 'MatchItem': matchItem,
         'BSFlag': bsflag, 'AccVolume': accVolume, 'AccTurnover': accTurnover, 'AskAvgPrice': askAvgPrice,
         'BidAvgPrice': bidAvgPrice, 'TotalAskVolume': totalAskVolume, 'TotalBidVolume': totalBidVolume,
         'Open': open_p, 'High': high, 'Low': low, 'PreClose': preClose})

    for i in range(10):
        tick['BidPrice' + str(i + 1)] = f[symbol]['BidPrice10'][:][:, i]
        tick['AskPrice' + str(i + 1)] = f[symbol]['AskPrice10'][:][:, i]
        tick['BidVolume' + str(i + 1)] = f[symbol]['BidVolume10'][:][:, i]
        tick['AskVolume' + str(i + 1)] = f[symbol]['AskVolume10'][:][:, i]
    return tick


def update(symbol, tick):
    """
    更新Tick
    :param symbol: 600000.sh
    :param tradingday: '20150101'
    :param tick: dict:
            {'Time', 'Price', 'Volume', 'Turnover', 'MatchItem',
             'BSFlag','AccVolume', 'AccTurnover', 'AskPrice10',
             'AskVolume10','BidPrice10', 'BidVolume10', 'AskAvgPrice',
             'BidAvgPrice','TotalAskVolume', 'TotalBidVolume', 'Open', 'High',
             'Low', 'PreClose'}
    :return:
    """
    h5file = h5py.File(os.path.join(f'{tradingDay}_timeformat_tick.h5'), 'a')

    time = np.array(tick['Time'], dtype=np.uint32)
    price = np.array(tick['Price'], dtype=np.float32)
    volume = np.array(tick['Volume'], dtype=np.int64)
    turnover = np.array(tick['Turnover'], dtype=np.int64)
    matchItem = np.array(tick['MatchItem'], dtype=np.uint32)
    bsflag = np.array(tick['BSFlag'], dtype='S1')
    accVolume = np.array(tick['AccVolume'], dtype=np.int64)
    accTurnover = np.array(tick['AccTurnover'], dtype=np.int64)

    askAvgPrice = np.array(tick['AskAvgPrice'], dtype=np.float32)
    bidAvgPrice = np.array(tick['BidAvgPrice'], dtype=np.float32)
    totalAskVolume = np.array(tick['TotalAskVolume'], dtype=np.int64)
    totalBidVolume = np.array(tick['TotalBidVolume'], dtype=np.int64)
    open_p = np.array(tick['Open'], dtype=np.float32)
    high = np.array(tick['High'], dtype=np.float32)
    low = np.array(tick['Low'], dtype=np.float32)
    preclose = np.array(tick['PreClose'], dtype=np.float32)
    # 科创板
    afterPrice = np.array(tick['AfterPrice'], dtype=np.float32)
    afterVolume = np.array(tick['AfterVolume'], dtype=np.uint64)
    afterTurnover = np.array(tick['AfterTurnover'], dtype=np.uint64)
    afterMatchItems = np.array(tick['AfterMatchItems'], dtype=np.uint32)


    group = h5file.create_group(symbol.lower())
    group.create_dataset('Time', data=time, compression='gzip', compression_opts=9)
    group.create_dataset('Price', data=price, compression='gzip', compression_opts=9)
    group.create_dataset('Volume', data=volume, compression='gzip', compression_opts=9)
    group.create_dataset('Turnover', data=turnover, compression='gzip', compression_opts=9)
    group.create_dataset('MatchItem', data=matchItem, compression='gzip', compression_opts=9)
    group.create_dataset('BSFlag', data=bsflag, compression='gzip', compression_opts=9)
    group.create_dataset('AccVolume', data=accVolume, compression='gzip', compression_opts=9)
    group.create_dataset('AccTurnover', data=accTurnover, compression='gzip', compression_opts=9)

    group.create_dataset('AskAvgPrice', data=askAvgPrice, compression='gzip', compression_opts=9)
    group.create_dataset('BidAvgPrice', data=bidAvgPrice, compression='gzip', compression_opts=9)
    group.create_dataset('TotalAskVolume', data=totalAskVolume, compression='gzip', compression_opts=9)
    group.create_dataset('TotalBidVolume', data=totalBidVolume, compression='gzip', compression_opts=9)
    group.create_dataset('Open', data=open_p, compression='gzip', compression_opts=9)
    group.create_dataset('High', data=high, compression='gzip', compression_opts=9)
    group.create_dataset('Low', data=low, compression='gzip', compression_opts=9)
    group.create_dataset('PreClose', data=preclose, compression='gzip', compression_opts=9)

    group.create_dataset('AfterPrice', data=afterPrice, compression='gzip', compression_opts=9)
    group.create_dataset('AfterVolume', data=afterVolume, compression='gzip', compression_opts=9)
    group.create_dataset('AfterTurnover', data=afterTurnover, compression='gzip', compression_opts=9)
    group.create_dataset('AfterMatchItem', data=afterMatchItems, compression='gzip', compression_opts=9)

    AskPrice = np.array([tick[f'AskPrice{i + 1}'] for i in range(10)], dtype=np.float32)
    AskVolume = np.array([tick[f'AskVolume{i + 1}'] for i in range(10)], dtype=np.uint32)
    BidPrice = np.array([tick[f'BidPrice{i + 1}'] for i in range(10)], dtype=np.float32)
    BidVolume = np.array([tick[f'BidVolume{i + 1}'] for i in range(10)], dtype=np.uint32)

    group.create_dataset('AskPrice10', data=AskPrice.T, compression='gzip', compression_opts=9)
    group.create_dataset('AskVolume10', data=AskVolume.T, compression='gzip', compression_opts=9)
    group.create_dataset('BidPrice10', data=BidPrice.T, compression='gzip', compression_opts=9)
    group.create_dataset('BidVolume10', data=BidVolume.T, compression='gzip', compression_opts=9)

    h5file.close()


start = datetime.now()

tradingDay = '20200813'
# 1.生产标准化时间：
df_standand_time = get_standand_time()

f = h5py.File(os.path.join(tradingDay + '.h5'), 'r')
for symbol in f.keys():
    df_symbol_tick = read_tick(f, symbol)
    if df_symbol_tick.shape[0] == 0:
        continue
    df_symbol_tick1 = df_symbol_tick[df_symbol_tick['Time'] < 93000000]
    df_symbol_tick2 = df_symbol_tick[
        (df_symbol_tick['Time'] >= 93000000) & (df_symbol_tick['Time'] <= 150000000)]
    df_symbol_tick3 = df_symbol_tick[df_symbol_tick['Time'] > 150000000]
    df_symbol_tick2 = pd.merge(df_standand_time, df_symbol_tick2, how='left')

    df_symbol_tick2['AccVolume'].fillna(method='ffill', inplace=True)
    df_symbol_tick2['AccTurnover'].fillna(method='ffill', inplace=True)
    df_symbol_tick2.fillna(method='ffill', inplace=True)
    df_symbol_tick = pd.concat([df_symbol_tick1, df_symbol_tick2, df_symbol_tick3], axis=0)

    df_symbol_tick['Volume'] = df_symbol_tick['AccVolume'].diff(1)
    df_symbol_tick['Turnover'] = df_symbol_tick['AccTurnover'].diff(1)
    df_symbol_tick['Volume'].fillna(0, inplace=True)
    df_symbol_tick['Turnover'].fillna(0, inplace=True)
    df_symbol_tick.loc[df_symbol_tick.loc[:, 'Volume'] == 0, ['BSFlag']] = b' '
    update(symbol, df_symbol_tick)

end = datetime.now()
print(end - start)
