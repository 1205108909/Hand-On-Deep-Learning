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

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

etf_code = '510300.sh'
tradingDay = '20200810'
tick_file_path = 'Y:\\Data\\h5data\\stock\\tick\\'


def initialize(self):
    # 1.生产标准化时间：
    df_standand_time = self.get_standand_time()

    f = h5py.File(os.path.join(tradingDay + '.h5'), 'r')
    for symbol in f.keys():
        df_symbol_tick = self.read_tick(f, symbol)
        df_symbol_tick1 = df_symbol_tick[df_symbol_tick['Time'] < 93000000]
        df_symbol_tick2 = df_symbol_tick[
            (df_symbol_tick['Time'] >= 93000000) & (df_symbol_tick['Time'] < 150000000)]
        df_symbol_tick2 = pd.merge(df_standand_time, df_symbol_tick2, how='left')
        df_symbol_tick = pd.concat([df_symbol_tick1, df_symbol_tick2], axis=0)

    # 3.取出成分股tick数据，并进行时间标准化
    df_symbol_tick = self.get_symbol_tick(df_ETF_list, df_standand_time)
    etf_tick = self.read_h5_tick([self.etf_code], self.tradingDay, df_standand_time)
    etf_tick.set_index(['Time'], inplace=True, drop=True)

    # 4.溢价套利，按照AskPrice1买成分股，以BidPrice1卖出ETF，卖出ETF-预估现金-买入股票
    df_askprice1_tick = df_symbol_tick[['Symbol', 'Time', 'AskPrice1']]
    df_askprice1_tick = df_askprice1_tick.pivot_table(index='Symbol', columns='Time', values='AskPrice1')
    df_askprice1_tick.to_excel(self.writer_middle, sheet_name='df_askprice1_tick_pivot', index=True)
    df_amount_premiums = df_askprice1_tick.apply(lambda x: self.cal_cash_substitution(df_ETF_list, x))
    df_amount_premiums.to_excel(self.writer_middle, sheet_name='df_amount_premiums', index=True)
    df_premiums_sum = df_amount_premiums.sum(axis=0)
    df_premiums_sum.to_excel(self.writer_middle, sheet_name='df_premiums_sum', index=True)
    df_etf_premiums_profit = etf_tick['BidPrice1'] * etf_unit - etf_estimate_cash - df_premiums_sum
    df_etf_premiums_profit.to_excel(self.writer_middle, sheet_name='df_etf_premiums_profit', index=True)
    print(df_etf_premiums_profit)

    # 5.折价套利，按照BidPrice1卖出成分股，以AskPrice1买入ETF，卖出股票+预估现金-买入ETF
    df_bidprice1_tick = df_symbol_tick[['Symbol', 'Time', 'BidPrice1']]
    df_bidprice1_tick = df_bidprice1_tick.pivot_table(index='Symbol', columns='Time', values='BidPrice1')
    df_amount_discount = df_bidprice1_tick.apply(lambda x: self.cal_cash_substitution(df_ETF_list, x))
    df_discount_sum = df_amount_discount.sum(axis=0)
    df_etf_discount_profit = df_discount_sum - etf_tick['AskPrice1'] * etf_unit + etf_estimate_cash
    df_etf_discount_profit.to_excel(self.writer_middle, sheet_name='df_etf_discount_profit', index=True)

    # 6.合并折溢价利润
    df_etf_pre_dis = pd.concat([df_etf_premiums_profit, df_etf_discount_profit], axis=1)
    df_etf_pre_dis.columns = ['premiums_profit', 'discount_profit']
    df_etf_pre_dis.drop([93000000], inplace=True)  # 删除9:30:00的折溢价率，因为此时点,多数票没有tick数据
    df_etf_pre_dis.to_excel(self.writer_output, sheet_name='df_etf_pre_dis', index=True)

    self.writer_middle.save()
    self.writer_output.save()


def get_standand_time():
    tick_time = pd.date_range(f'{tradingDay} 9:30:00', f'{tradingDay} 14:57:00', freq='3s')
    tick_time = tick_time[(tick_time <= datetime.strptime(f'{tradingDay} 11:30:00', '%Y%m%d %H:%M:%S')) | (
            tick_time >= datetime.strptime(f'{tradingDay} 13:00:00', '%Y%m%d %H:%M:%S'))]
    series = pd.Series(tick_time)
    series = series.map(lambda x: x.hour * 10000000 + x.minute * 100000 + x.second * 1000)
    return pd.DataFrame({'Time': series})


def read_h5_tick(self, universe, tradingDay, df_standand_time):
    """
    :key df_standand_time 标准时间
    """
    time = datetime.now()
    if not os.path.exists(self.tick_file_path):
        self.logger.error(f"{self.tick_file_path} is not Existed")
        return
    f = h5py.File(os.path.join(self.tick_file_path, tradingDay + '.h5'), 'r')
    df_universe_tick = pd.DataFrame()
    for symbol in universe:
        df_symbol_tick = self.read_tick(f, symbol)
        df_symbol_tick = df_symbol_tick[
            (df_symbol_tick['Time'] >= 93000000) & (df_symbol_tick['Time'] < 145700000)]
        df_symbol_tick = df_symbol_tick[['Time', 'Price', 'AskPrice1', 'BidPrice1']]
        df_symbol_tick = pd.merge(df_standand_time, df_symbol_tick, how='left')
        df_symbol_tick.fillna(method='ffill', inplace=True)
        df_symbol_tick['Symbol'] = symbol
        df_universe_tick = pd.concat([df_universe_tick, df_symbol_tick], axis=0)
    time = datetime.now() - time
    print(time)
    f.close()
    return df_universe_tick


def read_tick(self, f, symbol):
    """
    read tick data
    :param symbol: '600000.sh' str
    :param tradingday: '20170104' str
    :return: pd.DataFrame
    """
    if symbol not in f.keys():
        return None
    time = f[symbol]['Time']
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
