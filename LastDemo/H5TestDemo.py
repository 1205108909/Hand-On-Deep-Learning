#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : Order2Tick.py
@Time : 2020/8/13 9:54
"""
import os
import h5py
import pandas as pd
import datetime as dt
from DataService.JYDataLoader import JYDataLoader

order_file_path = 'Y:\\Data\\h5data\\stock\\order\\'
transaction_file_path = 'Y:\\Data\\h5data\\stock\\Transaction\\'

date = '20161115'

jyLoader = JYDataLoader()
df_tradable_list = jyLoader.getTradableList(date)
df_tradable_list = df_tradable_list[~df_tradable_list['symbol'].isin(
    ['600710.sh', '601360.sh', '300372.sz', '001914.sz', '001872.sz', '600732.sh', '000155.sz', '000033.sz'])]
list_symbol = df_tradable_list['symbol'].tolist()


with h5py.File(os.path.join(f'{date}.h5'), 'r') as f:
    for symbol in list_symbol:
        time = f[symbol]['Time']
        orderKind = f[symbol]['OrderKind']
        bsflag = f[symbol]['BSFlag']
        price = f[symbol]['Price']
        volume = f[symbol]['Volume']
        ask_order = f[symbol]['AskOrder']
        bid_order = f[symbol]['BidOrder']

        transaction = pd.DataFrame(
            {'Time': time, 'OrderKind': orderKind, 'Bsflag': bsflag, 'Price': price, 'Volume': volume,
             'AskOrder': ask_order, 'BidOrder': bid_order})
        transaction.to_hdf(f'{date}_copy.h5', key=symbol)
#
# HDF5的读取：
start = dt.datetime.now()
for symbol in list_symbol:
    transaction = pd.read_hdf(f'{date}_copy.h5', key=symbol)
#
end = dt.datetime.now()

print(end - start)
