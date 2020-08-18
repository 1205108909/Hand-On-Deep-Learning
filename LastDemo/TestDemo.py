# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:12:50 2020

test  demo

@author: Administrator
"""

import warnings

from DataService.JYDataLoader import JYDataLoader

warnings.filterwarnings("ignore")
import pyarrow.parquet as pq
import os
import h5py
import pandas as pd
import datetime as dt


def get_tradable_list(date):
    jyLoader = JYDataLoader()
    df_tradable_list = jyLoader.getTradableList(date)
    df_tradable_list = df_tradable_list[~df_tradable_list['symbol'].isin(
        ['600710.sh', '601360.sh', '300372.sz', '001914.sz', '001872.sz', '600732.sh', '000155.sz', '000033.sz'])]
    list_symbol = df_tradable_list['symbol'].tolist()
    list_stockId = list(x.split('.')[0] for x in list_symbol)
    return list_symbol, list_stockId


def test_parquet_time(symbol_list, date):
    market_list_list = ['StandardTime', 'PreClose', 'Open', 'High', 'Low', 'Match', \
                        'Volume', 'TurnNum', 'Turnover', \
                        'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', \
                        'AskPrice6', 'AskPrice7', 'AskPrice8', 'AskPrice9', 'AskPrice10', \
                        'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5', \
                        'AskVolume6', 'AskVolume7', 'AskVolume8', 'AskVolume9', 'AskVolume10', \
                        'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5', \
                        'BidPrice6', 'BidPrice7', 'BidPrice8', 'BidPrice9', 'BidPrice10', \
                        'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5', \
                        'BidVolume6', 'BidVolume7', 'BidVolume8', 'BidVolume9', 'BidVolume10']
    code_df = pq.read_table(f'stock_code.parquet').to_pandas()
    path = f'{date}_Market.parquet'
    transaction_column_list = ['DataCreatedTime', 'TransactionPrice', 'TransactionVolume', \
                               'TransactionAmount', 'TransactionCondition', 'TransactionDirection', \
                               'AskOrderID', 'BidOrderID']
    patho = f'{date}_Transaction.parquet'
    parquet = pq.ParquetFile(path)
    parqueto = pq.ParquetFile(patho)
    for stock_id in symbol_list:
       if stock_id == '000001':
            stock_index = code_df[code_df['SecurityID'] == stock_id]['RowGroupIndex'].values[0]
            table = parquet.read_row_group(stock_index, market_list_list)
            market_df = table.to_pandas()
            market_df.to_csv("tick_df.csv")
            tableo = parqueto.read_row_group(stock_index, transaction_column_list)
            transaction_df = tableo.to_pandas()
            # transaction_df.to_csv("transaction_df.csv")


def test_h5_time(symbol_list, date):
    with h5py.File(os.path.join(date + '.h5'), 'r') as f:
        for symbol in symbol_list:
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



date = '20161115'
list_symbol, list_stockId = get_tradable_list(date)

start = dt.datetime.now()
test_parquet_time(list_stockId, date)
middle = dt.datetime.now()
print(f'test_parquet_time:{middle - start}')

test_h5_time(list_symbol, date)
end = dt.datetime.now()
print(f'test_h5_time:{end - middle}')
