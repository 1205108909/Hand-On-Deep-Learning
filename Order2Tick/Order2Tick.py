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
import numpy as np
from Utility import Log


class orderbook(object):
    def __init__(self, time, bsflag, order, price, volume):
        self.time = time
        self.order = order
        self.bsflag = bsflag
        self.price = price
        self.volume = volume


class Order2Tick(object):
    tradingDay = '20200805'
    symbol = '002043.sz'
    order_file_path = 'Y:\\Data\\h5data\\stock\\order\\'
    transaction_file_path = 'Y:\\Data\\h5data\\stock\\transaction\\'
    start = 92500
    end = 93003

    def __init__(self):
        self.logger = Log.get_logger(__name__)
        # 1.从逐笔成交中取得最新价last_price
        df_symbol_transaction = self.read_transaction(self.symbol, self.tradingDay)
        df_symbol_order = self.read_order(self.symbol, self.tradingDay)
        df_symbol_transaction_after_0925 = df_symbol_transaction[(df_symbol_transaction['Time'] > 92500 * 1000)]

        ask_order_list = []
        bid_order_list = []
        for i, row_ in df_symbol_transaction_after_0925.iterrows():
            last_price = row_['Price']
            trans_before_trantime = df_symbol_transaction[
                df_symbol_transaction['Time'] < row_['Time']]  # transaction 之前的 transaction
            order_before_trantime = df_symbol_order[df_symbol_order['Time'] < row_['Time']][
                ['Time', 'FunctionCode', 'OrderNumber', 'Price', 'Volume']]  # transaction 之前的 order

            for i, row in order_before_trantime.iterrows():
                order = orderbook(row['Time'], row['FunctionCode'], row['OrderNumber'], row['Price'], row['Volume'])
                if order.bsflag == b'S':
                    ask_order_list.append(order)
                else:
                    bid_order_list.append(order)

            for i, row in trans_before_trantime.iterrows():
                for askorder in ask_order_list:
                    if (askorder.order != 0) and (askorder.order == row['AskOrder']):
                        ask_order_list.remove(askorder)
                        break
                for bidorder in bid_order_list:
                    if (bidorder.order != 0) and (bidorder.order == row['BidOrder']):
                        bid_order_list.remove(bidorder)
                        break

            def takePrice(elem):
                return elem.price

            ask_order_list.sort(key=takePrice, reverse=True)
            bid_order_list.sort(key=takePrice)

            dict_ask_order_ = {}
            dict_bid_order_= {}
            for order in ask_order_list:
                if order.price in dict_ask_order_:
                    dict_ask_order_[order.price] += order.volume
                else:
                    dict_ask_order_[order.price] = order.volume

            for order in bid_order_list:
                if order.price in dict_bid_order_:
                    dict_bid_order_[order.price] += order.volume
                else:
                    dict_bid_order_[order.price] = order.volume


            row_['AskOrders'] = dict_ask_order_
            row_['BidOrders'] = bid_order_list

        # df_symbol_transaction_bet_start_end = df_symbol_transaction[
        #     (df_symbol_transaction['Time'] > self.start * 1000) & (df_symbol_transaction['Time'] <= self.end * 1000)]
        # last_price = df_symbol_transaction_bet_start_end.iloc[-1]['Price']
        # print(f'last_price = {last_price}')
        #
        # # 2.读取order数据合并同一price的buy与sell的量
        # df_symbol_order = self.read_order(self.symbol, self.tradingDay)
        # df_symbol_order_bet_start_end = df_symbol_order[
        #     (df_symbol_order['Time'] > self.start * 1000) & (df_symbol_order['Time'] <= self.end * 1000)]
        # df_symbol_tick_start_end = df_symbol_order_bet_start_end.groupby(['Price', 'FunctionCode'])['Volume'].apply(
        #     lambda x: sum(x))
        # df_symbol_tick_start_end = df_symbol_tick_start_end.unstack()
        # df_symbol_tick_start_end.reset_index(inplace=True, drop=False)
        # df_symbol_tick_start_end['Price'] = round(df_symbol_tick_start_end['Price'], 2)
        # df_symbol_tick_start_end.set_index('Price', drop=False, inplace=True)
        # df_symbol_tick_start_end.columns = ['avgPrice', 'BuyVolume', 'SellVolume']
        #
        # df_symbol_tick_start_end.fillna(0, inplace=True)
        # df_symbol_tick_start_end = df_symbol_tick_start_end.apply(
        #     lambda x: x['SellVolume'] - x['BuyVolume'] if x['avgPrice'] > last_price else x['BuyVolume'] - x[
        #         'SellVolume'], axis=1
        # )
        # df_symbol_tick_start_end.sort_index(inplace=True, ascending=False)
        # df_symbol_tick_start_end.to_csv(f'{self.tradingDay}_{self.start}_{self.end}_tick_{self.symbol}.csv',
        #                                 header=True)

    def read_order(self, symbol, tradingday):
        """
        read order data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        with h5py.File(os.path.join(self.order_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
            if symbol not in f.keys():
                return None
            time = f[symbol]['Time']
            order_number = f[symbol]['OrderNumber']
            orderKind = f[symbol]['OrderKind']
            function_code = f[symbol]['FunctionCode']
            price = f[symbol]['Price']
            volume = f[symbol]['Volume']

            order = pd.DataFrame(
                {'Time': time, 'OrderKind': orderKind, 'OrderNumber': order_number, 'Price': price, 'Volume': volume,
                 'FunctionCode': function_code})

        return order

    def read_transaction(self, symbol, tradingday):
        """
        read transaction data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        with h5py.File(os.path.join(self.transaction_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
            if symbol not in f.keys():
                return None
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

        return transaction


if __name__ == '__main__':
    order_2_tick = Order2Tick()
