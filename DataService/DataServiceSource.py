#!/usr/bin/env python
# encoding: utf-8

"""
@author: zhaoyu
@contact: 541840146@qq.com
@file: TradableListDownLoader.py
@time: 2020/3/30 17:47
"""

from configparser import ConfigParser
import pymssql
import pandas as pd
import sys

from Utility import Log


sys.path.append('..')


class DataServiceSource(object):

    def __init__(self):
        self.server = ""
        self.user = ""
        self.password = ""
        self.database = ""
        self.initialize()
        self.conn = None
        self.logger = Log.get_logger(__name__)

    def initialize(self):
        cfg = ConfigParser()
        cfg.read('config.ini')
        self.server = cfg.get('DataService2', 'server')
        self.user = cfg.get('DataService2', 'user')
        self.password = cfg.get('DataService2', 'password')
        self.database = cfg.get('DataService2', 'database')

    def get_connection(self):
        for i in range(3):
            try:
                self.conn = pymssql.connect(self.server, self.user, self.password, self.database)
                return self.conn
            except pymssql.OperationalError as e:
                self.logger.error(e)

    def get_facts21(self, tradingday):
        """
        返回微观结构值
        :param start: '20150101'
        :param end: '20150130'
        :return: df :
        """
        stockId = []
        ats = []
        tradePeriod = []
        quoteSize = []
        turnoverPeriod = []
        tickPeriod = []
        spreadPeriod = []
        twSpread = []
        adv20 = []
        mdv21 = []
        with self.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                stmt = 'SELECT stockId, ats, tradePeriod, quoteSize, turnoverPeriod, tickPeriod, spreadPeriod, twSpread, adv20, mdv21 FROM Facts21 WHERE TradingDay = \'%s\'  ' % (tradingday)
                cursor.execute(stmt)
                for row in cursor:
                    stockId.append(row['stockId'])
                    ats.append(row['ats'])
                    tradePeriod.append(row['tradePeriod'])
                    quoteSize.append(row['quoteSize'])
                    turnoverPeriod.append(row['turnoverPeriod'])
                    tickPeriod.append(row['tickPeriod'])
                    spreadPeriod.append(row['spreadPeriod'])
                    twSpread.append(row['twSpread'])
                    adv20.append(row['adv20'])
                    mdv21.append(row['mdv21'])
        data = pd.DataFrame({'stockId': stockId, 'ats': ats, 'tradePeriod': tradePeriod, 'quoteSize': quoteSize, 'turnoverPeriod': turnoverPeriod, 'tickPeriod': tickPeriod, 'spreadPeriod': spreadPeriod,\
                             'twSpread': twSpread, 'adv20':adv20, 'mdv21':mdv21})
        return data

    def get_dailyFacts(self, tradingday):
        """
        返回微观结构值
        :param start: '20150101'
        :param end: '20150130'
        :return: df :
        """
        stockId = []
        ats = []
        tradePeriod = []
        quoteSize = []
        turnoverPeriod = []
        tickPeriod = []
        spreadPeriod = []
        twSpread = []
        accVolume = []
        closePrice = []
        adjustedClose = []
        with self.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                stmt = 'SELECT stockId, ats, tradePeriod, quoteSize, turnoverPeriod, tickPeriod, spreadPeriod,twSpread,accVolume, closePrice, ' \
                       'adjustedClose FROM DailyFacts WHERE TradingDay = \'%s\'  ' % (
                    tradingday)
                cursor.execute(stmt)
                for row in cursor:
                    stockId.append(row['stockId'])
                    ats.append(row['ats'])
                    tradePeriod.append(row['tradePeriod'])
                    quoteSize.append(row['quoteSize'])
                    turnoverPeriod.append(row['turnoverPeriod'])
                    tickPeriod.append(row['tickPeriod'])
                    spreadPeriod.append(row['spreadPeriod'])
                    twSpread.append(row['twSpread'])
                    accVolume.append(row['accVolume'])
                    closePrice.append(row['closePrice'])
                    adjustedClose.append(row['adjustedClose'])
        data = pd.DataFrame({'stockId': stockId, 'ats': ats, 'tradePeriod': tradePeriod, 'quoteSize': quoteSize,
                             'turnoverPeriod': turnoverPeriod, 'tickPeriod': tickPeriod, 'spreadPeriod': spreadPeriod,
                             'twSpread': twSpread, 'accVolume': accVolume, 'closePrice': closePrice, 'adjustedClose': adjustedClose})
        return data

    def get_dailyVolumeBin(self, tradingday):
        """
        返回微观结构值
        :param start: '20150101'
        :param end: '20150130'
        :return: df :
        """
        stockId = []
        bin1 = []
        bin2 = []
        bin3 = []
        bin4 = []
        bin5 = []
        bin6 = []
        bin7 = []
        bin8 = []
        bin9 = []
        bin10 = []
        bin11 = []
        bin12 = []
        bin13 = []
        bin14 = []
        bin15 = []
        bin16 = []
        bin17 = []
        bin18 = []
        bin19 = []
        bin20 = []
        bin21 = []
        bin22 = []
        bin23 = []
        bin24 = []
        bin25 = []
        bin26 = []
        bin27 = []
        bin28 = []
        bin29 = []
        bin30 = []
        bin31 = []
        bin32 = []
        bin33 = []
        bin34 = []
        bin35 = []
        bin36 = []
        with self.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                stmt = 'SELECT stockId, bin1, bin2, bin3, bin4, bin5, bin6,bin7,bin8, bin9, bin10,' \
                       'bin11,bin12,bin13,bin14, bin15, bin16, bin17, bin18, bin19, bin20,' \
                       'bin21,bin22, bin23, bin24,bin25,bin26,bin27,bin28,bin29,bin30,' \
                       'bin31, bin32, bin33,bin34,bin35,bin36 FROM DailyVolumeBin WHERE TradingDay = \'%s\'  ' % (
                    tradingday)
                cursor.execute(stmt)
                for row in cursor:
                    stockId.append(row['stockId'])
                    bin1.append(row['bin1'])
                    bin2.append(row['bin2'])
                    bin3.append(row['bin3'])
                    bin4.append(row['bin4'])
                    bin5.append(row['bin5'])
                    bin6.append(row['bin6'])
                    bin7.append(row['bin7'])
                    bin8.append(row['bin8'])
                    bin9.append(row['bin9'])
                    bin10.append(row['bin10'])
                    bin11.append(row['bin11'])
                    bin12.append(row['bin12'])
                    bin13.append(row['bin13'])
                    bin14.append(row['bin14'])
                    bin15.append(row['bin15'])
                    bin16.append(row['bin16'])
                    bin17.append(row['bin17'])
                    bin18.append(row['bin18'])
                    bin19.append(row['bin19'])
                    bin20.append(row['bin20'])
                    bin21.append(row['bin21'])
                    bin22.append(row['bin22'])
                    bin23.append(row['bin23'])
                    bin24.append(row['bin24'])
                    bin25.append(row['bin25'])
                    bin26.append(row['bin26'])
                    bin27.append(row['bin27'])
                    bin28.append(row['bin28'])
                    bin29.append(row['bin29'])
                    bin30.append(row['bin30'])
                    bin31.append(row['bin31'])
                    bin32.append(row['bin32'])
                    bin33.append(row['bin33'])
                    bin34.append(row['bin34'])
                    bin35.append(row['bin35'])
                    bin36.append(row['bin36'])
        data = pd.DataFrame({'stockId': stockId, 'bin1': bin1, 'bin2': bin2, 'bin3': bin3,
                             'bin4': bin4, 'bin5': bin5, 'bin6': bin6,'bin7': bin7, 'bin8': bin8,
                             'bin9': bin9, 'bin10': bin10, 'bin11': bin11, 'bin12': bin12, 'bin13': bin13, 'bin14': bin14,
                             'bin15': bin15, 'bin16': bin16, 'bin17': bin17, 'bin18': bin18, 'bin19': bin19,'bin20': bin20,
                             'bin21': bin21, 'bin22': bin22, 'bin23': bin23, 'bin24': bin24, 'bin25': bin25,
                             'bin26': bin26, 'bin27': bin27, 'bin28': bin28, 'bin29': bin29, 'bin30': bin30, 'bin31': bin31,'bin32': bin32,
                             'bin33': bin33, 'bin34': bin34, 'bin35': bin35, 'bin36': bin36})
        return data

    def get_volumeDistribution(self, tradingday):
        """
        返回微观结构值
        :param start: '20150101'
        :param end: '20150130'
        :return: df :
        """
        groupFlag = []
        bin1 = []
        bin2 = []
        bin3 = []
        bin4 = []
        bin5 = []
        bin6 = []
        bin7 = []
        bin8 = []
        bin9 = []
        bin10 = []
        bin11 = []
        bin12 = []
        bin13 = []
        bin14 = []
        bin15 = []
        bin16 = []
        bin17 = []
        bin18 = []
        bin19 = []
        bin20 = []
        bin21 = []
        bin22 = []
        bin23 = []
        bin24 = []
        bin25 = []
        bin26 = []
        bin27 = []
        bin28 = []
        bin29 = []
        bin30 = []
        bin31 = []
        bin32 = []
        bin33 = []
        bin34 = []
        bin35 = []
        bin36 = []
        with self.get_connection() as conn:
            with conn.cursor(as_dict=True) as cursor:
                stmt = 'SELECT groupFlag, bin1, bin2, bin3, bin4, bin5, bin6,bin7,bin8, bin9, bin10,' \
                       'bin11,bin12,bin13,bin14, bin15, bin16, bin17, bin18, bin19, bin20,' \
                       'bin21,bin22, bin23, bin24,bin25,bin26,bin27,bin28,bin29,bin30,' \
                       'bin31, bin32, bin33,bin34,bin35,bin36 FROM VolumeDistribution WHERE TradingDay = \'%s\'  ' % (
                    tradingday)
                cursor.execute(stmt)
                for row in cursor:
                    groupFlag.append(row['groupFlag'])
                    bin1.append(row['bin1'])
                    bin2.append(row['bin2'])
                    bin3.append(row['bin3'])
                    bin4.append(row['bin4'])
                    bin5.append(row['bin5'])
                    bin6.append(row['bin6'])
                    bin7.append(row['bin7'])
                    bin8.append(row['bin8'])
                    bin9.append(row['bin9'])
                    bin10.append(row['bin10'])
                    bin11.append(row['bin11'])
                    bin12.append(row['bin12'])
                    bin13.append(row['bin13'])
                    bin14.append(row['bin14'])
                    bin15.append(row['bin15'])
                    bin16.append(row['bin16'])
                    bin17.append(row['bin17'])
                    bin18.append(row['bin18'])
                    bin19.append(row['bin19'])
                    bin20.append(row['bin20'])
                    bin21.append(row['bin21'])
                    bin22.append(row['bin22'])
                    bin23.append(row['bin23'])
                    bin24.append(row['bin24'])
                    bin25.append(row['bin25'])
                    bin26.append(row['bin26'])
                    bin27.append(row['bin27'])
                    bin28.append(row['bin28'])
                    bin29.append(row['bin29'])
                    bin30.append(row['bin30'])
                    bin31.append(row['bin31'])
                    bin32.append(row['bin32'])
                    bin33.append(row['bin33'])
                    bin34.append(row['bin34'])
                    bin35.append(row['bin35'])
                    bin36.append(row['bin36'])
        data = pd.DataFrame({'groupFlag': groupFlag, 'bin1': bin1, 'bin2': bin2, 'bin3': bin3,
                             'bin4': bin4, 'bin5': bin5, 'bin6': bin6,'bin7': bin7, 'bin8': bin8,
                             'bin9': bin9, 'bin10': bin10, 'bin11': bin11, 'bin12': bin12, 'bin13': bin13, 'bin14': bin14,
                             'bin15': bin15, 'bin16': bin16, 'bin17': bin17, 'bin18': bin18, 'bin19': bin19,'bin20': bin20,
                             'bin21': bin21, 'bin22': bin22, 'bin23': bin23, 'bin24': bin24, 'bin25': bin25,
                             'bin26': bin26, 'bin27': bin27, 'bin28': bin28, 'bin29': bin29, 'bin30': bin30, 'bin31': bin31,'bin32': bin32,
                             'bin33': bin33, 'bin34': bin34, 'bin35': bin35, 'bin36': bin36})
        return data

if __name__ == '__main__':
    source = DataServiceSource()
    source.initialize()
    source.get_connection()
    data = source.get_facts21('20200327')
    print(data.head())