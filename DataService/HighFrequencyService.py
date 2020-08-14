import h5py
import pandas as pd
import os.path
from Utility import Log
from configparser import ConfigParser
from Order2Tick.Constants import HighFrequencyType


class HighFrequencyService(object):
    log = Log.get_logger(__name__)
    # 配置为高频数据地址

    cfg = ConfigParser()
    cfg.read('config.ini')
    tick_file_path = cfg.get('Level2Tick', 'to')
    transaction_file_path = cfg.get('Transaction', 'to')
    order_file_path = cfg.get('Order', 'to')
    orderqueue_file_path = cfg.get('OrderQueue', 'to')
    k60s_minuteBar = cfg.get('MinuteBar', 'to')

    dictTypeAndPath = {HighFrequencyType.tick: tick_file_path, HighFrequencyType.transaction: transaction_file_path,
                       HighFrequencyType.order: order_file_path, HighFrequencyType.orderqueue: orderqueue_file_path,
                       HighFrequencyType.k60s: k60s_minuteBar}

    @classmethod
    def get_tradable_stock(cls, tradingday):
        """
        read tradble symbol
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        h5file = h5py.File(os.path.join(cls.tick_file_path, ''.join([tradingday, '.h5'])), 'r')
        list = []
        for s in h5file.keys():
            list.append(s)
        h5file.close()
        return list

    @classmethod
    def read_tick(cls, symbol, tradingday):
        """
        read tick data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        with h5py.File(os.path.join(cls.tick_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
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

    @classmethod
    def read_transaction(cls, symbol, tradingday):
        """
        read transaction data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        with h5py.File(os.path.join(cls.transaction_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
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

    @classmethod
    def read_order(cls, symbol, tradingday):
        """
        read order data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: pd.DataFrame
        """
        with h5py.File(os.path.join(cls.order_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
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

    @classmethod
    def read_orderqueue(cls, symbol, tradingday):
        """
        read orderqueue data
        :param symbol: '600000.sh' str
        :param tradingday: '20170104' str
        :return: list [orderqueueitem]
                 orderqueueitem {} dict
        """
        orderqueue = []
        with h5py.File(os.path.join(cls.orderqueue_file_path, ''.join([tradingday, '.h5'])), 'r') as f:
            if symbol not in f.keys():
                return None
            time = f[symbol]['Time'][:]
            side = f[symbol]['Side'][:]
            price = f[symbol]['Price'][:]
            orderItems = f[symbol]['OrderItems'][:]
            abItems = f[symbol]['ABItems'][:]
            abVolume = f[symbol]['ABVolume'][:]

            for i in range(len(time)):
                orderqueueItem = {'Time': time[i], 'Side': side[i], 'Price': price[i], 'OrderItems': orderItems[i],
                                  'ABItems': abItems[i], 'ABVolume': abVolume[i, :]}
                orderqueue.append(orderqueueItem)
        return orderqueue

    @classmethod
    def get_main_contact(cls, contract, tradingDay):
        """
        获取主力合约文件名
        :param contract: 'IF'、'IC'、'IH' str
        :param '20190228' str
        """
        files = os.listdir(cls.index_future_path + tradingDay)
        fl = pd.DataFrame(files)
        return fl.iloc[(cls.index_future_path + tradingDay + '\\' + fl[fl[0].str.contains(contract)][0]).apply(
            os.path.getsize).idxmax()][0]

    @classmethod
    def read_fut_tick(cls, contract, tradingDay):
        """
        获取期货地址
        :param contract: 'IF'、'IC'、'IH' str
        :param '20190228' str
        """
        symbol = cls.get_main_contact(contract, tradingDay)
        tick = pd.read_csv(cls.index_future_path + tradingDay + '\\' + symbol)
        return tick[(tick['Time'] >= 90000000) & (tick['Time'] <= 151500000)]

    @classmethod
    def read_h5_key(cls, constant, tradingDay):
        try:
            symbols = []
            with h5py.File(os.path.join(cls.dictTypeAndPath[constant], tradingDay + '.h5'), 'r') as f:
                for symbol in f.keys():
                    symbols.append(symbol)
            return symbols
        except Exception as e:
            cls.log.error(f'{constant}--{tradingDay} read_h5_key is Fail')
            cls.log.error(e)


if __name__ == '__main__':
    import datetime

    symbol_list = pd.read_csv('d:\work\model_prod\pattern_universe.csv')
    symbol_list = symbol_list[symbol_list['Symbol'].str.contains('sz')]
    print(datetime.datetime.now())
    symbol_list.apply(lambda s: HighFrequencyService.read_tick(s['Symbol'], '20200102'), axis=1)
    print(datetime.datetime.now())
