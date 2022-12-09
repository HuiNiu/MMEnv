'''
Author: niuhui nhzju2016@163.com
Date: 2022-08-19 10:30:47
LastEditors: niuhui nhzju2016@163.com
LastEditTime: 2022-08-26 03:44:42
FilePath: /MM/rl4mm/utils/const.py
Description: 做市基类
'''
from calendar import c
from datetime import datetime
import sys

##############
# 这样定义可以免实例化

class _const:
    """常量类"""
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value

# sys.modules[__name__] = _const()

Direction = _const()
Direction.BUY = 1
Direction.SELL = 2


Offset = _const()
Offset.OPEN = 1
Offset.CLOSE = 2
Offset.CLOSE_TODAY = 4


OrderStatus = _const()
OrderStatus.SUBMITTING = 1
OrderStatus.REJECTED = 2
OrderStatus.ACCEPTED = 4
OrderStatus.PART_TRADED = 8
OrderStatus.ALL_TRADED = 16
OrderStatus.CANCELED = 32


OrderType= _const()
OrderType.MARKET = 1
OrderType.LIMIT = 2

# class Direction():
#     def __init__(self):
#         self.BUY = 1
#         self.SELL = 2

# class Offset():
#     def __init__(self):
#        self.OPEN = 1
#        self.CLOSE = 2
#        self.CLOSE_TODAY = 4

# class OrderStatus():
#     def __init__(self):
#         self.SUBMITTING = 1
#         self.REJECTED = 2
#         self.ACCEPTED = 4
#         self.PART_TRADED = 8
#         self.ALL_TRADED = 16
#         self.CANCELED = 32

# class OrderType():
#     def __init__(self):
#         self.MARKET = 1
#         self.LIMIT = 2
######
class Contract():
    """
    ticker_id:以整数来表示一个合约。使用数值类型来表示合约可以方便快速地进行函数之间的传参、在不同模块间通过协议进行传递合约信息,例如发单程序通过指定发单协议中的ticker_id,来告诉trader需要交易哪个合约。需要注意的是:ticker_id是从1开始连续递增的整数,0是一个无效的ticker_id
    ticker:交易所合约代码,和交易所交互时需要用到。例如:IF2109, 10003205
    instr:合约的内部命名表示。例如:FUT_CCFX_IF:202109, OPT_XSHG_510050:202112:C:3.1
    code:
    contract_type:合约品种。例如:IF, IH
    contract_unit:合约乘数,用于资金、持仓信息等计算
    list_date:上市日期
    expire_date:到期日
    exchange:交易所代码,下单时可能要设置exchange字段。例如:CFFEX, SHFE
    price_tick:最小价格跳动
    lower_limit_price:跌停板价格
    upper_limit_price:涨停板价格
    long_margin_rate:多头保证金率
    short_margin_rate:空头保证金率
    max_market_order_volume:市价单单次最大数量
    min_market_order_volume:市价单单次最小数量
    max_limit_order_volume:限价单单次最大数量
    min_limit_order_volume:限价单单次最小数量
    product_type:合约类型,Futures/Options/Stock/Index/ETF/...
    underlying_pre_close:期权独有属性。标的物的昨收价
    underlying_symbol:期权独有属性。标的物的代码
    underlying_type:期权独有属性。标的物的类型。Futures/Options/Stock/Index/ETF/...
    exercise_price:期权独有属性。行权价格
    exercise_date:期权独有属性。行权日
    """
    def __init__(self,
                    obj, 
                    long_margin_rate,
                    short_margin_rate,
                    commission_rate, 
                    price_tick,
                    contract_unit,
                    ):
        self.obj = obj
        self.price_tick = price_tick
        self.contract_unit = contract_unit
        self.long_margin_rate = long_margin_rate
        self.short_margin_rate = short_margin_rate
        self.commission_rate = commission_rate
        
class OrderResponse():
    """订单类
    订单状态转移有下面几种可能:
        被柜台/交易所拒绝: submitting -> rejected
        订单发出去,经过多次（或一次）成交之后全成: submitting -> accepted -> part_traded -> (part_traded -> ...) all_traded
        订单发出去部分成交之后,剩下的撤销: submitting -> accepted -> part_traded -> (part_traded -> ...) canceled
        订单发出去之后没有成交就被撤销: submitting -> accepted(这个状态也可能不存在) -> canceled
    
    """
    def __init__(self, id, direction, offset, ordertype, price, volume,queuenumber = 0):
        self.order_id = id                       # 订单id
        self.direction = direction              # 方向 BUY = 1 SELL = -1
        self.offset = offset                    # 开平 OPEN =1, CLOSE = 2, CLOSE_TODAY =4
        self.status = OrderStatus.SUBMITTING    # 订单状态 SUBMITTING=1, REJECTED=2, ACCEPTED =4, PART_TRADED = 8, ALL_TRADED = 16, CANCELED = 32
        self.ordertype = ordertype              # 订单类型 MARKET = 1, LIMIT = 2
        self.price = price                      # 订单原价格
        self.original_volume =volume            # 订单原始数量
        self.traded_volume =0                   # 已成交数量
        self.queue_number = queuenumber          # 排队值
    
        
        
class TradeResponse():
    """成交记录"""
    def __init__(self,id,direction,offset,volume,price, trade_dt):
        self.order_id = id          # 订单号
        self.direction = direction  # 方向 
        self.offset = offset        # 开平
        self.volume = volume        # 本次成交量
        self.price = price          # 本次成交价
        self.trade_datetime = trade_dt

class Position():
    """持仓"""
    def __init__(self,longvol,shortvol) -> None:
        self.long_volume = longvol
        self.short_volume = shortvol
        self.long_yd_volume = 0     #多头昨仓（期货市场不需要,所以设为0）
        self.short_yd_volume = 0
