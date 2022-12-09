'''
Author: niuhui nhzju2016@163.com
Date: 2022-08-24 16:23:18
LastEditors: niuhui nhzju2016@163.com
LastEditTime: 2022-09-17 20:30:51
FilePath: /rl4mm/environment/orderManager.py
Description: OrderManager
'''

import sys
import datetime as dt
import time
from utils.contract import CON_LIST
from utils.const import (Direction,
                         Offset,
                         OrderStatus,
                         OrderType,
                         Contract,
                         OrderResponse,
                         )

class SimpleMktData:
    def __init__(self, obj):
        self.obj = obj
        self.bid = []
        self.ask = []
        self.bid_sz = []
        self.ask_sz = []

class Trade:
    def __init__(self, obj, price, volume, ts):
        self.obj = obj
        self.price = price
        self.volume = volume # positive for buy and negative for sell
        self.ts = ts

class OrderManager:
    """
    - my_quote
    - order_dict
    - done_orderid_set
    """
    def __init__(self, obj):
        self.obj = obj
        self.contract= Contract(**CON_LIST[obj])
        self.half_tick_sz = self.contract.price_tick/2
        self.my_quote = SimpleMktData(obj)
        self.target_quote = SimpleMktData(obj)  #没用到
        self.fire_map = SimpleMktData(obj)
        self.done_orderid_set = set() # fastest way to check if an order id is done
        self.order_dict = {} # order_id: list(orders)
        self.fire_done = True
        self.last_mkt_order_update_ts = dt.datetime(2020,1,1)

    def update_order(self, order):
        """
        - add new order
        - remove done order
        """
        if not order.order_id in self.order_dict:
            # new order submitted, add to order dict
            self.order_dict[order.order_id] = [order]
        else:
            # existing order_id
            # self.order_dict[order.order_id] = [order]
            self.order_dict[order.order_id].append(order)
            
        # self.order_dict[order.order_id].append(order)
        self.update_order_dict(order)

        # update my quote
        self.update_my_quote()
        
    def update_my_quote(self):
        # benchmark
        t0 = time.time()
        new_quote = SimpleMktData(self.obj)
        fire_done = True # flag for all order status
        # start looping all the alive orders
        for _, order_list in self.order_dict.items():
            if len(order_list) == 0:
                continue
            order = order_list[-1]
            if order.status == OrderStatus.SUBMITTING:
                # firing in progress, don't know the result yet.
                fire_done = False
                continue
            else:  # either ACCEPTED or PART_TRADED. perhaps sending cancel (ignore that for now)
                if order.direction == Direction.BUY:
                    if order.price in new_quote.bid:
                        idx = new_quote.bid.index(order.price)
                        new_quote.bid_sz[idx] += order.original_volume-order.traded_volume
                    else:
                        new_quote.bid.append(order.price)
                        new_quote.bid_sz.append(order.original_volume-order.traded_volume)
                else:
                    if order.price in new_quote.ask:
                        idx = new_quote.ask.index(order.price)
                        new_quote.ask_sz[idx] += order.original_volume-order.traded_volume
                    else:
                        new_quote.ask.append(order.price)
                        new_quote.ask_sz.append(order.original_volume-order.traded_volume)
        
        self.fire_done = fire_done
        
        # Done inserting, sort it
        rix = sorted(enumerate(new_quote.bid), key=lambda x:x[1], reverse=True)
        new_quote.bid_sz = [new_quote.bid_sz[v[0]] for v in rix]
        new_quote.bid = [v[1] for v in rix]
        
        rix = sorted(enumerate(new_quote.ask), key=lambda x:x[1], reverse=True)
        new_quote.ask_sz = [new_quote.ask_sz[v[0]] for v in rix]
        new_quote.ask = [v[1] for v in rix]

        self.my_quote = new_quote
        
        
    def update_order_dict(self, order):
        t0 = time.time()
        if order.status == OrderStatus.SUBMITTING:
            # firing in progress, wait for exchange response
            return
        elif order.status == OrderStatus.REJECTED:
            # rejected by exchange or risk limits, 
            # this order id already done, just pop it out and ignore
            order_hist = self.order_dict.pop(order.order_id)
            self.done_orderid_set.add(order.order_id)
        elif order.status == OrderStatus.ACCEPTED:
            # this order is accepted by exchange and become alive, now let's see what happen next
            return
        elif order.status == OrderStatus.PART_TRADED:
            # partly traded, still alive, let the main logic determine what happen next
            return
        elif order.status == OrderStatus.ALL_TRADED:
            # this order is fully traded, remove it from order_dict and add order id to done set
            order_hist = self.order_dict.pop(order.order_id)
            self.done_orderid_set.add(order.order_id)
        elif order.status == OrderStatus.CANCELED:
            # this order is officially canceled, remove it from order_dict and add order id to done set
            order_hist = self.order_dict.pop(order.order_id)
            self.done_orderid_set.add(order.order_id)
        
        self.order_dict = dict(sorted(self.order_dict.items(),key=lambda x: x[1][0].order_id))
        # logger.log_info(f"OrderManager UpdateOrderDict ts: {1000*(time.time()-t0):.4f}ms")
   
    def get_clean_mktdata(self, tick):
        # remove my quote and see the market data
        # if our quote order haven't DONE -> TODO
        clean_mkt = SimpleMktData(self.obj)
        clean_mkt.bid = tick.bid[:]
        clean_mkt.bid_sz = tick.bid_volume[:]
        clean_mkt.ask = tick.ask[:]
        clean_mkt.ask_sz = tick.ask_volume[:]
        
        # BID
        if len(clean_mkt.bid) > 0 and len(self.my_quote.bid) > 0:
            for _, (p, s) in enumerate(zip(self.my_quote.bid, self.my_quote.bid_sz)):
                i_ori = -1
                for miv, mkt_v in enumerate(clean_mkt.bid): #  error if px not found, that means either traded or out of top 5 level
                    if abs(mkt_v - p) < self.half_tick_sz:
                        i_ori = miv
                        break
                if i_ori >= 0:
                    clean_mkt.bid_sz[i_ori] -= s
                else:
                    # logger.log_info(f"px: {p:.5f} not found in top {len(tick.bid)} level of mktdata")
                    # might possibly add the remaining quotes to cancel list
                    break
            
            clean_mkt.bid = [clean_mkt.bid[x[0]] for x in enumerate(clean_mkt.bid_sz) if x[1] > 0]
            clean_mkt.bid_sz = [clean_mkt.bid_sz[x[0]] for x in enumerate(clean_mkt.bid_sz) if x[1] > 0]

        # ASK
        if len(clean_mkt.ask) > 0 and len(self.my_quote.ask) > 0:
            for _, (p, s) in enumerate(zip(self.my_quote.ask, self.my_quote.ask_sz)):
                i_ori = -1
                for miv, mkt_v in enumerate(clean_mkt.ask): #  error if px not found, that means either traded or out of top 5 level
                    if abs(mkt_v - p) < self.half_tick_sz:
                        i_ori = miv
                        break
                if i_ori >= 0:
                    clean_mkt.ask_sz[i_ori] -= s
                else:
                    # logger.log_info(f"px: {p:.5f} not found in top {len(tick.ask)} level of mktdata")
                    # might possibly add the remaining quotes to cancel list
                    break
            
            clean_mkt.ask = [clean_mkt.ask[x[0]] for x in enumerate(clean_mkt.ask_sz) if x[1] > 0]
            clean_mkt.ask_sz = [clean_mkt.ask_sz[x[0]] for x in enumerate(clean_mkt.ask_sz) if x[1] > 0]

        return clean_mkt

    def order_in_the_mkt(self):
        # returns True only order_dict is not empty
        if len(self.order_dict) == 0:
            return False
        else:
            return True




if __name__ == "__main__":
    ### This is an example of the use of OrderManager
    om = OrderManager("RB")
    # add new order
    order1 = OrderResponse(id=1,
                           direction=Direction.BUY,
                           offset=Offset.OPEN,
                           ordertype=OrderType.LIMIT,
                           price=2,
                           volume=10,
                          )
    order2 = OrderResponse(id=2,
                           direction=Direction.SELL,
                           offset=Offset.OPEN,
                           ordertype=OrderType.LIMIT,
                           price=4,
                           volume=10,
                          )
    order1.status = order2.status = OrderStatus.ACCEPTED
    om.update_order(order1)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)
    om.update_order(order2)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)
    
    # update order
    order2.status=OrderStatus.PART_TRADED 
    order2.traded_volume = 3
    om.update_order(order2)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)

    # add new order
    order3 = OrderResponse(id=3,
                           direction=Direction.BUY,
                           offset=Offset.OPEN,
                           ordertype=OrderType.LIMIT,
                           price=1,
                           volume=5,
                          )
    
    order4 = OrderResponse(id=4,
                           direction=Direction.SELL,
                           offset=Offset.OPEN,
                           ordertype=OrderType.LIMIT,
                           price=3,
                           volume=2,
                          )
    order3.status = order4.status = OrderStatus.ACCEPTED
    om.update_order(order3)
    om.update_order(order4)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)
    print( "\n, done_orderid_set",om.done_orderid_set)

    # remove traded order
    order3.status = OrderStatus.ALL_TRADED
    order3.traded_volume = 5
    om.update_order(order3)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)
    print("order_dict",om.order_dict.items(), "\n, done_orderid_set",om.done_orderid_set)

    # 补单
    order5 = OrderResponse(id=5,
                           direction=Direction.SELL,
                           offset=Offset.OPEN,
                           ordertype=OrderType.LIMIT,
                           price=3,
                           volume=6,
                          )
    order5.status = OrderStatus.ACCEPTED
    om.update_order(order5)
    print("---\n","ask: ",om.my_quote.ask,"ask_sz: ",om.my_quote.ask_sz,"\n bid: ",om.my_quote.bid,"bid_sz: ",om.my_quote.bid_sz,)
    print( "\n, done_orderid_set",om.done_orderid_set)
