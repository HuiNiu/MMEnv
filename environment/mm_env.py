'''
Author: niuhui nhzju2016@163.com
Date: 2022-08-18 17:02:29
LastEditors: NiuHui nhzju2016@163.com
LastEditTime: 2022-10-12 16:11:03
FilePath: /MM/rl4mm/environment/mm_env.py
Description: 做市环境
'''
import sys
# sys.path.append("/home/develop/workspace/MM/rl4mm")
import gym
from turtle import st
import pandas as pd
import numpy as np
import datetime as dt
import os, time, copy
from dataDownloader.factors_manager import Config
from environment.orderManager import SimpleMktData, OrderManager,Trade
from utils.contract import CON_LIST
from utils.const import (Direction,
                         Offset,
                         OrderStatus,
                         OrderType,
                         Contract,
                         OrderResponse,
                        #  TradeResponse,
                         Position,)


class DataGenerator():
    """
    load data form files
    """

    def __init__(self, 
                 obj,
                 submit_delay = 0,
                 window_len = 2,
                 max_step = 1800, # 15min
                 train_idx = 600,
                 test_idx = 700,
                 batch_size = 2,
                 val_mode = False,
                 test_mode = False,
                 open_ts = 120,
                 time_skip = 0,
                 trend = 'all',
                 ) :
        # load data
        self.obj = obj
        self.config = Config(config_file_path="/home/develop/workspace/MM/rl4mm/data/config.yaml")
        self.obj_config = self.config.load_obj_config(obj=obj)
        self.tick_data_path = os.path.join(self.config.config["tick_data_path"], obj)
        self.factor_path = os.path.join(self.config.config["final_factor_path"], obj)
        self.calendar = pd.read_csv(os.path.join(self.config.config["calendar_path"],obj+'_calendar.csv'))
        ## pick dataset
        if trend == 'all':
            self.files = [self.calendar.loc[i,'date']+'.csv' for i in range(self.calendar.shape[0])]
        elif trend == 'vol':
            calendar = self.calendar[self.calendar['day_trend'] == 0].reset_index(drop = True)
            self.files = [calendar.loc[i,'date']+'.csv' for i in range(calendar.shape[0])]
        elif trend == 'trend':
            calendar = self.calendar[self.calendar['day_trend'] != 0].reset_index(drop = True)
            self.files = [calendar.loc[i,'date']+'.csv' for i in range(calendar.shape[0])]
        elif trend == 'up':
            calendar = self.calendar[self.calendar['day_trend'] == 1].reset_index(drop = True)
            self.files = [calendar.loc[i,'date']+'.csv' for i in range(calendar.shape[0])]
        elif trend == 'down':
            calendar = self.calendar[self.calendar['day_trend'] == -1].reset_index(drop = True)
            self.files = [calendar.loc[i,'date']+'.csv' for i in range(calendar.shape[0])]

        # self.files = os.listdir(self.factor_path)
        self.files.sort(key=lambda x:x[:-4])
        # self.files.pop(207) # delate 2020-06-18

        self.lob_factors = self.obj_config["factors"][:-40]
        self.kline_factors = self.obj_config["factors"][-40:]
        self.num_features=len(self.obj_config["factors"])     #这先用原始fac代替
        self.open_ts = open_ts
        self.time_skip = time_skip

        test_num = max(len(self.files) - test_idx, 20)
        test_num = min(test_num,len(self.files)//2)
        self.train_idx = min(train_idx,len(self.files)-test_num*2)
        self.test_idx = len(self.files) - test_num
        print(self.train_idx,"NUM OF TEST DAYS",test_num)
        print('TRAIN DATE:',self.train_idx,self.files[self.train_idx][:-4], 
              'TEST DATE:', self.test_idx,self.files[self.test_idx][:-4],
              'END DATE:',  len(self.files), self.files[-1][:-4])
        self.val_mode = val_mode
        self.test_mode = test_mode
        self.submit_delay = submit_delay
        self.batch_size = batch_size
        self.cursor = [[],0]       # [[day_indexs],[time_indexes]]
        self.window_len = window_len
        self._max_episode_steps =max_step
        self._order_set = np.array([])
        self.tmp_order = np.array([])
        self.val_order = np.array([])
        self.test_order = np.arange(self.test_idx,len(self.files))
        self.tmp_test_order = np.array([])
        a =np.arange(len(self.files))[self.train_idx:self.test_idx]
        b = [np.random.permutation(a[i:i+21]) for i in range(0,len(a),21)] 
        for l in b:
            self._order_set = np.append(self._order_set,l[:min(16,len(l))])
            if len(l)>16: 
                self.val_order = np.append(self.val_order,l[16:])
        self._order_set=self._order_set.astype(int)
        self.val_order=self.val_order.astype(int)
        print("NUM OF TRAIN DAYS:", self._order_set.shape[0])
        self.__sample_p = np.arange(1,len(self._order_set)+1)/len(self._order_set)
        self.__sample_p /= sum(self.__sample_p)
        self.step_cnt = 1
        self.cursor =[[],1]
    
    def _step(self):
        if self.test_mode:
            if self.cursor[0][0] >= len(self.files):
                return None,None,None,True
        
        obs, lob, submit_lob, future_lob = self._get_data()
        if self.cursor[1] in [0,9000,16201]: # 9:30 10:30 13:30
            self.cursor[1] += max(self.open_ts,self.time_skip +1)
            self.step_cnt += 1
        else:
            self.cursor[1] += 1 + self.time_skip
            self.step_cnt += 1
        # if (self.cursor[1] in [9000,16201,26642]) or (self.step_cnt>=self._max_episode_steps): # NOTE: 缺点是会出现不等长的episode，可能有问题；
        # if (self.cursor[1] >= min(26642,self.num_tick)) or (self.step_cnt>=self._max_episode_steps): # NOTE: 缺点是午间休市会出现价格跳水，建议这个情况用‘vol’模式学习也不错
        if (self.cursor[1] >= min(26642,self.num_tick)) or (self.step_cnt>=self._max_episode_steps) or (self.cursor[1] in [16201,26642]): # 午间休息平仓
            done = True
            # print(self.tick_data[0].loc[self.cursor[1],'ts'])
        else:
            done = False
        return obs,lob,submit_lob,future_lob,done
    
    def reset(self,start_point = None):
        self.step_cnt = 1 
        if len(self.cursor[0])==0:
            self.num_tick = 27000          
        if start_point is not None:
            self.cursor = [np.array([start_point]),self.window_len]
        elif self.test_mode:
            if len(self.tmp_test_order) == 0:
                self.tmp_test_order = self.test_order.copy()
            # self.cursor = [np.arange(self.test_idx,min(self.batch_size+self.test_idx,len(self._order_set))),1]
            self.cursor[0] = self.tmp_test_order[:min(self.batch_size,len(self.tmp_test_order))]
            self.tmp_test_order = self.tmp_test_order[min(self.batch_size,len(self.tmp_test_order)):]
            self.reload_daily()
            self.cursor[1] = self.window_len
            print('TEST DATE:',self.cursor[0],self.files[self.cursor[0][0]])
        elif self.val_mode:
            self.cursor[1] = self.window_len
            self.cursor[0] = self.val_order[:min(self.batch_size,len(self.val_order))]
            self.reload_daily()
        elif (self.cursor[1] >= min(26642,self.num_tick)) or len(self.cursor[0])==0:
            self.cursor[1] = self.window_len + np.random.randint(max(self.time_skip,1))
            if len(self.tmp_order) ==0:
                self.tmp_order = np.random.permutation(self._order_set).copy()
            self.cursor[0] = self.tmp_order[:min(self.batch_size,len(self.tmp_order))]
            self.tmp_order = self.tmp_order[min(self.batch_size,len(self.tmp_order)):]
            self.reload_daily()
        done = False
        obs,lob,submit_lob,future_lob = self._get_data()

        return obs,lob,submit_lob,future_lob,done

    def _load_daily(self,idex=0):
        file = self.files[idex]
        tick_data = pd.read_csv(os.path.join(self.tick_data_path,file))
        factor_data = pd.read_csv(os.path.join(self.factor_path,file))
        factor_data = factor_data[self.obj_config["factors"]]
        num_tick = factor_data.shape[0]
        # print(tick_data['ts'].iloc[[9000-1,16200,26642]]) #10:15 11:30 14:57
        return tick_data,factor_data,num_tick
    
    def reload_daily(self):
        self.num_tick = 27000
        tick_data,factor_data=[],[]
        for day in self.cursor[0]:
            df1,df2,n = self._load_daily(day)
            tick_data.append(df1)
            factor_data.append(df2)
            self.num_tick = min(self.num_tick,n-self.time_skip-120)
        self.tick_data = tick_data
        self.factor_data = factor_data

    def _get_data(self):
        states = np.zeros((len(self.cursor[0]),     #batch_size
                                self.window_len,
                                self.num_features))
        lob = pd.DataFrame(index=range(len(self.cursor[0])),columns=self.tick_data[0].columns)
        submit_lob = pd.DataFrame(index=range(len(self.cursor[0])),columns=self.tick_data[0].columns)
        future_lob = pd.DataFrame(index=range(len(self.cursor[0])),columns=self.tick_data[0].columns)
        for i in range(len(self.cursor[0])):
            try:
                states[i] = self.factor_data[i].iloc[self.cursor[1]-self.window_len+1 : self.cursor[1]+1].values
            except Exception as e:
                print(e.__class__.__name__,':',e)
                print(self.num_tick,self.cursor)
                print(self.factor_data[0].index, self.factor_data[0].shape[0])
                print(states[i].shape[0])
                continue
            lob.iloc[i] = self.tick_data[i].iloc[ self.cursor[1] ]
            submit_lob.iloc[i] = self.tick_data[i].iloc[self.cursor[1] + self.submit_delay ]
            future_lob.iloc[i] = self.tick_data[i].iloc[self.cursor[1] + self.submit_delay + 1]
        return states,lob,submit_lob,future_lob
    
    def eval(self):
        self.test_mode = False
        self.val_mode = True
    
    def test(self):
        self.test_mode = True
        self.val_mode = False
    
    def train(self):
        self.test_mode = False
        self.train_mode = False


class SimExchange():
    """simulate exchange
    - match_egnine: 撮合引擎"""
    def __init__(self,
                 obj =  'RB',
                 max_spread = 8+6,
                 min_sz = 10,
                 ):
        self.obj = obj
        self.contract = Contract(**CON_LIST[obj])
        self.max_spread = max_spread
        self.min_sz = min_sz
        pass

    def gen_target_quotes(self,action,lob1):
        """
        e.g. action= np.array([[.5,.1,0.9,1],[-0.1,.4,0.3,0.4]])
        # action[i]: 
        # - dim1: mid, [0,1]之间的连续值, 映射到 [-4,+4]之间的0.5跳档位
        # - dim2: spread, [0,1]之间的连续值, 映射到max_spread -6中间

        """
        price_tick = self.contract.price_tick
        target_quotes = [SimpleMktData(obj=self.obj) for i in range(action.shape[0])]
        
        ## conver action to target_mid and target_spread
        lob1['mid'] = (lob1['ap1']+lob1['bp1'])/2
        bins =  np.arange(-4.25,4.5,.5)
        c=pd.cut((action[:,0]*2-1)*4, bins)
        try:
            target_mid = pd.Series([((c[i].left+.25)* price_tick+lob1.loc[i,'mid']) for i in range(action.shape[0])]) 
        except Exception as e:
            print("ACTION_MID:",action[:,0],'BINS:',c)
            print(e)
        # print(target_mid)

        
        max_sp =  self.max_spread - 6
        bins = np.arange(-1,max_sp+3)
        c=pd.cut(action[:,1]*max_sp, bins)
        try:
            target_spread = pd.Series([x.right  for x in c])
        except(AttributeError):
            print("ACTION_SPREAD:",action[:,1],'BINS:',c)
            raise  AttributeError("'float' object has no attribute 'right'")

        # print(target_spread)

        ## generate target mid and target ask/bid
        # NOTE: 如果quote_mid和target_mid一样，符号可能是0，就要再处理一下
        target_mid = target_mid + ((target_mid + target_spread/2)%price_tick)*np.sign(lob1['mid']-target_mid)
        target_ask = target_mid + target_spread/2
        target_bid = target_mid - target_spread/2
        target_ask -= target_ask%price_tick #处理方式是挂窄一个价差
        target_bid += target_bid%price_tick
        target_ask = np.minimum(lob1['ap5'],
                                np.maximum(lob1['bp1']+price_tick,target_ask),
                                np.minimum(lob1['ap1']-price_tick+max_sp*price_tick,target_ask))
        target_bid = np.maximum(lob1['bp5'],
                                np.minimum(lob1['ap1']-price_tick,target_bid),
                                np.maximum(lob1['bp1']+price_tick-max_sp*price_tick,target_bid))
        ## generate ask/bid size
        ## version1: 对称挂单
        # a = action[:,2]**2 + action[:,2] +1
        # v3 = ((self.min_sz/a) *(action[:,2]**2))//1
        # v2 = ((self.min_sz/a) *(action[:,2]))//1
        # v1 = self.min_sz - v2 -v3
        ## version2: 允许不对称挂单
        av1 = self.min_sz*action[:,2]//1
        av2 = self.min_sz - av1
        bv1 = self.min_sz*action[:,3]//1
        bv2 = self.min_sz - bv1



        ## generate target quotes
        for i in range(action.shape[0]):
            ask = target_ask[i]
            bid = target_bid[i]
            target_quotes[i].ask = [ask+j*price_tick for j in range(1,-1,-1)]
            target_quotes[i].bid = [bid-j*price_tick for j in range(0,2)]
            target_quotes[i].ask_sz = [av2[i],av1[i]]
            target_quotes[i].bid_sz = [bv1[i],bv2[i]]
        
        return target_quotes

    def submit_order(self,om,lob2):
        """ 
        submit orders
        - convert order status: SUBMITTING -> ACCEPTED
        - add queue_number
        """
        # TODO:目前只允许挂单不允许市价单
        for _, order in om.order_dict.items():
            order=order[-1]
            if order.status != OrderStatus.SUBMITTING:
                continue
            order.status = OrderStatus.ACCEPTED
            if order.direction == Direction.BUY:
                if order.price >= lob2['ap1']:
                    order.ordertype = OrderType.MARKET
                elif order.price > lob2['bp1']:
                    pass
                elif order.price >= lob2['bp5']:
                    bid = [lob2['bp'+str(i+1)] for i in range(5)]
                    try:
                        idx = bid.index(order.price)
                        order.queue_number = lob2['bv'+str(idx+1)]
                    except:
                        order.queue_number = 0 #有可能5档价量不连续
                else:
                    order.price = lob2['bp5']
                    order.queue_number = lob2['bv5']
            elif order.direction == Direction.SELL:
                if order.price <= lob2['bp1']:
                    order.ordertype = OrderType.MARKET
                elif order.price < lob2['ap1']:
                    pass
                elif order.price <= lob2['ap5']:
                    ask = [lob2['ap'+str(i+1)] for i in range(5)]
                    try:
                        idx = ask.index(order.price)
                        order.queue_number = lob2['av'+str(idx+1)]
                    except:
                        order.queue_number = 0
                else:
                    order.price = lob2['ap5']
                    order.queue_number = lob2['av5']
            om.update_order(order)
        return om

    def trade_order(self,om,lob2,lob3):
        """
        执行order
        return: om, trade_record
        """
        trade_record = []
        ## calculate traded volume, 假设撤单率为0
        ## TODO: 补充撤单率假设
        ask = [lob2['ap'+str(i+1)] for i in range(5)]
        new_ask = [lob3['ap'+str(i+1)] for i in range(5)]
        ask_sz = [lob2['av'+str(i+1)] for i in range(5)]
        new_ask_sz = [lob3['av'+str(i+1)] for i in range(5)]
        traded_av = [0 for i in  range(5)]
        ask_change = int((new_ask[0]-ask[0])//self.contract.price_tick)

        bid = [lob2['bp'+str(i+1)] for i in range(5)]
        new_bid = [lob3['bp'+str(i+1)] for i in range(5)]
        bid_sz = [lob2['bv'+str(i+1)] for i in range(5)]
        new_bid_sz = [lob3['bv'+str(i+1)] for i in range(5)]
        traded_bv = [0 for i in  range(5)]
        bid_change = int((new_bid[0]-bid[0])//self.contract.price_tick)
        ## TEST:
        # print("ask_change:",ask_change,"tick\n",
        #       "bid_change:",bid_change,"tick\n")

        if ask_change > 0:
            for j in range(min(ask_change,5)):
                ## NOTE:考虑了不连续的问题
                if ask[j] < new_ask[0]:
                    traded_av[j] = 1e5
        elif ask_change == 0:
            if lob3['px'] >= ask[0]:
                traded_av[0] = lob3['sz']
            elif lob3['px'] > bid[0]:
                ask_ratio = (ask[0] - lob3['px'])/(ask[0] - bid[0])
                traded_av[0] = round(lob3['sz'] * ask_ratio)
        else:
            for j in range(max(0,5+ask_change)):
                # NOTE:这里还有5档不连续的问题
                try:
                    idx =new_ask.index(ask[j])
                    traded_av[j] = np.maximum(ask_sz[j] - new_ask_sz[idx],0)
                except:
                    traded_av[j] = np.maximum(ask_sz[j] ,0)
        
        if bid_change < 0:
            for j in range(min(-int(bid_change),5)): #bid change可以大于5
                if bid[j] > new_bid[0]:
                    traded_bv[j] = 1e5
        elif bid_change == 0:
            if lob3['px']<=bid[0]:
                traded_bv[0] = lob3['sz']
            elif lob3['px'] <ask[0]:
                bid_ratio = (lob3['px'] - bid[0])/(ask[0] -bid[0])
                traded_bv[0] = round(lob3['sz'] * bid_ratio)
        else:
            for j in range(max(0,int(5-bid_change))):
                try:
                    idx = new_bid.index(bid[j])
                    traded_bv[j] = np.maximum(bid_sz[j] - new_bid_sz[idx],0)
                except:
                    traded_bv[j] = np.maximum(bid_sz[j] ,0)
        ## TEST:
        # print("traded ask sz:", traded_av,"\n",
            #   "traded bid sz:", traded_bv,"\n")

        # excute orders
        om2 = OrderManager(obj=self.obj)
        om2.order_dict = om.order_dict.copy()
        for _, order in om2.order_dict.items():
            if len(order) == 0:
                continue
            order=order[-1]
            if order.direction == Direction.SELL:
                if order.price < new_ask[0]:
                    if ask_change >0:
                        order.status = OrderStatus.ALL_TRADED
                        order.traded_volume = order.original_volume
                        order.queue_number = 0
                        trade_record.append(Trade(obj=self.obj,price=order.price,volume=order.original_volume*(-1),ts=lob2['ts']))
                    else:
                        order.queue_number -= traded_av[0] #否则按照卖一成交计算
                else:
                    if order.price >= ask[0]:
                        try:
                            idx = ask.index(order.price)
                            order.queue_number -= traded_av[idx]
                        except:
                            order.queue_number -=0
                    else:
                        # nothing happened
                        pass
                if order.queue_number < 0:
                    order.status = OrderStatus.PART_TRADED
                    new_traded_volume = min(abs(order.queue_number),order.original_volume-order.traded_volume)
                    order.traded_volume += new_traded_volume
                    if order.traded_volume == order.original_volume:
                        order.status = OrderStatus.ALL_TRADED
                    order.queue_number = 0
                    trade_record.append(Trade(obj=self.obj,price=order.price,volume=new_traded_volume*(-1),ts=lob2['ts']))
                    
            elif order.direction == Direction.BUY:
                if order.price > new_bid[0]:
                    if bid_change<0:
                        order.status = OrderStatus.ALL_TRADED
                        order.traded_volume = order.original_volume
                        order.queue_number = 0
                        trade_record.append(Trade(obj=self.obj,price=order.price,volume=order.original_volume,ts=lob2['ts']))
                    else:
                        order.queue_number -= traded_bv[0] #否则按照买一成交量计算
                else:
                    if order.price <= bid[0]:
                        try:
                            idx = bid.index(order.price)
                            order.queue_number -= traded_bv[idx]
                        except:
                            pass
                    else:
                        # nothing happened
                        pass
                if order.queue_number < 0:
                    order.status = OrderStatus.PART_TRADED
                    new_traded_volume = min(abs(order.queue_number),order.original_volume-order.traded_volume)
                    order.traded_volume += new_traded_volume
                    if order.traded_volume == order.original_volume:
                        order.status = OrderStatus.ALL_TRADED
                    order.queue_number = 0
                    trade_record.append(Trade(obj=self.obj,price=order.price,volume=new_traded_volume,ts=lob2['ts']))
            # update om  
            om.update_order(order)
            ## TEST:
            # print("order_price:",order.price,
            #       "order_volume：",order.traded_volume,
            #       "order_queue:",order.queue_number)

        return om, trade_record
    
    def _step(self,oms,lob2,lob3):
        trades = []
        for k in range(lob2.shape[0]):
            oms[k]=self.submit_order(om=oms[k],lob2=lob2.iloc[k]) 
            oms[k],trade_record = self.trade_order(om=oms[k],lob2=lob2.iloc[k],lob3=lob3.iloc[k])
            trades.append(trade_record)
        return oms,trades
            


class SimAccount():
    """
    simulate account
    """

    def __init__(self,
                 obj = 'RB',
                 max_spread = 8,
                 batch_size = 2,
                 initial_capital = 1e10,
                 max_inventory =20000,
                 bonus_ratio=.4,
                 initial_inventory = '0',
                 max_init = 500,
                 ):
        

        self.obj=obj
        self.max_spread = max_spread
        self.max_inventory = max_inventory
        self.contract = Contract(**CON_LIST[obj])
        self.commission_fee = self.contract.commission_rate
        self.contract_unit = self.contract.contract_unit
        self.initial_capital = initial_capital
        self.bonus_rate = self.commission_fee * bonus_ratio
        self.initial_inventory = initial_inventory
        self.max_init = max_init
        self.reset(batch_size=batch_size)
        

    def reset(self, batch_size,initial_capital = None):
        self.batch_size = batch_size
        initial_capital = initial_capital or self.initial_capital
        self.initial_balance = np.array([initial_capital]*batch_size)      # 初始结余
        self.balance = np.array([0.]*batch_size)                # 结余
        self.margin = np.array([0.]*batch_size)                 # 保证金
        self.frozen = np.array([0.]*batch_size)                 # 冻结金额,假设保证金率为0
        self.commission =np.array([0.] *batch_size)             # taker手续费
        self.maker_bonus = np.array([0.]*batch_size)           # maker补贴rebate
        self.floating_pnl =np.array([0.]*batch_size)            # 所有持仓的浮动盈亏之和
        self.realized_pnl =np.array([0.]*batch_size)            # 所有持仓的已实现的盈亏之和
        self.balance_pnl = np.array([0.]*batch_size)
        self.available = np.array([0.]*batch_size)              # 可用资金
        self.total_equity =np.array([0.]*batch_size)            # 总权益

        self.balance = self.initial_balance + self.realized_pnl
        self.available = self.balance - self.margin -self.frozen + self.floating_pnl
        self.total_equity = self.balance + self.floating_pnl
        self.position = [Position(longvol=0,shortvol=0) for i in range(batch_size)]
        self.oms = [OrderManager(obj=self.obj) for i in range(batch_size)]
        self.order_cnt = np.array([0 for i in range(batch_size)])
        self.num_deal = np.array([0 for i in range(batch_size)])     
        self.inventory = np.array([0 for i in range(batch_size)])
        ## NOTE: random initial inventory
        if self.initial_inventory == 'random':
            Inv = 200
            max_Init = self.max_init
            init_inv = np.random.randint(low=-max_Init//Inv, high=max_Init//Inv,size=self.inventory.shape) * Inv
            self.inventory += init_inv
        else:
            try:
                init_inv = int(self.initial_inventory)
                if (abs(init_inv) >= self.max_inventory/2):
                    raise ValueError("The absolute value of 'initial_inventory' must be samller than 'max_inventory/2'!")
                self.inventory += init_inv
            except Exception as e:
                print("ERROR: initial_inventory must be str(int) or 'random'!")
                print(e)
        print("INIT_INV:",self.inventory)
        self.num_deal += abs(init_inv)
        for i in range(batch_size):
            if self.inventory[i] >0:
                self.position[i].long_volume += self.inventory[i]
            elif self.inventory[i] <0:
                self.position[i].short_volume += abs(self.inventory[i])
        self.is_init = True


    def generate_order(self,target_quote,om, order_cnt):
        """
        cancel orders
        submit new orders
        """
        ask_sz = []
        # ask_queue = []
        bid_sz = []
        # bid_queue = []
        for ap,av in zip(target_quote.ask,target_quote.ask_sz):
            if ap not in om.my_quote.ask:
                ask_sz.append(av)
            else:
                idx = om.my_quote.ask.index(ap)
                ask_sz.append(av-om.my_quote.ask_sz[idx])
        for bp,bv in zip(target_quote.bid,target_quote.bid_sz):
            if bp not in om.my_quote.bid:
                bid_sz.append(bv)
            else:
                idx = om.my_quote.bid.index(bp)
                bid_sz.append(bv-om.my_quote.bid_sz[idx])

        ## cancel orders
        # Cancel the latest order first
        dict1 = dict(sorted(om.order_dict.items(),key=lambda x: x[1][0].order_id,reverse=True))
        for _, order in dict1.items():
            order = order[-1]
            if order.direction == Direction.BUY:
                if order.price not in target_quote.bid:
                    order.status = OrderStatus.CANCELED
                    om.update_order(order)
                else:
                    idx = target_quote.bid.index(order.price)
                    # 需要撤单
                    if bid_sz[idx]<0:
                        # print('PARTED CANCEL')
                        cxl_sz = min(order.original_volume-order.traded_volume, -1*bid_sz[idx])
                        order.original_volume -= cxl_sz
                        if order.original_volume == order.traded_volume:
                            order.status = OrderStatus.ALL_TRADED
                        om.update_order(order)
            elif order.direction == Direction.SELL:
                if order.price not in target_quote.ask:
                    order.status = OrderStatus.CANCELED
                    om.update_order(order)
                else:
                    idx = target_quote.ask.index(order.price)
                    # 需要撤单
                    if ask_sz[idx]<0:
                        # print('PARTED CANCEL')
                        cxl_sz = min(order.original_volume-order.traded_volume, -1*ask_sz[idx])
                        order.original_volume -= cxl_sz
                        if order.original_volume == order.traded_volume:
                            order.status = OrderStatus.ALL_TRADED
                        # else:
                        #     order.status = OrderStatus.PART_TRADED
                        om.update_order(order)
            
        ## add new orders:
        for ap, av in zip(target_quote.ask, ask_sz):
            if av>0:
                order_cnt += 1
                new_order = OrderResponse(id=order_cnt,
                                            direction=Direction.SELL,
                                            offset=Offset.OPEN,
                                            ordertype=OrderType.LIMIT,
                                            price=ap,
                                            volume=av,
                                            queuenumber=0)
                om.update_order(new_order)
        for bp, bv in zip(target_quote.bid, bid_sz):
            if bv>0:
                order_cnt += 1
                new_order = OrderResponse(id=order_cnt,
                                            direction=Direction.BUY,
                                            offset=Offset.OPEN,
                                            ordertype=OrderType.LIMIT,
                                            price=bp,
                                            volume=bv,
                                            queuenumber=0)
                om.update_order(new_order)
        return om, order_cnt

    def gen_orders(self,target_quotes):
        for i in range(len(self.oms)):
            self.oms[i],self.order_cnt[i]=self.generate_order(target_quote=target_quotes[i],om=self.oms[i],order_cnt=self.order_cnt[i])
        return
    
    def _step(self,trades,lob1,lob3,off_pos=False):
        CONST_0 = 1
        old_balance = self.balance.copy()
        old_equity =self.total_equity.copy()
        old_ndeals = self.num_deal.copy()
        new_num_deal = self.num_deal.copy()
        bonus = np.array([0.]*lob3.shape[0])
        new_comission= np.array([0.]*lob3.shape[0])
        new_real_pnl = np.array([0.]*lob3.shape[0])
        for i in range(len(self.oms)):
            trade_record = trades[i]
            if self.is_init:
                init_floating_pnl = self.floating_pnl.copy()
                init_floating_pnl[i] = self.inventory[i]*lob1.loc[i,'px']/CONST_0
            for trade in trade_record:
                trade_pnl = -trade.price * trade.volume /CONST_0 # NOTE：防止数字太大溢出
                trade_bonus = abs(trade_pnl) *self.bonus_rate
                self.realized_pnl[i] += trade_pnl
                new_real_pnl[i] +=  trade_pnl 
                bonus[i] += trade_bonus
                if trade.volume>0:
                    self.position[i].long_volume += trade.volume
                elif trade.volume<0:
                    self.position[i].short_volume -= trade.volume
            self.inventory[i] = (self.position[i].long_volume - self.position[i].short_volume)
            self.num_deal[i] = self.position[i].long_volume + self.position[i].short_volume
            self.floating_pnl[i] = self.inventory[i]*lob3.loc[i,'px']/CONST_0  # NOTE：防止数字太大溢出
            new_num_deal[i] =self.num_deal[i]-old_ndeals[i]
            ## NOTE: 达到最大库存则全部平仓；不知道合理不合理,强平一半仓位太麻烦;
            if (abs(self.inventory[i])>=self.max_inventory) or off_pos:
                ## RECORD
                # print('---off position----')
                # print(i,self.inventory[i])
                # for trade in trade_record:
                    # print('OFF:::',trade.price, trade.volume)
                off_pnl = self.inventory[i]*lob3.loc[i,'px']/CONST_0
                self.realized_pnl[i] += off_pnl
                self.floating_pnl[i] -= off_pnl
                new_comission[i] = abs(off_pnl)*self.commission_fee
                self.position[i].long_volume = 0
                self.position[i].short_volume = 0
                self.inventory[i] = 0
                self.num_deal[i] = 0
                self.oms[i] = OrderManager(obj=self.obj) 
                self.order_cnt[i] = 1

        #TODO：补上保证金和冻结金额，虽然好像用不到
        self.commission += new_comission
        if self.is_init:
            self.initial_balance -= init_floating_pnl
            old_balance -= init_floating_pnl
            self.is_init = False
        self.balance = self.initial_balance + self.realized_pnl -self.commission
        self.available = self.balance - self.margin -self.frozen + self.floating_pnl
        self.total_equity = self.balance + self.floating_pnl
        

        
        new_realized_pnl = self.balance-old_balance
        new_floating_pnl = self.total_equity-old_equity
        inventory = self.inventory
        
        # try:
        #     spread = np.array([(self.oms[i].my_quote.ask[-1]-self.oms[i].my_quote.bid[0]) for i in range(len(self.oms))])
        # except:
        #     spread = np.array([lob3.loc[i,'ap1'] -lob3.loc[i,'bp1'] for i in range(lob3.shape[0])])
        ap1 = np.array([0.]*lob3.shape[0])
        bp1 = np.array([0.]*lob3.shape[0])
        for i in range(lob3.shape[0]):
            try: ap1[i]  = self.oms[i].my_quote.ask[-1]
            except: ap1[i]  = lob3.loc[i,'ap1']
            try: bp1[i]  = self.oms[i].my_quote.bid[0]
            except: bp1[i]  = lob3.loc[i,'bp1']
        spread = np.clip((ap1-bp1)/self.contract.price_tick,1,self.max_spread)
        ## CHECK:
        # print('NEW_REAL0',new_realized_pnl[0],'NEW_REAL1',new_real_pnl[0],
        #       'FEE',self.commission[0],new_comission[0],'\n',
        #       'NEW_FLOAT',new_floating_pnl[0],'INV',inventory,'\n',
        #       'BONUS',bonus[0],'DEAL',new_num_deal[0])
        return new_realized_pnl,new_floating_pnl,inventory,new_num_deal,bonus,spread,ap1,bp1
        



class MMEnv():
    def __init__(self,
                 obj = 'RB',
                 max_spread=8+6,
                 max_inventory =100,
                 init_inventory = '0',
                 max_initinv = 500,
                 init_capital = 1e10,
                 bonus_ratio = .4,
                 min_sz = 20,
                 submit_delay = 0,
                 window_len = 1,
                 time_skip = 0,
                 train_idx = 600,
                 test_idx = 700,
                 batch_size = 1,
                 mode = 'train',
                 max_step = 2.7e4,
                 regr_label = [],
                 cl_label = [],
                 trend = 'all',
                 ):
        self.window_len = window_len
        self.time_skip = time_skip
        self.val_mode = False
        self.test_mode = False
        self.test_idx = test_idx
        self.mode = mode
        self.batch_size = batch_size
        self._max_episode_steps = max_step
        self.max_inventory = max_inventory
        self.bonus_ratio = bonus_ratio
        self.regr_label = regr_label
        self.cl_label = cl_label
        #data
        self.src = DataGenerator(obj=obj,submit_delay=submit_delay,window_len=window_len,max_step=max_step,
                                 train_idx=train_idx,test_idx=test_idx,batch_size=batch_size,time_skip=time_skip,trend=trend)
        self.ex = SimExchange(obj=obj,max_spread=max_spread,min_sz=min_sz)
        self.account = SimAccount(obj=obj,batch_size=batch_size,initial_capital=init_capital,max_inventory=max_inventory,
                                    initial_inventory=init_inventory,max_init=max_initinv)
        self.action_space = gym.spaces.Box(low=0,high=1,shape=(4,),dtype = np.float32)
        if mode == 'valid':
            self.set_eval()
        elif mode == 'test':
            self.set_test()
        self.reset()
    
    def reset(self):
        mkt_obs,lob,submit_lob, trade_lob,done = self.src.reset()
        self.account.reset(batch_size = lob.shape[0])
        self.num_mktFeatures = mkt_obs.shape[-1]
        self.num_pvtStates = 4
        self.num_signals = len(self.regr_label) + len(self.cl_label)
        self.lob=lob
        self.submit_lob = submit_lob
        self.trade_lob = trade_lob
        pvt_obs = np.zeros((lob.shape[0],4))
        signal = self.get_signal(lob)
        done=False
        return mkt_obs,pvt_obs,signal,done

    def step(self, action):
        
        # DONE: convert action to target quote
        target_quotes = self.ex.gen_target_quotes(action,self.lob)

        # DONE: generate orders from target quote
        self.account.gen_orders(target_quotes)

        # DONE: order excution -> update account and calculate reward,update my_quote
        self.account.oms,trades=self.ex._step(self.account.oms,lob2=self.submit_lob,lob3=self.trade_lob)
        mkt_obs,lob,submit_lob, trade_lob,done=self.src._step()
        total_equity = self.account.total_equity.min()
        done = done or (self.src.step_cnt >= self._max_episode_steps) or (total_equity<=0)
        pnl1,pnl2,inventory,num_deal,bonus,spread,ap1,bp1 = self.account._step(trades,self.lob,self.trade_lob,done)
        reward = {"realized_pnl":pnl1,
                  "floating_pnl":pnl2,
                  "inventory":inventory,
                  "spread":spread,
                  "num_deal": num_deal,
                  "bonus": bonus,
                  "quote_ask":ap1,
                  "quote_bid":bp1,
                   }
        # NEXT STEP  
        
        pvt_obs = np.zeros((lob.shape[0],4))
        pvt_obs[:,0]=inventory
        pvt_obs[:,1]=action[:,0]
        for i in range(lob.shape[0]):
            pvt_obs[i,2] = np.sum(self.account.oms[i].my_quote.ask_sz)
            pvt_obs[i,3] = np.sum(self.account.oms[i].my_quote.bid_sz)
        self.lob = lob
        self.submit_lob = submit_lob
        self.trade_lob = trade_lob
        self.inventory = inventory
        
        signal = self.get_signal(lob)
        # if done:
        #     comission = np.array([0.]*lob.shape[0])
        #     for i in range(lob.shape[0]):
        #         comission[i] = abs(self.inventory[i])*trade_lob.loc[i,'px']*self.account.commission_fee /1e4
        #     reward['floating_pnl'] -= comission
            # print('FEE',comission) 
        # TODO: 达到最大库存设为down
        # if self.mode == 'train':
        #     done = done or (abs(self.inventory.mean())>=self.max_inventory)
        return mkt_obs,pvt_obs,signal,reward,done

    def get_signal(self,lob):
        signal = np.zeros((lob.shape[0],self.num_signals))
        if len(self.cl_label) >0:
            for j1,sig in enumerate(self.cl_label):
                signal[:,j1] = lob.loc[:,sig]
        if len(self.regr_label) >0:
            for j2,sig in enumerate(self.regr_label):
                signal[:,len(self.cl_label)+j2] = lob.loc[:,sig]
        return signal

    def set_eval(self):
        self.src.eval()

    def set_test(self):
        self.src.test()

    def set_train(self):
        self.src.train()






    
if __name__ == "__main__":
    ## TEST
    # action = np.array([3,2,])
    src = DataGenerator(obj='RB')
    obs,lob1,lob2,lob3,done = src.reset()
    print(lob1,lob2,lob3,)

    
    ex = SimExchange(obj='RB')
    ac = SimAccount('RB')
    om =OrderManager('RB')

    action = np.array([[0.3,.2,0.3,0.4],[0.8,.7,0.9,1]])
    tq=ex.gen_target_quotes(action=action,lob1=lob1)
    print("target quote:\n",
          "ask:",tq[0].ask,
          "ask_sz:",tq[0].ask_sz,"\n", 
          "bid:",tq[0].bid,
          "bid_sz:",tq[0].bid_sz, "\n", 
          " -----\n",
          "ask:",tq[1].ask,
          "ask_sz:",tq[1].ask_sz, "\n", 
          "bid:",tq[1].bid,
          "bid_sz:",tq[1].bid_sz, "\n", 
          "----\n",
          )

    ac.generate_order(target_quote=tq[0],om=om,order_cnt=2)
    ex.submit_order(om=om,lob2=lob2.iloc[0])
    mq = om.my_quote
    od = om.order_dict
    print("submit quote:\n",
          "ask:",mq.ask,
          "ask_sz:",mq.ask_sz,"\n", 
          "bid:",mq.bid,
          "bid_sz:",mq.bid_sz, "\n", 
          "----\n",
          "order_queue: ",od[3][0].queue_number,"\n",
          )

    om,tr = ex.trade_order(om=om,lob2=lob2.iloc[0],lob3=lob3.iloc[0])
    mq = om.my_quote
    # od = om.order_dict
    print("my quote:\n",
          "ask:",mq.ask,
          "ask_sz:",mq.ask_sz,"\n", 
          "bid:",mq.bid,
          "bid_sz:",mq.bid_sz, "\n", 
          "----\n",
        #   "order_dict: ",od,"\n",
          )
    print("trade_record:",len(tr))
    for i in range(len(tr)):
        print(i,":\n",
              "trade_price:",tr[i].price,"\n",
              "trade_volume:",tr[i].volume,"\n",
            #   "trade_direction:",tr[i].direction,"\n",
              "------\n")
    
    print('----Test Cancellation-----------')
    tq2 = tq[0]
    tq2.ask = copy.deepcopy(mq.ask)
    tq2.ask_sz = copy.deepcopy(mq.ask_sz)
    tq2.ask_sz[0] = 1
    print("target quote2:\n",
          "ask:",tq2.ask,
          "ask_sz:",tq2.ask_sz,"\n", 
          "bid:",tq2.bid,
          "bid_sz:",tq2.bid_sz, "\n", 
          "----\n",
          )
    ac.generate_order(target_quote=tq2,om=om,order_cnt=2)
    om = ex.submit_order(om=om,lob2=lob2.iloc[0])
    mq = om.my_quote
    od = om.order_dict
    print("submit quote:\n",
          "ask:",mq.ask,
          "ask_sz:",mq.ask_sz,"\n", 
          "bid:",mq.bid,
          "bid_sz:",mq.bid_sz, "\n", 
          "----\n",
          "order_queue: ",od[3][0].queue_number,"\n",
          )
    
    om,tr = ex.trade_order(om=om,lob2=lob2.iloc[0],lob3=lob3.iloc[0])
    mq = om.my_quote
    # od = om.order_dict
    print("my quote:\n",
          "ask:",mq.ask,
          "ask_sz:",mq.ask_sz,"\n", 
          "bid:",mq.bid,
          "bid_sz:",mq.bid_sz, "\n", 
          "----\n",
        #   "order_dict: ",od,"\n",
          )
    print("trade_record:",len(tr))
    for i in range(len(tr)):
        print(i,":\n",
              "trade_price:",tr[i].price,"\n",
              "trade_volume:",tr[i].volume,"\n",
            #   "trade_direction:",tr[i].direction,"\n",
              "------\n")



    ## TEST MM_ENV
    # env = MMEnv(obj='RB')
    # env.reset()
    # # print('obs:',obs)
    # _,_,r,done = env.step(action)
    # print("reward:", r,"\n",
    #       "done:",done)
    # print("position:",env.account.position[0].long_volume,env.account.position[0].short_volume,)
    # MAX_STEP = 1e3
    # t1 = time.time()
    # for i in range(int(MAX_STEP)):
    #     action = np.random.rand(2,4)
    #     mkt_obs,pvt_obs,r,done = env.step(action)
    #     print("\n",
    #           "action:\n",action,"\n",
    #         #   "reward:", r,"\n",
    #           "balance:",env.account.balance,"\n",
    #           "total_equity:",env.account.total_equity,"\n",
    #           "done:",done)
    #     print("position:",env.account.position[0].long_volume,env.account.position[0].short_volume,)
    #     if done:
    #         env.reset()
    # t2 = time.time()
    # print("TIME:",(t2-t1)/60,"min\n",
    #       "STEP:",MAX_STEP)
    

    

