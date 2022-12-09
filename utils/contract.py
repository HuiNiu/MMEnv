'''
Author: niuhui nhzju2016@163.com
Date: 2022-08-18 19:51:35
LastEditors: niuhui nhzju2016@163.com
LastEditTime: 2022-09-16 11:08:58
FilePath: /MM/rl4mm/utils/contract.py
Description: 合约信息
'''


RB_CON = {
    "obj": 'RB',
    "long_margin_rate": 0.13,
    "short_margin_rate": 0.13,
    "commission_rate": 1e-4,
    "price_tick":1,          # 最小报价单位
    "contract_unit":10,   # 合约乘数
}

FU_CON = {
    "obj": 'FU',
    "long_margin_rate": 0.15,
    "short_margin_rate": 0.15,
    "commission_rate": 5e-5,
    "price_tick":1,          # 最小报价单位
    "contract_unit":10,   # 合约乘数
}

CU_CON = {
    "obj": 'CU',
    "long_margin_rate": 0.12,
    "short_margin_rate": 0.12,
    "commission_rate": 1e-4,
    "price_tick":10,          # 最小报价单位
    "contract_unit":5,   # 合约乘数
}

CON_LIST = {"RB":RB_CON,
            "FU":FU_CON,
            "CU":CU_CON,
            }