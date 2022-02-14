import string
import numpy as np
import pandas as pd
import pulp as pl
import scipy.optimize as opt

PLATFORM_FEE_RATE = 0.3

ch_stat = pd.read_csv('chopt.csv', encoding='utf-8')
ch_stat['进件占比'] = ch_stat['进件数'] / ch_stat['进件数'].sum()
ch_stat['客户占比'] = ch_stat['客户数'] / ch_stat['客户数'].sum()
ch_stat['模型通过率'] = ch_stat['客户数'] / ch_stat['进件数']
ch_stat['总授信金额'] = ch_stat['客户数'] * ch_stat['平均授信额度']
ch_stat['放款金额'] = ch_stat['总授信金额'] * ch_stat['支用率']
ch_stat['利息收入'] = ch_stat['放款金额'] * ch_stat['加权利率'] * (1 - PLATFORM_FEE_RATE)
ch_stat['损失金额'] = ch_stat['放款金额'] * ch_stat['损失率']
ch_stat['FTP扣除前利润'] = ch_stat['利息收入'] - ch_stat['损失金额']
ch_stat['FTP扣除前利润率'] = ch_stat['FTP扣除前利润'] / ch_stat['放款金额']

ch_stat.head()

# pl.listSolvers(onlyAvailable=True)
solver = pl.PULP_CBC_CMD()
rng = range(ch_stat.shape[0])

# 第一个优化问题，优化FTP扣除前利润
prob1 = pl.LpProblem('Loan_Channel_Allocation_1', sense=pl.LpMaximize)
p = [pl.LpVariable('p_'+string.ascii_uppercase[i], 0, 1) for i in rng]
# 目标
obj = pl.lpSum([ch_stat['FTP扣除前利润'][i] * p[i] for i in rng])
prob1 += obj
# 约束
prob1 += pl.lpSum(p[i] for i in rng) == 1
prob1 += p[1] >= 0.2
prob1 += p[2] >= 0.2
prob1 += p[3] >= 0.35

MIN_APPROVAL_RATE = 0.72
MAX_LOSS_RATE = 0.025
MIN_WEIGHTED_INTEREST_RATE = 0.13

c1 = pl.lpSum([ch_stat['模型通过率'][i] * p[i] for i in rng])
prob1 += c1 >= MIN_APPROVAL_RATE
c2 = pl.lpSum([ch_stat['损失率'][i] * p[i] for i in rng])
prob1 += c2 <= MAX_LOSS_RATE
c3 = pl.lpSum([ch_stat['加权利率'][i] * p[i] for i in rng])
prob1 += c3 >= MIN_WEIGHTED_INTEREST_RATE
# 求解
prob1.solve(solver)
print('Status:', pl.LpStatus[prob1.status])
# 打印结果
if pl.LpStatus[prob1.status] == 'Optimal':
    p_opt1 = {pi.name: np.round(pi.value(), 3) for pi in p}
    print(p_opt1)
    print('[约束] 总体通过率: ', np.round(c1.value(), 3),
          ' (', MIN_APPROVAL_RATE, ')', sep='')
    print('[约束] 总体损失率: ', np.round(c2.value(), 3),
          ' (', MAX_LOSS_RATE, ')', sep='')
    print('[约束] 总体加权利率: ', np.round(c3.value(), 3),
          ' (', MIN_WEIGHTED_INTEREST_RATE, ')', sep='')
    print('[目标] FTP扣除前利润: ', np.round(obj.value(), 2), sep='')


# 第二个优化问题，优化放款金额
prob2 = pl.LpProblem('Loan_Channel_Allocation_2', sense=pl.LpMaximize)
p = [pl.LpVariable('p_'+string.ascii_uppercase[i], 0, 1) for i in rng]
# 目标
obj = pl.lpSum([ch_stat['放款金额'][i] * p[i] for i in rng])
prob2 += obj
# 约束
prob2 += pl.lpSum(p[i] for i in rng) == 1
prob2 += p[1] >= 0.2
prob2 += p[2] >= 0.2
prob2 += p[3] >= 0.35

MIN_APPROVAL_RATE = 0.7
MAX_LOSS_RATE = 0.025
MIN_WEIGHTED_INTEREST_RATE = 0.13
MIN_PROFIT_RATE = 0.07

c1 = pl.lpSum([ch_stat['模型通过率'][i] * p[i] for i in rng])
prob2 += c1 >= MIN_APPROVAL_RATE
c2 = pl.lpSum([ch_stat['损失率'][i] * p[i] for i in rng])
prob2 += c2 <= MAX_LOSS_RATE
c3 = pl.lpSum([ch_stat['加权利率'][i] * p[i] for i in rng])
prob2 += c3 >= MIN_WEIGHTED_INTEREST_RATE
c4 = pl.lpSum([ch_stat['FTP扣除前利润率'][i] * p[i] for i in rng])
prob2 += c4 >= MIN_PROFIT_RATE
# 求解
prob2.solve(solver)
print('Status:', pl.LpStatus[prob2.status])
# 打印结果
if pl.LpStatus[prob2.status] == 'Optimal':
    p_opt2 = {pi.name: np.round(pi.value(), 3) for pi in p}
    print(p_opt2)
    print('[约束] 总体通过率: ', np.round(c1.value(), 3),
          ' (', MIN_APPROVAL_RATE, ')', sep='')
    print('[约束] 总体损失率: ', np.round(c2.value(), 3),
          ' (', MAX_LOSS_RATE, ')', sep='')
    print('[约束] 总体加权利率: ', np.round(c3.value(), 3),
          ' (', MIN_WEIGHTED_INTEREST_RATE, ')', sep='')
    print('[约束] FTP扣除前利率: ', np.round(c4.value(), 3),
          ' (', MIN_PROFIT_RATE, ')', sep='')
    print('[目标] 放款金额: ', np.round(obj.value(), 2), sep='')


constrains = (
      {'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0},
      {'type': 'ineq', 'fun': lambda p: p[1] - 0.2},
      {'type': 'ineq', 'fun': lambda p: p[2] - 0.2},
      {'type': 'ineq', 'fun': lambda p: p[3] - 0.35},
      {'type': 'ineq', 'fun': lambda p: np.dot(ch_stat['模型通过率'], p) - MIN_APPROVAL_RATE},
      {'type': 'ineq', 'fun': lambda p: -np.dot(ch_stat['损失率'], p) + MAX_LOSS_RATE},
      {'type': 'ineq', 'fun': lambda p: np.dot(ch_stat['加权利率'], p) - MIN_WEIGHTED_INTEREST_RATE}
)

bnds = 
