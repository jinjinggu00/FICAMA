import numpy as np
# 参数设置，默认值情况以paresr_util.py中的参数为准
epoch=0
device='cuda:0'
experiment_root='../output'
debug=False
local_match=0
reg_rate=0
threshold=3
gamma=0.1
iter=0
R_=np.random.randn(250, 15, 15)
D_=np.random.randn(250, 15, 15)
mod='train'
backbone='st_gcn'
dataset='ntu120'
metric='eucl'
mi=1
mix=1
mixjoint=1
pe1='positional'
pe2='positional'
sat=1
m1='eucl'
m2='tcmhm'