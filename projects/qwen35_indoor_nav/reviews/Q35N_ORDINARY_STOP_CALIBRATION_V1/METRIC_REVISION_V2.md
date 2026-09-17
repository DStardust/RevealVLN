# 因果前缀评分数值修正：门槛不变

FIT V2实际采集64/64已通过原轨迹审核。fit_v2.py在b=0复现检查中因路线12的SPL相差4.66e-9而退出，尚未计算任何非零偏置候选。全64检查最大SPL误差1.44e-8，成功、nDTW、步数、距离和导出的欧氏路径长度均完全相同。

原因：导出的path_length_m用float64坐标列表计算，而官方Habitat SPL的内部路程来自原生float32 agent position之差的np.linalg.norm，再逐步累加。理论公式相同，有限精度不同。

counterfactual_v2.py继续使用原来的动作选择、截断时刻、目标半径、网格和所有FIT/DEV门槛；仅使用原封存OfficialMetrics中SPL类原样的_euclidean_distance方法，按实际float32位置和原累加顺序重放SPL内部路程。它不修改导出的path_length_m、轨迹、原始结果或容差（复现仍为1e-9）。依赖同一个项目内Habitat环境及numpy版本，不安装新依赖。

fit_v3.py只绑定新评分实现与V2采集目录；旧fit.py/fit_v2.py/失败保留。先复现全部64条b=0，再按原九点网格拟合一次；未使用DEV，也没有新增模拟动作。新数值实现与检验在正式拟合前封存。
