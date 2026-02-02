# Introduction
这是一个基于`Hebb`学习的数字生命。
`hebb`学习的关键实现如下：
```python
W += modulation * mu * np.outer(input, target)
```
原理是强化正确地帮助神经网络做出决策的那些权重。
其中`modulation`是类似于强化学习中`reward`的奖励值。
# Usage
使用非常简单，只需要`clone`本仓库之后运行hebb.py即可：
```bash
git clone https://github.com/oncepython/DigitalLife.git
cd DigitalLife
python hebb.py
```
程序没有注释，有注释的地方都是AI改的，其他没有注释的是我自己写的。
所以别瞎改。
