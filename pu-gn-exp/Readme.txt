<模型训练部分>
1.在安装了gn-exp之后，将polymer_unit文件夹拷贝到src文件夹，并拷贝一份PURS.py到src文件夹。
2.将需要处理的数据拷贝到data文件夹，其格式请参考gn-exp给出的例子--delaney-processed.csv
3.运行命令行--python -m polymer_unit.train --experiment polymer_unit/train.yaml 
4.如果对模型的训练结果不满意，请参考“ github.com/baldassarreFe/graph-network-explainability”中的调参脚本进行调参

<可视化部分>
1.将polymer_unit.ipynb拷贝到src文件夹

提示：加载的PURS.py脚本可能会需要安装一些对应的软件，根据报错提示安装即可