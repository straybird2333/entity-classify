# readme

此文件夹下的代码主要用来将“transform”后的结果输出为包含"uid","content","ent0","ent1","ent2","typ0","typ1","typ2","domain","source"字段

已经处理好的数据集上传至hdfs的路径：lfy/ner_classify



## 流程：

- `ray_ner_label.py`：输入transform后的结果,首先由于文件过大会进行切分得到若干的`parquet`文件（保存在raw文件夹中）, 然后通过PaddleNER进行推理得到结果"uid","output","results",results中包含有所有的实体三元组(ent,type,count)，输出文件的格式也是若干的`parquet`,结果输出到tags文件夹中

- `ner_label.py`：通过筛选策略对实体三元组中的(ent, type,count)进行筛选。这里的策略是选择出现次数最高的前三个"uid","content","ent0","ent1","ent2","typ0","typ1","typ2"，结果输出至final文件夹下

- `ner_classify.py`（顺序可以和前两个任意一个替换不改变最终结果）：只利用content字段，对文本进行分类，增加一个新字段"domain"，

- `filter.py`：筛选知识库