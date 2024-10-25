# Generative Models Project 1

## 组员
- 吴童
- 潘聿阳
- 丁俊喆

## Task
![alt text](imgs/task1.png)
![alt text](imgs/task2.png)
- [data](https://disk.pku.edu.cn/anyshare/zh-cn/link/AACFBB9D65250E423A88C6F7677041F9FD?_tb=none&expires_at=2025-01-25T20%3A04%3A19%2B08%3A00&item_type=folder&password_required=false&title=Generative%20Model%20Homework&type=anonymous)

## 使用方法
- conda配置环境
```sh
conda env create -f env.yaml
```

- 训练
  - 在config.py中调整配置
  - 在tokenizer.py中调整tokenizer
然后运行：
```sh
rm -rf output/*
python train.py
```

- ppl测试
  - 注意修改模型load路径为output/下训好的checkpoint
```
python eval.py
```

