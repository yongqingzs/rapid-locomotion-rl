- 0117
```txt
failed:
1. 更改日志为 tensorboard，迭代速度很慢
2. 将 ml_logger 替换为 tensorboard，出现问题

log:
1. 072748.649209 原始配置
2. 092449.713244 速度上限提高到 3.0 m/s
```
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/rapid-locomotion-rl/scripts/train.py
```