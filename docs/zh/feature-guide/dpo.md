# 偏好对齐（DPO）

v1 当前尚未实现 DPO 训练入口。`llamafactory-cli dpo` 会进入启动流程，但训练器路由仍会抛出 `NotImplementedError`。

## 当前可用部分

- 数据层已经定义 DPO 标准样本格式：`chosen_messages` / `rejected_messages`
- 数据转换层提供 `pair` converter，可将偏好对数据转换为 DPO 样本
- 分布式启动器保留了 `dpo` 命令入口，便于后续接入训练器

## 数据格式

DPO 数据准备方式见 [数据准备](data_preparation.md#dpo偏好对齐样本格式)，数据集 YAML 配置见 [DatasetInfo](../parameter-reference/dataset_info.md)。

## 后续实现入口

DPO 训练器接入后，应补齐以下内容：

- 最小可运行训练命令
- 全参数和 PEFT 配置示例
- DPO loss 相关参数
- Checkpoint、导出和推理衔接方式
