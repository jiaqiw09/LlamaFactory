# KernelConfig

`ModelArguments.kernel_config` 的子配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `auto` | 插件名，当前仅 `auto` |
| `include_kernels` | `str` | `None` | `"auto"` 全部启用 / `"id1,id2"` 手动指定 / `None` 不启用 |

## 示例

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

各硬件后端提供的可用 kernel 列表见 [多后端支持](../multi-backend/index.md)。
