# KernelConfig

`ModelArguments.kernel_config` 的子配置，启用注册到 `KernelPlugin` 的算子替换。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 当前仅支持 `auto` |
| `include_kernels` | `str \| bool` | `None` | 启用范围 |

`include_kernels` 取值：

| 值 | 行为 |
|----|------|
| `None` / `False` / 空串 | 不启用任何 kernel |
| `"auto"` 或 `True` | 启用所有已注册的默认 kernel |
| `"id1,id2,..."` | 启用指定 kernel ID 列表（逗号分隔） |

## 注册时的设备过滤

`scan_all_kernels` 在导入阶段扫描 `kernels/ops/` 目录，每个 kernel 在 `register_kernel` 时检查 `_device == get_current_accelerator().type`：当前设备不匹配的 kernel **不会**被注册到默认表中，因此 `include_kernels="auto"` 总是只启用与当前后端兼容的算子。具体可用列表随后端而变。

## 示例

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

```yaml
kernel_config:
  name: auto
  include_kernels: npu_swiglu,npu_rms_norm
```

各算子的实现位置与扩展方式见 [custom-kernels](../developer-guide/plugins/custom-kernels/overview.md)。
