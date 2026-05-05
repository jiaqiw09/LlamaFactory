# 自定义算子

v1 支持通过 `kernel_config` 为不同硬件后端加载优化的融合算子。

## 配置

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

- `include_kernels: auto` — 自动启用当前硬件所有可用 kernel
- `include_kernels: "kernel_id1,kernel_id2"` — 手动指定
- `include_kernels: null` — 不启用

`auto` 模式下，kernel 会在启动时自动发现和注册；只有与当前硬件匹配的 kernel 会被启用。显式指定不兼容的 kernel 会报错。

## GPU

GPU 使用 Flash Attention 2 等标准算子，通过安装 `flash-attn` 启用：

```bash
pip install flash-attn
```

无需额外配置，模型加载时会自动检测并启用。

## 其他硬件后端

各硬件后端提供的专属融合算子见 [多后端支持](../multi-backend/index.md)。开发者文档见 [Kernel 插件 API](../developer-guide/plugins/custom-kernels/kernels_api.md)。
