# v1 Plugin Configuration Design

End-to-end design for constraining user-facing plugin configurations in `src/llamafactory/v1/`. Replaces the current `PluginConfig`-as-dict + `TypedDict` documentation pattern with strict, schema-validated dataclasses.

---

## 1. Goals

1. **Fail fast on bad input.** Unknown keys, missing required fields, wrong types → raise at parse time, not at runtime three layers deep.
2. **Type-safe consumer access.** Plugin functions read `config.r` (typed, IDE-aware) instead of `config.get("r", 8)` (string lookup, default-as-magic).
3. **Honest schemas.** Every field appears in a schema only if a real consumer reads it. No "shared" base classes that lie about universality.
4. **Single user-facing namespace per slot.** YAML stays flat. Internal structuring is invisible to users.
5. **Schema co-located with consumer.** Each plugin module owns its variant schema. No central "schemas registry" in the config layer.
6. **No new runtime dependencies.** Pure stdlib (`dataclasses`, `typing`).

## 2. Non-goals

- Hydra-style file/group composition.
- Pydantic dependency.
- CLI-style `--peft_config.r=16` overrides (not how llamafactory users invoke training).
- Preserving arbitrary user-defined dict shapes (we deliberately reject unknown keys).

---

## 3. Background: what's wrong with the status quo

Current state ([model_args.py](../src/llamafactory/v1/config/model_args.py), [arg_utils.py](../src/llamafactory/v1/config/arg_utils.py)):

- All plugin slots (`peft_config`, `dist_config`, `quant_config`, `init_config`, `kernel_config`, `optim_config`, `lr_scheduler_config`) accept a `dict | str | None`.
- `get_plugin_config` normalizes them into `PluginConfig` — a `dict` subclass with a `.name` property. No validation.
- Per-variant schemas exist as `TypedDict` (e.g. `LoraConfigDict` in [peft.py](../src/llamafactory/v1/plugins/model_plugins/peft.py)) but are documentation only — they're never enforced.
- Plugin functions read fields via `config.get("key", default)`. Defaults live at the call site.

Failure modes this allows:

```yaml
peft_config:
  name: lora
  lroa_rank: 16          # typo: silently ignored, r defaults to 8
  use_dora: yes          # not a bool: silently coerced
```

```yaml
dist_config:
  name: fsdp2
  config_file: ./ds.json # wrong slot: silently ignored, no error
```

```yaml
quant_config:
  name: bnb              # missing quantization_bit: silently defaults to 4 inside a warning log
```

All three should be hard errors. None of them are today.

---

## 4. Core pattern: tagged-union dataclasses

Each slot accepts a "tagged union" of variant dataclasses, discriminated by a `name` field.

```python
@dataclass
class LoraConfig(StrictConfigMixin):
    name: Literal["lora"] = "lora"
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] | str = "all"
    # ... only fields LoRA actually consumes

@dataclass
class FreezeConfig(StrictConfigMixin):
    name: Literal["freeze"] = "freeze"
    freeze_trainable_layers: int = 2
    freeze_trainable_modules: list[str] | str = "all"

# At parse time, raw_dict["name"] picks which class is instantiated.
```

This is the right shape because:

- **Per-variant fields are validated.** `LoraConfig` rejects `freeze_*` fields and vice versa.
- **The discriminator is a real value** (`config.name == "lora"`), not just an existence check.
- **Static type narrowing works**: `if isinstance(cfg, LoraConfig): cfg.r` is type-safe.
- **It maps 1:1 to the existing plugin dispatch** — `PeftPlugin("lora")(...)` was already keyed by name.

We considered Hydra structured configs (ConfigStore + config groups) and Pydantic discriminated unions. Both work; both add weight (Hydra: file-layout discipline; Pydantic: a non-trivial dependency that infects type signatures). The dataclass tagged-union approach gets ~95% of the benefit at ~0% of the cost for this codebase.

---

## 5. The `StrictConfigMixin` and its tradeoffs

To keep migration incremental, the dataclass exposes a small dict-like surface so existing `config.get(...)` callsites keep working:

```python
class StrictConfigMixin:
    def get(self, key: str, default: Any = None) -> Any: ...
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
```

**This is a transitional bridge, not a permanent abstraction.** Two real semantic gotchas to be aware of:

1. **`__contains__` always returns True for declared fields.** Old dict semantics: "key exists in dict." New semantics: "field is declared on the class." If any code branches on `if "freeze_extra_modules" in config`, behavior shifts.

2. **`get(key, default)` returns the field value even when it's `None`.** Old `dict.get("compute_dtype", torch.float16)` returns `torch.float16` if the key is absent. New `StrictConfigMixin.get("compute_dtype", torch.float16)` returns `None` if the field is declared with default `None`. This already bit us once in `quantization.py` and required a workaround helper.

**Long-term direction:** kill the bridge. Migrate all plugin internals from `config.get("r", 8)` to `config.r`. Defaults belong in the schema, not at every call site. The migration is mechanical and per-plugin.

---

## 6. Where schemas live: unified with `BasePlugin`

Today, each slot has two parallel registries that must stay in sync:

```python
# Registry A: plugin functions (existing, in BasePlugin._registry)
@PeftPlugin("lora").register()
def get_lora_model(model, config, is_train): ...

# Registry B: variant schemas (new, in each plugin module)
PEFT_CONFIGS = {"lora": LoraConfig, "freeze": FreezeConfig}
```

This is a maintenance trap: add a variant to one, forget the other, and dispatch silently breaks.

**Design:** extend `BasePlugin` so a plugin name carries both its callable and its schema in a single registration:

```python
@PeftPlugin("lora", config=LoraConfig).register()
def get_lora_model(model: HFModel, config: LoraConfig, is_train: bool) -> HFModel:
    ...

@PeftPlugin("freeze", config=FreezeConfig, aliases={"old_name": "new_name"}).register()
def get_freeze_model(model: HFModel, config: FreezeConfig, is_train: bool) -> HFModel:
    ...
```

Internally `BasePlugin._registry[name]` stores `{callable, config_cls, aliases}`. The parser does:

```python
def parse_slot(slot: str, raw: dict) -> Any:
    name = raw["name"]
    plugin_cls = PLUGIN_TYPES[slot]                    # PeftPlugin, DistributedPlugin, ...
    config_cls = plugin_cls.get_config_cls(name)       # LoraConfig
    return strict_dataclass_from_dict(config_cls, raw)
```

Benefits:

- One source of truth per variant.
- Schema and consumer function visibly co-located in source.
- Config layer never imports plugin packages. Plugin import-time registration populates the registry naturally; layering is preserved (`config > utils`, not `config > plugins`).

---

## 7. Slot scope: deciding which fields go where

The trickiest part isn't *how* to validate — it's *where each field belongs*. The previous `dist_config` was a cautionary tale: it accumulated fields by topical association ("looks like a parallelism thing") rather than by consumer.

### The two-question test

For any field, ask:

1. **"If the user changes `<slot>.name` from variant A to variant B, should this field's value need to change?"**
   - No → field doesn't belong to this variant. Push out.
   - Yes → field is variant-specific. Keep.

2. **"Does this field have a dedicated consumer plugin separate from the variant plugin?"**
   - Yes → field belongs to *that* plugin's slot, not this one. Extract.
   - No → field is consumed by the variant itself. Keep.

### Worked example: `dist_config`

Walking the existing `DistributedConfig + FSDP2Config + DeepSpeedConfig` fields through these questions:

| Field | Q1: changes when name changes? | Q2: separate plugin? | Verdict |
|---|---|---|---|
| `timeout` | No (process group level) | No | Universal — duplicate in each variant |
| `cp_size` | No (sequence parallel works on top of any backend) | Yes (`sequence_parallel` plugin) | Extract to its own slot |
| `cp_mode` | No | Yes (same plugin) | Extract to same slot as `cp_size` |
| `dp_size` | Yes (FSDP2 reads via mesh; DeepSpeed derives from world_size + its own config_file) | No | FSDP2-specific |
| `mp_replicate_size` | Yes (only FSDP2 HSDP uses it; DeepSpeed ignores) | No | FSDP2-specific |
| `mp_shard_size` | Yes (same as above) | No | FSDP2-specific |
| `reshard_after_forward` | Yes | No | FSDP2-specific |
| `offload_params` | Yes | No | FSDP2-specific |
| `dcp_path` | Yes | No | FSDP2-specific |
| `config_file` | Yes (DeepSpeed only) | No | DeepSpeed-specific |

Two structural changes fall out:

1. **Pull `cp_size + cp_mode` out of `dist_config` entirely.** They're consumed by [sequence_parallel.py](../src/llamafactory/v1/plugins/model_plugins/parallelization/sequence_parallel.py), which is independent of which sharding backend you pick. Putting them under `dist_config` lies about their scope.

2. **Push `dp_size, mp_replicate_size, mp_shard_size` down into `FSDP2Config`.** They were sitting in the `DistributedConfig` base class as if shared, but DeepSpeed's wrapping doesn't read them. Putting them in the base lies about their applicability.

The only field that *is* truly universal is `timeout` — and it's a single field, not worth a base class. Just declare it in each variant.

---

## 8. The "flat outer, routed inner" pattern

The user-facing YAML stays a single block per slot. Internal Python structuring is invisible.

For slots that need internal split (rare — only when one user-facing slot covers multiple consumers), use a routing parser.

### When this applies

- `peft_config`, `quant_config`, `init_config`, `kernel_config`: simple tagged union, no routing needed.
- `dist_config`: arguably needs no routing once `cp_*` extracts to `sp_config`. Each variant is self-contained.
- `optim_config`, `lr_scheduler_config`: simple tagged union.

So with the slot decomposition above, **routing is mostly unneeded.** The pattern is documented here in case future slots accumulate cross-cutting fields again.

### Mechanism (for future use)

```python
@dataclass
class FacadeConfig:
    """User sees one slot. Internals split into named sub-schemas."""
    common: CommonSubConfig
    backend: BackendVariantA | BackendVariantB | ...

    @property
    def name(self) -> str:
        return self.backend.name

def parse_routed(raw: dict) -> FacadeConfig:
    name = raw.pop("name")
    backend_cls = BACKEND_REGISTRY[name]

    common_fields = {f.name for f in fields(CommonSubConfig)}
    backend_fields = {f.name for f in fields(backend_cls)} - {"name"}

    common_kwargs, backend_kwargs, unknown = {}, {"name": name}, []
    for k, v in raw.items():
        if k in common_fields:
            common_kwargs[k] = v
        elif k in backend_fields:
            backend_kwargs[k] = v
        else:
            unknown.append(k)

    if unknown:
        raise ValueError(
            f"Unknown keys: {unknown}\n"
            f"  Common: {sorted(common_fields)}\n"
            f"  Backend ({name}): {sorted(backend_fields)}"
        )

    return FacadeConfig(
        common=CommonSubConfig(**common_kwargs),
        backend=backend_cls(**backend_kwargs),
    )
```

### Required safety net: import-time field-collision check

If any field name appears in both `CommonSubConfig` and a backend variant, routing breaks silently. Detect at module import:

```python
def _check_no_collisions() -> None:
    common_fields = {f.name for f in fields(CommonSubConfig)}
    for name, cls in BACKEND_REGISTRY.items():
        backend_fields = {f.name for f in fields(cls)} - {"name"}
        overlap = common_fields & backend_fields
        if overlap:
            raise RuntimeError(
                f"Field name collision between CommonSubConfig and {cls.__name__}: "
                f"{sorted(overlap)}. Rename one side."
            )

_check_no_collisions()
```

### Tradeoff acknowledged

Flat YAML hides the internal split from users. `cp_size` and `offload_params` at the same indentation level *look* equivalent but aren't. **Mitigation**: error messages distinguish the buckets (see parser above), and documentation calls out the scope difference. **In practice** we avoid this whole pattern when possible — push fields down to their owners (Section 7) so each slot is self-contained.

---

## 9. Cross-slot validation

Some constraints span multiple slots and can't live inside any single schema:

- `peft_config.name == "lora"` AND `init_config.name == "init_on_meta"` → error ([model_engine.py:207](../src/llamafactory/v1/core/model_engine.py:207))
- `dist_config.name == "deepspeed"` AND `sp_config.cp_size > 1` → likely error (deepspeed wrapping doesn't honor cp dim)
- `quant_config.name == "bnb"` AND `quant_config.quantization_bit == 4` AND not training → forbids fsdp+qlora device_map auto

Today these are scattered as runtime `raise ValueError` calls inside engines. They should be:

1. **Collected at the top level**, e.g. `TrainingArguments.__post_init__` calls `_validate_cross_slot_constraints()`.
2. **Declared as a list of named rules**, each rule being a function that takes the fully-parsed args and raises with a clear message.
3. **Run after all per-slot parsing is done**, before any plugin is invoked.

This gives a single authoritative location for "what combinations are illegal," rather than scattering them through trainer/engine code where they fire late.

```python
def _validate_cross_slot_constraints(self) -> None:
    rules: list[Callable[[Self], None]] = [
        _rule_lora_disallows_init_on_meta,
        _rule_deepspeed_disallows_cp_size_gt_1,
        _rule_bnb_4bit_train_requires_specific_device_map,
    ]
    for rule in rules:
        rule(self)
```

---

## 10. Final schema layout

Concrete shape after applying the principles above. (Truncated to the interesting slots; analogous patterns apply to the others.)

### 10.1 Sequence parallel (sub-schema under `dist_config`)

Cross-cutting fields (`cp_size`, `cp_mode`) keep their internal-only schema, but
**stay under the user-facing `dist_config` namespace**. Users still write a
single flat `dist_config:` block in YAML; the parser routes `cp_*` keys into a
``SequenceParallelConfig`` sub-object via the [§8](#8-the-flat-outer-routed-inner-pattern)
façade pattern.

```python
# src/llamafactory/v1/plugins/model_plugins/parallelization/sequence_parallel.py

@dataclass
class SequenceParallelConfig(StrictConfigMixin):
    """Cross-cutting parallelism. Consumed by sequence_parallel plugin
    (model side) and by data engine (sequence chunking). Backend-orthogonal.
    """
    cp_size: int = 1
    cp_mode: Literal["ulysses"] = "ulysses"

    def __post_init__(self) -> None:
        if self.cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {self.cp_size}")
```

No top-level slot is added; this dataclass is created as a sub-object on the
``DistConfig`` wrapper (see §10.2). The discriminator is ``cp_mode`` (the SP
implementation choice), not ``name``.

### 10.2 Distributed backend (and the `DistConfig` façade)

Each backend variant is a self-contained dataclass; ``timeout`` is duplicated
across variants on purpose (see §7 / §13).

```python
# src/llamafactory/v1/plugins/trainer_plugins/distributed/hub.py

@dataclass
class FSDP2Config(StrictConfigMixin):
    name: Literal["fsdp2"] = "fsdp2"
    timeout: int = 18000
    dp_size: int | None = None
    mp_replicate_size: int = 1
    mp_shard_size: int | None = None
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None


@dataclass
class DeepSpeedConfig(StrictConfigMixin):
    name: Literal["deepspeed"] = "deepspeed"
    timeout: int = 18000
    config_file: str = ""  # required, validated in __post_init__
    # Note: no dp_size, no mp_*, no offload_params.
    # DeepSpeed parallelism settings live inside the config_file.
```

The user keeps writing one flat ``dist_config:`` block. Internally the
``DistributedPlugin.parse_config`` routes the flat input into a
``DistConfig`` wrapper holding two sub-objects:

```python
@dataclass
class DistConfig:
    backend: FSDP2Config | DeepSpeedConfig | ...   # backend-specific fields
    sp: SequenceParallelConfig                     # cross-cutting cp_size / cp_mode

    @property
    def name(self) -> str:
        # Backwards-compat: ``args.dist_config.name`` returns the backend name.
        return self.backend.name
```

The router computes ``sp_field_names`` and ``backend_field_names`` from the
respective dataclass schemas, dispatches each input key to the correct bucket,
and rejects unknown keys with an error message that lists both buckets. A
defensive collision check at parse time ensures no field name is claimed by
both ``SequenceParallelConfig`` and any backend variant.

Consumers access the two halves explicitly:

* ``args.dist_config.backend.offload_params`` — backend-specific
* ``args.dist_config.sp.cp_size`` — cross-cutting
* ``args.dist_config.name`` — preserved for legacy callsites

### 10.3 PEFT

```python
# src/llamafactory/v1/plugins/model_plugins/peft.py

@dataclass
class LoraConfig(StrictConfigMixin):
    name: Literal["lora"] = "lora"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | str = "all"
    use_rslora: bool = False
    use_dora: bool = False
    modules_to_save: list[str] | str | None = None
    adapter_name_or_path: list[str] | str | None = None
    # export-time fields kept here for now; could split into ExportConfig later
    export_dir: str | None = None
    export_size: int = 5
    export_hub_model_id: str | None = None
    infer_dtype: Literal["auto", "float16", "float32", "bfloat16"] = "auto"
    export_legacy_format: bool = False

@dataclass
class FreezeConfig(StrictConfigMixin):
    name: Literal["freeze"] = "freeze"
    freeze_trainable_layers: int = 2
    freeze_trainable_modules: list[str] | str = "all"
    freeze_extra_modules: list[str] | str | None = None
    cast_trainable_params_to_fp32: bool = True
```

Registration via the unified `BasePlugin` extension:

```python
@PeftPlugin("lora", config=LoraConfig, aliases={"lora_rank": "r"}).register()
def get_lora_model(model: HFModel, config: LoraConfig, is_train: bool) -> HFModel:
    target_modules = config.target_modules
    if target_modules == "all":
        target_modules = _find_all_linear_modules(model)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=not is_train,
        r=config.r,                           # ← attribute access, not .get
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        modules_to_save=config.modules_to_save,
        use_rslora=config.use_rslora,
        use_dora=config.use_dora,
    )
    ...
```

### 10.4 Other slots

`init_config`, `kernel_config`, `quant_config`, `optim_config`, `lr_scheduler_config` follow the same pattern. Detail:

- `init_config`: collapse `InitOnMetaConfig`, `InitOnRank0Config`, `InitOnDefaultConfig` into one `InitConfig` with `name: Literal["init_on_meta", "init_on_rank0", "init_on_default"]` until variants actually diverge.
- `quant_config`: collapse `AutoQuantConfig` and `BnbQuantConfig` similarly. They're field-identical today.
- `compute_dtype: Any` should be `compute_dtype: str | torch.dtype | None` — the loose `Any` defeats strict typing.

### 10.5 Top-level args wiring

```python
# src/llamafactory/v1/config/training_args.py

@dataclass
class TrainingArguments:
    # ... unchanged top-level fields ...
    dist_config: PluginArgument = None       # single user-facing slot
    optim_config: PluginArgument = None
    lr_scheduler_config: PluginArgument = None

    def __post_init__(self) -> None:
        # Slot parsing happens lazily in core/trainer where plugins are imported.
        # __post_init__ only normalizes raw input shape — does NOT import plugins.
        self.dist_config = normalize_plugin_argument(self.dist_config)
        self.optim_config = normalize_plugin_argument(self.optim_config)
        self.lr_scheduler_config = normalize_plugin_argument(self.lr_scheduler_config)
```

Strict parsing is centralized in ``core/utils/config_parsing.py`` and called
from each trainer entry point (e.g. ``run_sft``):

```python
def parse_training_plugin_configs(training_args) -> None:
    from ...plugins.trainer_plugins.distributed.hub import DistributedPlugin
    if isinstance(training_args.dist_config, dict):
        # parse_config does flat-outer / routed-inner: produces a DistConfig
        # façade with .backend (variant) and .sp (SequenceParallelConfig).
        training_args.dist_config = DistributedPlugin.parse_config(training_args.dist_config)
```

This preserves the layering rule that ``config/`` does not depend on
``plugins/``.

### 10.6 YAML examples

User-facing form is **one flat ``dist_config:`` block**. The flat-outer / routed-inner
parser dispatches keys to either the backend dataclass or the SP dataclass; users
do not need to nest them.

FSDP2 + sequence parallel:
```yaml
dist_config:
  name: fsdp2
  timeout: 18000
  mp_shard_size: 8
  offload_params: true
  cp_size: 2          # routed to SequenceParallelConfig internally
  cp_mode: ulysses    # routed to SequenceParallelConfig internally
```

DeepSpeed:
```yaml
dist_config:
  name: deepspeed
  timeout: 18000
  config_file: ./examples/deepspeed/zero3.json
  # No mp_*, no dp_size — they're not part of DeepSpeedConfig.
  # Anything DeepSpeed-specific lives in config_file.
```

Switching ``name: fsdp2 → deepspeed`` will reject backend-specific keys
(``offload_params``, ``mp_shard_size``) with a clear error listing both the
backend bucket and the SP bucket — precisely the validation gap the design set
out to close.

---

## 11. Migration plan

The migration is incremental. Each step is independently shippable.

### Step 1: schema infrastructure

- [ ] `StrictConfigMixin` + `strict_dataclass_from_dict` + `parse_named_dataclass_config` in `arg_utils.py`. (Done in current draft.)
- [ ] Fix `DeepSpeedConfig` field-order TypeError (use `kw_only=True` or sentinel default + `__post_init__` validation).
- [ ] Audit `StrictConfigMixin.get/__contains__` semantics or commit to migrating callsites away from dict access.
- [ ] Move per-slot strict parsing out of `config/__post_init__` into `core/` consumers to preserve layering.

### Step 2: per-slot migration (one slot per PR)

Order by risk, lowest first:
1. `init_config` (trivial — collapse three empty classes into one)
2. `kernel_config` (single variant)
3. `quant_config` (collapse Auto/Bnb)
4. `peft_config` (most fields; alias support; export logic touchpoints)
5. `dist_config` + new `sp_config` slot extraction
6. `optim_config` / `lr_scheduler_config` (need plugin variants registered first)

For each: add tests for unknown-key rejection, missing-required rejection, alias application, name-only string shorthand.

### Step 3: unify schema with `BasePlugin`

Extend `BasePlugin.__init__` and `register()` to accept `config=` and `aliases=`. Migrate registrations one slot at a time. Drop the parallel `XXX_CONFIGS` dicts.

### Step 4: cross-slot validation refactor

Collect existing `raise ValueError` constraints scattered through `model_engine.py`, `base_trainer.py`, etc. into a single `_validate_cross_slot_constraints()` registry on `TrainingArguments`/`ModelArguments`.

### Step 5: kill the dict bridge

Once all callsites use attribute access, remove `StrictConfigMixin.get/__getitem__/__contains__`. Keep `StrictConfigMixin` as a marker base class only (or drop it entirely if no shared behavior remains).

---

## 12. Open questions

- **Aliases with deprecation warnings.** When `lora_rank` is auto-mapped to `r`, we should warn (not silently rewrite) so users migrate. Add `logger.warning_rank0` inside the alias-application path.
- **Registry timing.** If a plugin module isn't imported before its config is parsed, the registry is empty. Resolution: parse-time imports happen in `core/`, which always imports the plugin modules. Document this invariant.
- **Sub-config sharing across slots.** E.g. if both `peft_config.export_*` and a future `merge_config` need export fields, do we extract `ExportConfig`? Not yet — wait for the second use site.
- **JSON schema export for documentation.** Once schemas are dataclasses, generating user-facing YAML reference docs from them is a small addition. Defer until Step 5 lands.

---

## 13. Things deliberately not done, and why

- **No Pydantic.** Validation cost-benefit doesn't justify the dependency. We get 95% of the value from stdlib dataclasses.
- **No Hydra.** File-based config groups conflict with the "one YAML per experiment" habit; CLI override is not a user need; OmegaConf's `DictConfig` leaks into downstream code.
- **No deep inheritance / mixin hierarchies for shared fields.** Past three variants, the diamond grows faster than the duplication. We prefer 50 lines of duplicated field declarations over 200 lines of mixin choreography. Repetition that mirrors reality is easier to maintain than abstraction that hides it.
- **No pre-splitting of equivalent variant classes.** `init_on_meta`, `init_on_rank0`, `init_on_default` get one shared `InitConfig` until they actually diverge. Future-proofing happens at the dataclass-field level (adding fields is non-breaking), not at the class level.
- **No "everything inherits from a base PluginConfig."** We allow each variant dataclass to be self-contained. Common behavior (`StrictConfigMixin`) is a transitional concern; once the dict bridge dies, the marker class can be dropped.
