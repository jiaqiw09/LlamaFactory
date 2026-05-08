# LlamaFactory v1 Documentation Convention

This file defines documentation ownership and writing format for the v1 codebase (`src/llamafactory/v1/`). It is a shared convention for docs under `docs/`; it does not decide which language to write in, feature status, roadmap, or hardware support policy.

## Scope

Use this file to decide:

- Which documentation module owns a piece of information
- What shape each type of document should have
- Which Markdown formatting rules to follow
- How pages should link to each other without duplicating content

Do not use this file to maintain:

- Feature support matrices
- Release status or implementation roadmaps
- Full parameter values that belong in reference pages
- Hardware-specific instructions that belong in Multi-Backend docs
- Long examples that belong in feature or developer pages

## Documentation Modules

### Feature Guide

Owns user-facing workflows: "How do I do X?"

Use Feature Guide pages for complete task paths such as data preparation, SFT, inference, model export, distributed training, and custom kernels.

Each page should contain:

- The user goal and when to use this page
- A minimal runnable command or config
- End-to-end YAML examples
- Expected outputs or follow-up actions when useful
- Links to Parameter Reference for parameter details
- Links to Developer Guide only when the user needs implementation context

Do not put full parameter tables or deep implementation rationale in Feature Guide pages.

### Parameter Reference

Owns configuration facts: "What does this parameter mean?"

Use Parameter Reference pages for top-level argument dataclasses and nested plugin configs.

Each page should contain:

- One parameter table with minimal columns: parameter, type, default, description
- Valid values or plugin names when the code exposes a closed set
- Short examples only when they clarify syntax
- Links back to relevant Feature Guide pages for usage context

For top-level argument dataclasses, read fields from `src/llamafactory/v1/config/`.

For nested plugin configs:

- Prefer config keys declared in `TypedDict`
- Infer defaults from `config.get("key", default)` or clearly documented fallback logic
- Cross-check that documented keys are read by the code path or intentionally reserved

Do not explain end-to-end workflows in Parameter Reference pages.

### Developer Guide

Owns design and extension information: "Why is this designed this way, and how do I extend it?"

Use Developer Guide pages for architecture, core modules, plugin mechanisms, code paths, and extension points.

Each page should contain:

- Design motivation
- Key code paths and owning source files
- Runtime flow or lifecycle
- Extension points and registration patterns
- Small code snippets that illustrate the mechanism
- Links to Feature Guide pages for user-facing usage

Do not duplicate user tutorials or parameter tables in Developer Guide pages.

### Multi-Backend

Owns hardware/backend differences: "What changes on this backend?"

Use Multi-Backend pages for backend-specific dependencies, setup, limitations, kernels, communication behavior, and deviations from the baseline docs.

Each page should contain only backend-specific differences and link back to the baseline Feature Guide or Parameter Reference page for shared behavior.

Do not duplicate full GPU/common workflows in Multi-Backend pages.

## Entry Pages

Quick Start is an entry page, not a full documentation module. It should provide the shortest path to a successful first run and link out to the four modules above for details.

Index pages should summarize what their child pages own. Keep them short; they are navigation surfaces, not feature docs.

## Link Discipline

One piece of information should have one home.

- Feature Guide links to Parameter Reference for parameter details
- Parameter Reference links to Feature Guide for usage examples
- Developer Guide links to Feature Guide for user-facing context
- Multi-Backend links to baseline docs for shared behavior

Prefer relative links to nearby docs. Avoid empty links.

## Markdown Format

### Titles

- Use semantic titles, not numbered titles
  - Good: `## DataConverterPlugin`
  - Good: `### Alpaca Converter`
  - Bad: `## 1. DataConverterPlugin Overview`
  - Bad: `### 2.1 Alpaca Format`
- Use one `#` page title per file
- Use `##` for major sections and `###` for subsections
- Use `####` only when a page genuinely needs deep nesting
- Parameter Reference page titles should use class/config names, such as `# DataArguments` or `# PeftConfig`

### Tables

- Keep tables narrow and scannable
- Prefer 4 columns or fewer; use 5 only when the extra column prevents ambiguity
- Use centered alignment only for compact label/status columns
- Put long explanation in prose before or after the table, not inside table cells

```markdown
| name | type | default | description |
|------|------|---------|-------------|
| `name` | `str` | `auto` | Plugin name |
```

### Code Blocks

Always specify a language tag:

- YAML configs: ````yaml`
- Python code: ````python`
- Bash commands: ````bash`
- JSON examples: ````json`
- Plain text or diagrams: ````text`

### Lists

- Use `-` for unordered lists
- Use `1. 2. 3.` for ordered lists only when order matters
- Indent nested items with 4 spaces

### Notes

Use blockquotes for short notes and warnings. Bold the leading marker (e.g. `Note`, `Warning`) followed by a colon, then the message.

Use `<details>` only for optional long content that would interrupt the main flow:

```html
<details>
<summary>Alternative: load YAML from HF Hub</summary>

Content.

</details>
```

### Section Separators

Do not use `---` horizontal rules between sections. Headings already provide separation.
