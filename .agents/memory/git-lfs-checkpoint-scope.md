---
name: Git LFS checkpoint scope
description: A repository-specific warning about changing LFS attributes for existing model files.
---

When adding Git LFS coverage for model outputs, scope the rule to the dedicated checkpoint directory or explicitly listed files instead of matching every existing `.pth` file.

**Why:** Broadening `.gitattributes` over existing non-LFS checkpoints caused the working tree to replace full binary files with 132-byte LFS pointer files.

**How to apply:** Restore any affected binaries from the verified project backup or HEAD, then use a narrow rule such as `ai-model/checkpoints/*.pth` for future outputs.