# Third-party material and licensing scope

AISimulate is an external dependency, not a bundled or modified simulator.
Upstream: https://github.com/ai-dynamo/aisimulate.
The separately inspected source checkout is
`5c466b567d0ef62d983958ddfcbccbad0e201a29`. The installed runtime is
`0.13.0.dev202610050000000066`; its exact build commit is not established.
See `environment/runtime.json` for the binary and model-configuration hashes.

## Derived communication-event logic

The topology overlay's MoE dispatch event expansion derives from the
vLLM/SGLang case structure of upstream
`crates/core/src/perfmodel/operators/moe_dispatch.rs` at the commit above.
It was modified to emit communication-event descriptors rather than native
timings. Hierarchy tiers, hop delays, and collective schedules are external
study assumptions, not a measured Google implementation.

SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0.

The unmodified upstream license and third-party notices are preserved in
`LICENSES/AISIMULATE-APACHE-2.0.txt` and
`LICENSES/AISIMULATE-THIRD-PARTY-NOTICES.md`. Their references to upstream files
do not imply that those files are redistributed in this repository.

## Architecture metadata and hardware inputs

AISimulate's bundled architecture configurations are accessed at runtime and
identified by hash, not copied here. No model weights or measured kernel
datasets are included. Model identities are `nvidia/GLM-5.2-NVFP4` and
`moonshotai/Kimi-K3`; their publisher terms remain separate from this code.
The factual hardware resource inputs and saved analytical results are study
artifacts. The original Jalapeno slides, Slack materials, and Huawei source
documents are not redistributed. Assumptions and source limitations are
recorded in `configs/catalog.json` and the historical methodology snapshot.

## Original project material

No new blanket license is assigned here to the user's original orchestration,
reports, or result artifacts. Public redistribution of that original material
requires the owner's licensing decision. The Apache notices above apply to
the identified derived material; they do not purport to license all project
content. Upstream dependencies retain their respective licenses.
