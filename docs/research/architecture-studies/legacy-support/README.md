# Shared historical adapters

These source snapshots make the analytical extensions used by the architecture
studies inspectable. They are not called by the current ICI implementation.

- `frontier-hardware-study/engine.py`: operation-level estimates, hardware
  counterfactuals, memory admission, routing/communication and DSA extensions.
- `frontier-hardware-study/run.py` and `extend.py`: architecture construction and
  sweeps reused by the 10T extension.
- `unified-prefill-study/`: cached-prefix prefill helpers referenced by the
  frontier study.
- `deepseek-v4-5t-sol/` and `attention-architecture-only-sol/`: earlier controlled
  scaling/attention source context.

The sibling `source/` directories contain each study's own wrappers. Saved
manifests identify the source hashes used where recorded. Current captured files
are not evidence that every earlier experiment used the same revision. Full
engine distributions, model weights, environment folders, logs and estimate
caches are not included. See the [collection guide](../README.md#scope-and-reproduction).
