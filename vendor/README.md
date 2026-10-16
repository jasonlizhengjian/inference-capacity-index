# Frozen AISim runtime

The binary wheel is distributed as a private GitHub release asset rather than
stored in Git (it exceeds GitHub's normal file-size limit).

From the repository root:

```bash
gh release download v1.0.0 \
  --repo jasonlizhengjian/inference-capacity-index --pattern '*.whl' --dir vendor
```

The complete reproduction archive includes it. `environment/setup_current.py`
verifies its SHA-256 against `reference/2026-10-15/runtime.json` before installing.
The wheel retains upstream license notices. No model weights are included.
