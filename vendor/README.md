# Frozen AISim runtime

The binary wheel is distributed as a private GitHub release asset rather than
stored in Git (it exceeds GitHub's normal file-size limit).

From the repository root:

```bash
gh release download current-reproduction-2026-10-16 \
  --repo jasonlizhengjian/agent-capacity-index --pattern '*.whl' --dir vendor
```

The complete reproduction archive includes it. `environment/setup_current.py`
verifies its SHA-256 against `reference/2026-10-15/runtime.json` before installing.
The wheel retains upstream license notices. No model weights are included.
