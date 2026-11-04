# Inference Performance Index (IPI)

The former Inference Capacity Index (ICI) is now the **Inference Performance
Index (IPI)**. This rename changes labels and packaging, not model assumptions,
hardware specifications, scores, numerical methods or the v1 score definition.

## Current names

- Repository: <https://github.com/jasonlizhengjian/inference-performance-index>
- Python distribution: `inference-performance-index`
- Entry point: `python3 reproduce.py` (unchanged)
- New reproduction bundles: `inference-performance-index-v1.0.0.tar.gz`, with
  an `inference-performance-index/` directory inside.

GitHub redirects the former repository URL. Existing clones can update their
remote with:

```sh
git remote set-url origin https://github.com/jasonlizhengjian/inference-performance-index.git
```

## Identifiers retained for reproducibility

`agent_index` Python imports, the historical `agent-index` command, and saved
fields such as `agent_index_GB300e`, `ici_relative_to_gb300`, `overall_ICI_GB300e`
and `VR200_ICI` are retained for compatibility. They refer to the same scores;
the historical command still targets the older snapshot described in the README.

Frozen reference files, hash-listed historical documents and archived studies
retain their original bytes and terminology. Current renderers and report
figures use IPI. Rebuilding saved results does not require renaming CSV headers.

The `v1.0.0` tag and its existing downloadable assets retain their original
contents and hashes. Its archive is still named
`inference-capacity-index-v1.0.0.tar.gz`; its documentation and `visualizations.zip`
may display ICI. Use the default branch for the renamed documentation and figures.
The release title uses IPI, but this does not imply its historical bundles were
rebuilt. New local bundles use the new name, as documented above.
