"""Opt-in communication sensitivity; frozen publication/replay is unchanged.

Native leaf timings are retained. Overlap is reconstructed as max(branch sums),
with communication adjustments INSIDE each branch. No AISim patch is needed.
Startup is a per-collective-call sensitivity, not a measured per-hop latency.
Bandwidth efficiency scales exposed communication operations only, uniformly
across existing tiers; it is not a topology/routing simulator.
"""
import copy
import json
import math

from .topology import events, phase_hops

COMM = {'CustomAllReduce', 'Nccl', 'MoeDispatch', 'P2P'}


class OperationTree:
    def __init__(self, specs, hop_overlay=False):
        self.leaves = []
        self.info = []
        self.hop_overlay = hop_overlay
        self.tree = ('sum', [self._walk(s) for s in specs])
        self.ops_json = json.dumps(self.leaves)

    def _walk(self, spec):
        kind, value = next(iter(spec.items()))
        if kind == 'Overlap':
            if value.get('scale_factor', 1) != 1:
                raise ValueError('Unaudited overlap multiplier')
            return ('max', [('sum', [self._walk(s) for s in value[k]])
                            for k in ('group_a', 'group_b')])
        # Unknown wrappers must not hide communication from this adjustment.
        if any(isinstance(v, (dict, list)) and self._contains_comm(v)
               for v in value.values()):
            raise ValueError('Communication inside unaudited wrapper: '+kind)
        calls = hop_ms = 0.
        if kind == 'P2P':
            calls = value.get('scale_factor', 1) if value['pp_size'] > 1 else 0.
            if calls and self.hop_overlay:
                raise ValueError('TPU hop assumptions do not cover PP')
        else:
            ev = events(spec)
            calls = sum(e['occurrences'] for e in ev)
            if self.hop_overlay:
                hop_ms = sum(e['occurrences'] * phase_hops(e['participants'], e['kind'])
                             for e in ev) * .25 / 1000
        i = len(self.leaves)
        leaf = copy.deepcopy(spec)
        # Preserve logits_gemm substring: AISim uses it for prefill token shape.
        leaf[kind]['name'] = f'leaf_{i}_'+value['name']
        self.leaves.append(leaf)
        self.info.append(dict(communication=kind in COMM, calls=calls, hop_ms=hop_ms))
        return ('leaf', i)

    @classmethod
    def _contains_comm(cls, obj):
        if isinstance(obj, dict):
            return bool(COMM.intersection(obj)) or any(cls._contains_comm(v) for v in obj.values())
        return isinstance(obj, list) and any(cls._contains_comm(v) for v in obj)

    def fold(self, times):
        def walk(node):
            kind, value = node
            if kind == 'leaf':
                return times[value]
            xs = [walk(n) for n in value]
            return max(xs, default=0.) if kind == 'max' else math.fsum(xs)
        return walk(self.tree)

    def timing(self, times, *, efficiency=1., startup_us=0., legacy=False,
               include_hops=True):
        if not 0 < efficiency <= 1 or startup_us < 0:
            raise ValueError('Invalid communication sensitivity')
        if legacy:
            if efficiency != 1 or startup_us:
                raise ValueError('Legacy is only the frozen baseline')
            return self.fold(times) + sum(i['hop_ms'] for i in self.info)
        adjusted = [t / efficiency + i['calls'] * startup_us / 1000
                    + (i['hop_ms'] if include_hops else 0.)
                    if i['communication'] else t for t, i in zip(times, self.info)]
        return self.fold(adjusted)


class NativeLeaves:
    """Pinned native ad-hoc operator API; parity-checked against full engine."""
    def __init__(self, config, graph, phase, hop_overlay):
        import aisimulate_core
        from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json
        self.phase = phase
        path = config['systems_paths'][0]
        spec = build_engine_spec_json(graph, model_path=config['model'],
            system=config['system'], backend=config['backend'],
            backend_version=config['backend_version'], kv_block_size=None,
            systems_path=path, nextn=3 if phase == 'decode' else 0,
            database_mode='SOL', shared_layer=False)
        self.handle = EngineHandle(aisimulate_core.engine_spec_bincode_from_json(spec), systems_path=path)
        specs = json.loads(spec)['generation_ops' if phase == 'decode' else 'context_ops']
        self.tree = OperationTree(specs, hop_overlay)

    def times(self, batch):
        prefill = self.phase == 'prefill'
        rows = self.handle.evaluate_ops_json(self.tree.ops_json, is_context=prefill,
            batch_size=batch if prefill else batch*4,
            s=1024 if prefill else 262145, prefix=262144 if prefill else 0)
        by_name = {r[0]: r[1] for r in rows}
        times = [by_name[next(iter(s.values()))['name']] for s in self.tree.leaves]
        if not all(math.isfinite(t) and t >= 0 for t in times):
            raise ValueError('Invalid native leaf timing')
        return times
