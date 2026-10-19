"""Agreement at saved observations; no fitting, new timings or interpolation."""
import hashlib
import json
import math
from statistics import mean, median

from compare import OUT, functions
from analyze import save, csvout


def output_candidates(g, recipes, de, pe):
    """Retain winning recipe provenance for the layout-restricted envelope."""
    rows = []
    r = g['input_output_ratio']
    for q in recipes:
        for p in q['decode_frontier']:
            s = p['speed'] * de
            c = p['capacity']
            if q['disaggregated']:
                c = min(c*q['decode_chips']/q['allocated_chips'],
                        q['prefill_rate']*pe*q['prefill_chips']/(r*s*q['allocated_chips']))
            else:
                c /= 1+r*s*c/(q['prefill_rate']*pe)
            rows.append(dict(speed=s, capacity=s*c,
                             recipe_id=q['representative_result_id'], batch=p['batch']))
    return rows


def summarize(rows, comparator):
    errors = [p[comparator+'_signed_error'] for p in rows
              if p[comparator+'_signed_error'] is not None]
    return dict(count=len(rows), evaluated_count=len(errors),
                mean_absolute_error=mean(abs(e) for e in errors) if errors else None,
                median_absolute_error=median(abs(e) for e in errors) if errors else None,
                mean_signed_error=mean(errors) if errors else None,
                max_absolute_error=max(map(abs, errors)) if errors else None,
                zero_predictions=sum(p[comparator] == 0 for p in rows),
                within_20_percent=sum(abs(e) <= .2 for e in errors)/len(errors) if errors else None)


def step_curve_error(measured, predicted, edges):
    """Exact log-uniform error of two step curves where measurements cover s.

    This treats the measured envelope as a step function, not as ground truth
    between benchmark observations. Empty measured ranges are not observations.
    """
    terms = []
    for a, b in zip(edges, edges[1:]):
        s = math.sqrt(a*b)
        m = measured(s)
        if m > 0:
            terms.append((math.log(b/a), m, predicted(s)))
    width = math.fsum(w for w, _, _ in terms)
    return dict(log_speed_coverage=min(1., width/math.log(edges[-1]/edges[0])),
                mean_absolute_error=math.fsum(w*abs(p/m-1) for w, m, p in terms)/width if width else None,
                normalized_absolute_area=math.fsum(w*abs(p-m) for w, m, p in terms)/math.fsum(w*m for w, m, p in terms) if width else None)


def main():
    data = json.loads((OUT/'curves.json').read_text())
    layouts = json.loads((OUT/'layout_restricted.json').read_text())
    summaries, points = [], []
    for g in data['groups']:
        recipes = [q for q in layouts['recipes']
                   if (q['model'], q['hardware']) == (g['model'], g['hardware'])]
        for de, pe in [(.5, .5), (.5, 1.), (1., 1.)]:
            candidates = output_candidates(g, recipes, de, pe)
            for metric in ['output', 'capacity']:
                measured, restricted, ideal, _, _ = functions(g, recipes, de, pe, metric)
                frontier = g['throughput_frontier'] if metric == 'output' else g['frontier']
                for band, lo, hi in [('index_band', 50., 400.),
                                     ('observed_range', g['observed_min_speed'], g['measured_max_speed'])]:
                    meta = dict(model=g['model'], hardware=g['hardware'], band=band,
                                low=lo, high=hi, decode_efficiency=de, prefill_efficiency=pe, metric=metric)
                    summary = dict(**meta, cohorts={})
                    for cohort, observations in [('frontier', frontier), ('all_observations', g['observations'])]:
                        selected = []
                        for p in observations:
                            s = p['speed']
                            if not lo <= s <= hi:
                                continue
                            actual = p['output_tps_chip'] if metric == 'output' else p['capacity']
                            prediction = restricted(s) if recipes else None
                            row = dict(**meta, cohort=cohort, result_id=p['id'], speed=s,
                                       measured=actual, restricted=prediction, ideal=ideal(s),
                                       restricted_signed_error=prediction/actual-1 if prediction is not None else None,
                                       ideal_signed_error=ideal(s)/actual-1,
                                       winning_recipe_id=None, winning_batch=None, winning_native_speed=None)
                            if metric == 'output' and recipes:
                                winner = max((x for x in candidates if x['speed'] >= s),
                                             key=lambda x: x['capacity'], default=None)
                                assert math.isclose(winner['capacity'] if winner else 0., prediction, rel_tol=1e-12)
                                if winner:
                                    row.update(winning_recipe_id=winner['recipe_id'], winning_batch=winner['batch'],
                                               winning_native_speed=winner['speed']/de)
                            selected.append(row)
                        points.extend(selected)
                        summary['cohorts'][cohort] = {c: summarize(selected, c) for c in ['restricted', 'ideal']}
                    # Both output curves are step functions. Include every jump;
                    # use no plotting grid, smoothing or numerical extrapolation.
                    summary['step_curve'] = None
                    if metric == 'output' and recipes:
                        edges = sorted({lo, hi, *(p['speed'] for p in frontier if lo < p['speed'] < hi),
                                        *(x['speed'] for x in candidates if lo < x['speed'] < hi)})
                        summary['step_curve'] = step_curve_error(measured, restricted, edges)
                    summaries.append(summary)
    sources = {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
               for name in ['curves.json', 'layout_restricted.json']}
    save('point_agreement.json', dict(sources=sources, summaries=summaries, point_csv='point_agreement.csv',
         definition='Mean of abs(predicted/measured - 1) at saved points; equal point weights. '
                    'Restricted predictions optimize over the union of admitted submitted layouts and batches, '
                    'not necessarily the layout of each observed point. No efficiency fitting.'))
    csvout('point_agreement.csv', points)
    csvout('point_agreement_summary.csv', [dict(**{k:v for k,v in s.items() if k not in ['cohorts','step_curve']},
           cohort=cohort, comparator=comp, **values)
           for s in summaries for cohort, cs in s['cohorts'].items() for comp, values in cs.items()])
    print('Saved agreement for', len(points), 'point/scenario/metric combinations')


if __name__ == '__main__':
    main()
