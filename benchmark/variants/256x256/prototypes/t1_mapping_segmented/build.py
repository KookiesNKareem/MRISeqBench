#!/usr/bin/env python3
"""One local segmented waveform prototype. No MRI/native/model calls or release."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pypulseq as pp
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.preflight import preflight
from mriseqbench.evaluation.sar import evaluate_sar


def make_system(h):
    return pp.Opts(B0=h['field_strength_T'], max_grad=h['max_gradient_mT_per_m'],
                   grad_unit='mT/m', max_slew=h['max_slew_T_per_m_per_s'], slew_unit='T/m/s',
                   rf_dead_time=h['rf_dead_time_us'] * 1e-6,
                   rf_ringdown_time=h['rf_ringdown_time_us'] * 1e-6,
                   adc_dead_time=h['adc_dead_time_us'] * 1e-6,
                   grad_raster_time=h['gradient_raster_us'] * 1e-6,
                   rf_raster_time=h['rf_raster_us'] * 1e-6,
                   adc_raster_time=h['adc_raster_us'] * 1e-6,
                   block_duration_raster=h['block_raster_us'] * 1e-6)


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def main():
    h = yaml.safe_load((ROOT / 'benchmark/hardware/high_performance.yaml').read_text())
    system = make_system(h)
    seq = pp.Sequence(system)
    tr, te_nominal, n, fov = .0046, .0023, 256, .2
    rf = pp.make_block_pulse(np.deg2rad(35), duration=.0002,
                            delay=.0001, use='excitation', system=system)
    inv = pp.make_block_pulse(np.pi, duration=.001,
                             delay=.000105, use='inversion', system=system)
    rf_center = rf.delay + rf.shape_dur / 2
    inv_center = inv.delay + inv.shape_dur / 2
    read = pp.make_trapezoid('x', flat_area=n / fov, flat_time=.00256,
                            rise_time=.00012, fall_time=.00012, system=system)
    pre_x = pp.make_trapezoid('x', area=-read.area / 2 - .5 / fov,
                             duration=.0007, system=system)
    post_x = pp.make_trapezoid('x', area=-read.area - pre_x.area,
                              duration=.0007, system=system)
    adc = pp.make_adc(n, dwell=.000010, delay=read.rise_time, system=system)
    adc_zero_offset = adc.delay + (n / 2 + .5) * adc.dwell
    # Readout block start on10us raster gives actualTE nearTR/2, within old symmetry tolerance.
    read_start = round((te_nominal + rf_center - adc_zero_offset) / .00001) * .00001
    te_actual = read_start + adc_zero_offset - rf_center
    delay_pre = read_start - pp.calc_duration(rf) - pp.calc_duration(pre_x)
    assert delay_pre >= 0
    delay_post = tr - (read_start + pp.calc_duration(read) + pp.calc_duration(post_x))
    assert delay_post >= 0
    nav_zero_relative = read_start + adc_zero_offset
    events, segments = [], []
    for i in range(120):
        events.append((float(i), 'trigger', None))
    for contrast, ti in enumerate([.1, .4, 1.2]):
        for segment in range(8):
            j = 8 * contrast + segment
            trigger = 2. + 5. * j
            start = trigger + .6
            inversion_center = start + nav_zero_relative - ti
            inversion_start = inversion_center - inv_center
            assert abs(inversion_start / .00001 - round(inversion_start / .00001)) < 1e-7
            info = {'contrast_index': contrast, 'nominal_TI_ms': ti * 1000,
                    'segment_index': segment, 'imaging_ky_indices': list(range(32 * segment, 32 * (segment + 1))),
                    'trigger_s': trigger, 'train_start_s': start,
                    'inversion_center_s': inversion_center,
                    'navigator_kzero_s': start + nav_zero_relative,
                    'navigator': {'ky_index': 128, 'adc_samples': 256,
                                  'role': 'timing/phase diagnostic only; exclude from imaging coverage/fitting'},
                    'readout_count': 33, 'train_duration_s': 33 * tr}
            segments.append(info)
            events.extend([(inversion_start, 'inversion', info), (start, 'train', info)])
    clock = 0.
    adc_roles = []
    for time, kind, info in sorted(events, key=lambda e: e[0]):
        assert time >= clock - 1e-9, (kind, time, clock)
        if time - clock > 1e-9:
            seq.add_block(pp.make_delay(time - clock))
            clock = time
        if kind == 'trigger':
            seq.add_block(pp.make_trigger('physio1', duration=.00001, system=system))
            clock += .00001
        elif kind == 'inversion':
            seq.add_block(inv, pp.make_delay(.00114))
            clock += .00114
        else:
            ys = [128] + info['imaging_ky_indices']
            for rno, ky in enumerate(ys):
                phase = (rno % 2) * np.pi
                r, a = copy.copy(rf), copy.copy(adc)
                r.phase_offset = phase
                a.phase_offset = phase
                gy = pp.make_trapezoid('y', area=(ky - 128) / fov, duration=.0007, system=system)
                gy_rewind = pp.make_trapezoid('y', area=-gy.area, duration=.0007, system=system)
                seq.add_block(r)
                seq.add_block(pre_x, gy)
                seq.add_block(pp.make_delay(delay_pre))
                seq.add_block(read, a)
                seq.add_block(post_x, gy_rewind)
                seq.add_block(pp.make_delay(delay_post))
                adc_roles.append({'contrast': info['contrast_index'], 'segment': info['segment_index'],
                                  'navigator': rno == 0, 'ky_index': ky,
                                  'start_s': clock, 'rf_center_s': clock + rf_center,
                                  'expected_kzero_s': clock + nav_zero_relative,
                                  'phase_rad': phase})
                clock += tr
    if clock < 120:
        seq.add_block(pp.make_delay(120 - clock))
    timing_ok, errors = seq.check_timing()
    assert timing_ok, errors[:10]
    path = HERE / 'segmented256_t1_mapping_prototype.seq'
    seq.set_definition('Name', 'HELD segmented256 T1 prototype; no image qualification')
    seq.set_definition('FOV', [.2, .2, .008])
    seq.set_definition('PrototypeStatus', 'UNQUALIFIED_NOT_RELEASED')
    seq.write(str(path))

    # Parse the written file and obtain measured integrated k-space/timing, notlabels.
    measured = pp.Sequence(system)
    measured.read(str(path))
    k_adc, _, _, _, t_adc = measured.calculate_kspace()
    assert k_adc.shape[1] == 24 * 33 * 256
    lines = k_adc.reshape(3, 24 * 33, 256)
    t_lines = t_adc.reshape(24 * 33, 256)
    coverage, measurements = [], []
    for c in range(3):
        chosen = [i for i, role in enumerate(adc_roles) if role['contrast'] == c and not role['navigator']]
        q = (lines[:2, chosen] * fov).transpose(1, 2, 0).reshape(-1, 2)
        rounded = np.rint(q).astype(int)
        assert np.max(np.abs(q - rounded)) < 1e-3
        cells, counts = np.unique(rounded, axis=0, return_counts=True)
        assert len(cells) == 65536 and counts.min() == counts.max() == 1
        assert np.array_equal(cells.min(axis=0), [-128, -128])
        assert np.array_equal(cells.max(axis=0), [127, 127])
        coverage.append({'contrast_index': c, 'unique_imaging_points': len(cells),
                         'duplicates_after_explicit_navigator_exclusion': int(counts.sum() - len(cells)),
                         'q_min': cells.min(axis=0), 'q_max': cells.max(axis=0)})
    for index, role in enumerate(adc_roles):
        info = segments[role['contrast'] * 8 + role['segment']]
        win = (t_lines[index] - info['trigger_s']) * 1000
        assert win.min() >= 600 - 1e-6 and win.max() <= 800 + 1e-6
        # x=0 lies at the128th zero-based node of the specified even grid.
        zero = int(np.argmin(np.abs(lines[0, index])))
        actual_te_ms = (t_lines[index, zero] - role['rf_center_s']) * 1000
        assert abs(actual_te_ms - tr * 500) <= .01
        if role['navigator']:
            actual_ti_ms = (t_lines[index, zero] - info['inversion_center_s']) * 1000
            assert abs(actual_ti_ms - info['nominal_TI_ms']) <= .002
            measurements.append({'contrast_index': role['contrast'], 'segment_index': role['segment'],
                                 'actual_TI_ms': actual_ti_ms, 'actual_TE_ms': actual_te_ms,
                                 'ADC_window_ms': [float(win.min()), float(((t_lines[index + 32] - info['trigger_s']) * 1000).max())]})
    assert np.diff([i['inversion_center_s'] for i in segments]).min() >= 3
    case = Catalog(ROOT).resolve(Catalog(ROOT).get('tasks', 't1_mapping'), hardware_ref='high_performance@1')
    case['task']['sequence']['duration_s'] = 120
    hw = preflight(path, case)
    sar = evaluate_sar(path, h)
    assert all(c['passed'] for c in hw + sar), [c for c in hw + sar if not c['passed']]
    balance = {}
    for axis, gradient in zip('xyz', measured.get_gradients()):
        if gradient is None:
            continue
        integral = gradient.antiderivative()
        values = [float(integral(min(role['start_s'] + tr, gradient.x[-1])) - integral(max(role['start_s'], gradient.x[0]))) for role in adc_roles]
        balance[axis] = max(map(abs, values))
        assert balance[axis] < .01 / fov
    grad = {}
    for axis, waveform in zip('xyz', measured.waveforms()):
        if waveform.size:
            times, g = waveform
            grad[axis] = {'peak_mT_per_m': float(np.max(np.abs(g)) / system.gamma * 1000),
                          'peak_slew_T_per_m_per_s': float(np.max(np.abs(np.diff(g) / np.diff(times))) / system.gamma)}
    report = {'status': 'waveform_geometry_timing_SAR_prototype_only',
              'qualified_for_agent_dispatch': False, 'current_s4_at1_compatible': False,
              'MRI_signal_simulations': 0, 'native_evaluations': 0, 'model_calls': 0,
              'matrix': [256, 256], 'FOV_mm': [200, 200], 'nominal_pixel_mm': .78125,
              'contrast_count': 3, 'segments_per_contrast': 8, 'imaging_lines_per_segment': 32,
              'extra_navigator_lines_per_segment': 1, 'total_ADC_samples_including_navigators': len(t_adc),
              'imaging_ADC_samples': 3 * 65536, 'navigator_ADC_samples': 24 * 256,
              'duration_s': measured.duration()[0], 'TR_ms': tr * 1000, 'nominal_TE_ms': te_nominal * 1000,
              'actual_TE_ms': te_actual * 1000,
              'min_inversion_preparation_separation_s': float(np.diff([i['inversion_center_s'] for i in segments]).min()),
              'max_absolute_readout_TR_net_gradient_area_cycles_per_m': balance, 'hardware_measurements': grad, 'hardware_checks': hw, 'synthetic_benchmark_SAR_checks': sar,
              'coverage': coverage, 'measured_segments': measurements,
              'sequence_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'limits': [
                  'Block180degree inversion is not clinically robust or adiabatic; no RF/B0/B1response qualification.',
                  '35degree nonselective readout, no slice profile or catalyzation/dummyTRs qualified.',
                  'Same navigator-based nominalTI does not make every peripheral ky node have thatTI; intra-train dynamics must enter reference simulation.',
                  'All navigator lines explicitly excluded from imaging coverage and map fit; their utility for phase correction is unqualified.',
                  'Repeated preparations may leave tissue-dependent state even with >=3RR separation. No equilibrium or fitting accuracy claim.',
                  'CurrentS4@1 assumes one whole image per trigger; it cannot assemble this segmented acquisition.',
                  'Known-PD/known-T2 conditionalT1 fit has not been run or validated; nativeGPUimage replay and independent high-detail target are pending.',
              ]}
    (HERE / 'segment-plan.json').write_text(json.dumps({'segments': segments, 'adc_roles': adc_roles}, indent=2) + '\n')
    (HERE / 'waveform-audit.json').write_text(json.dumps(report, indent=2, default=jsonable) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'duration_s', 'TR_ms', 'actual_TE_ms', 'total_ADC_samples_including_navigators', 'min_inversion_preparation_separation_s', 'hardware_measurements']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (HERE / 'construction-failure.txt').write_text(traceback.format_exc())
        raise
