"""Independent checks of exported control waveforms, timing and k-space."""

import numpy as np
import pypulseq as pp


def validate_readout(path, kind):
    assert kind in {"gradient_echo", "spin_echo"}
    seq = pp.Sequence()
    seq.read(str(path))
    spin_echo = kind == "spin_echo"
    te = 0.012 if spin_echo else 0.008
    rfs, clock = [], 0.0
    for index in seq.block_events:
        rf = seq.get_block(index).rf
        if rf is not None:
            assert np.allclose(rf.signal, rf.signal[0], rtol=0, atol=1e-6)
            rfs.append((clock + rf.delay + rf.shape_dur / 2, rf))
        clock += seq.block_durations[index]
    assert len(rfs) == (2 if spin_echo else 1), "RF count"
    assert rfs[0][1].use == "excitation", "excitation metadata"
    expected = [np.pi / 2, np.pi] if spin_echo else [np.pi / 6]
    for (_, rf), flip in zip(rfs, expected, strict=True):
        assert np.isclose(
            2 * np.pi * abs(rf.signal[0]) * rf.shape_dur, flip, atol=1e-6
        ), "RF flip"
    if spin_echo:
        assert rfs[1][1].use == "refocusing", "refocusing metadata"
        assert np.isclose(rfs[1][0] - rfs[0][0], te / 2, atol=1e-9), "refocusing timing"
        assert np.isclose(rfs[1][1].phase_offset, np.pi / 2, atol=1e-6), (
            "refocusing phase"
        )
    times, fp = seq.adc_times()
    assert len(times) == 32, "ADC count"
    assert np.allclose(np.diff(times), 1e-4, atol=1e-10), "ADC dwell"
    assert np.allclose(fp, 0, atol=1e-9), "ADC frequency/phase"
    measured_te = float(np.mean(times) - rfs[0][0])
    assert np.isclose(measured_te, te, atol=1e-9), "echo timing"
    assert np.isclose(seq.duration()[0], 0.2 if spin_echo else 0.1, atol=1e-9), "TR"
    k_adc = seq.calculate_kspace()[0]
    assert np.allclose(k_adc[0], (np.arange(32) - 15.5) / 0.2, atol=1e-4), (
        "readout k-space"
    )
    assert np.allclose(k_adc[1:], 0, atol=1e-9), "unexpected gradient axes"
    return {
        "kind": kind,
        "te_s": measured_te,
        "tr_s": float(seq.duration()[0]),
        "rf_count": len(rfs),
        "adc_count": len(times),
        "fov_m": 0.2,
        "k_adc_x_per_m": k_adc[0].tolist(),
    }
