#!/usr/bin/env python3
"""Small local contract check, without simulation or inference.

Drafts are curated directly since GPU qualification; qualification status lives
in docs/RESOLUTION256_STATUS.md.
"""
import json
import math
from pathlib import Path
import yaml
from mriseqbench.models import Task
from mriseqbench.evaluation.s3_sequence import prescribed_mask

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
profile = json.loads((HERE / 'profile.json').read_text())
assert profile['paid_dispatch_allowed'] is False
assert len(profile['tasks']) == 23
checked = []
arithmetic = {}
for row in profile['tasks']:
    assert not row['paid_dispatch_allowed']
    name = row['id'].split('@')[0]
    if row['status'] == 'protocol_redesign_required':
        assert row['artifact'].startswith('blocked/')
        continue
    draft = yaml.safe_load((HERE / row['artifact']).read_text())
    doc = Task.model_validate(draft)
    assert doc.version == 2 and doc.sequence.matrix == (256, 256)
    assert doc.sequence.fov_mm == (200, 200)
    if row.get('retained_ky_indices'):
        mask = prescribed_mask(name, (256, 256), doc.sequence.sampling.model_dump())
        assert mask.any(axis=0).nonzero()[0].tolist() == row['retained_ky_indices']
        assert len(row['retained_ky_indices']) == 86
    seq = doc.sequence
    tr = seq.tr_ms
    units = None
    if name in {'gre_t1w', 'gre_t2star', 'spin_echo_t2w', 'bssfp'}:
        units = 256
    elif name in {'radial_readout', 'ute_radial'}:
        units = seq.readout.spokes
        assert units >= math.ceil(math.pi * 256 / (1 if name == 'ute_radial' else 2))
    elif name == 'spiral_gre':
        units = seq.readout.interleaves
        assert units * seq.readout.turns >= 256 / 2
    elif name in {'tse_t2w', 'grase', 'propeller_tse'}:
        units = seq.readout.shots
        if name != 'propeller_tse':
            assert units * seq.readout.echo_train_length == 256
    elif seq.sampling is not None:
        mask = prescribed_mask(name, (256, 256), seq.sampling.model_dump())
        units = int(mask.any(axis=0).sum())
    if units is not None and tr is not None:
        nominal_s = units * (tr.target if tr.target is not None else tr.max) / 1000
        assert nominal_s <= seq.duration_s, (name, nominal_s, seq.duration_s)
        if name == 'gre_t1w':
            prepared_reference_s = (256 + 128) * .024
            assert prepared_reference_s <= seq.duration_s
        arithmetic[row['id']] = {'excitation_count': units, 'nominal_train_envelope_s': nominal_s, 'draft_duration_s': seq.duration_s}
        if name == 'gre_t1w':
            arithmetic[row['id']].update(dummy_TR_count=128, dummy_preparation_s=128 * .024, prepared_reference_s=prepared_reference_s)
    checked.append(row['id'])
assert len(checked) == 18
receipt = {
    'status': 'local_schema_and_arithmetic_checks_passed',
    'candidate_count': len(checked), 'blocked_count': 5, 'inventory_count': 23,
    'checked_task_refs': checked,
    'native_or_simulation_runs': 0, 'model_calls': 0,
        'nominal_count_TR_bounds': arithmetic,
}
(HERE / 'audit-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
