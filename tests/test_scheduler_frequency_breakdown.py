import json

from scripts.scheduler_frequency_breakdown import (
    choose_batch_size,
    has_per_target_records,
    load_plan_experiments,
)


def test_choose_batch_size_prefers_persisted_successful_batch():
    experiment = {'batch_size_cap': 32}
    state_experiment = {'batch_size': 8}

    assert choose_batch_size(experiment, state_experiment, 64) == 8


def test_choose_batch_size_uses_divisor_of_effective_batch():
    experiment = {'batch_size_cap': 10}

    assert choose_batch_size(experiment, None, 64) == 8


def test_load_plan_supports_args_and_commands(tmp_path):
    plan_path = tmp_path / 'plan.yaml'
    plan_path.write_text(
        '\n'.join([
            'name: example',
            'experiments:',
            '  - name: by_args',
            '    args:',
            '      data: ras1',
            '      model: scratch',
            '      task_type: uid',
            '      repr_type: uid',
            '  - name: by_command',
            '    command: python trainer.py --data ras2 --model scratch --task_type uid --repr_type uid',
        ])
        + '\n'
    )

    _, experiments = load_plan_experiments(plan_path)

    assert [experiment['name'] for experiment in experiments] == ['by_args', 'by_command']
    assert experiments[1]['base_args']['data'] == 'ras2'


def test_only_reuses_analysis_with_per_target_records(tmp_path):
    path = tmp_path / 'frequency.json'
    path.write_text(json.dumps({'records': [{'raw_item_id': 'i1', 'rank': 1}]}))

    assert has_per_target_records(path)

    path.write_text(json.dumps({'buckets': {}}))
    assert not has_per_target_records(path)
