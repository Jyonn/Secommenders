import importlib
import sys
import types


def _import_scheduler_without_optional_deps(monkeypatch):
    yaml_module = types.ModuleType('yaml')
    yaml_module.safe_load = lambda text: {}
    monkeypatch.setitem(sys.modules, 'yaml', yaml_module)

    notificator_module = types.ModuleType('notificator')
    notificator_module.Notificator = object
    monkeypatch.setitem(sys.modules, 'notificator', notificator_module)

    sys.modules.pop('scheduler', None)
    return importlib.import_module('scheduler')


def test_scheduler_plan_argument_accepts_multiple_and_repeated_values(monkeypatch):
    scheduler = _import_scheduler_without_optional_deps(monkeypatch)

    args = scheduler.build_arg_parser().parse_args(
        ['--plan', 'config/a.yaml', 'config/b.yaml', '--plan', 'config/c.yaml']
    )

    assert args.plan == [['config/a.yaml', 'config/b.yaml'], ['config/c.yaml']]
