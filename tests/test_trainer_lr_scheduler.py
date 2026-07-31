from types import SimpleNamespace

import torch

from trainer import Trainer


def _trainer(*, scheduler='constant', warmup_ratio=0.0, epochs=2):
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        accumulate_batch=1,
        epochs=epochs,
        lr_scheduler=scheduler,
        warmup_ratio=warmup_ratio,
    )
    trainer.train_loader = [None] * 5
    trainer._pnt = lambda _: None
    return trainer


def test_constant_scheduler_without_warmup_preserves_legacy_behavior():
    trainer = _trainer(epochs=0)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    assert trainer.build_lr_scheduler(optimizer) is None
    assert optimizer.param_groups[0]['lr'] == 1e-4


def test_cosine_scheduler_warms_up_then_decays_per_optimizer_step():
    trainer = _trainer(scheduler='cosine', warmup_ratio=0.2, epochs=2)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = trainer.build_lr_scheduler(optimizer)

    initial_lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    warmed_lr = optimizer.param_groups[0]['lr']
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]['lr']

    assert initial_lr < warmed_lr
    assert final_lr < warmed_lr
