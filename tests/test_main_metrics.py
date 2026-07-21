from trainer import Trainer


def _trainer(main_metric):
    trainer = object.__new__(Trainer)
    trainer.config = type('Config', (), {'main_metric': main_metric})()
    return trainer


def test_main_metric_pipe_parsing_and_direction():
    trainer = _trainer(' loss | ndcg@10 | loss ')

    assert trainer._main_metric_names() == ['loss', 'ndcg@10']
    assert trainer._main_metric_higher_is_better('loss') is False
    assert trainer._main_metric_higher_is_better('alignment_loss') is False
    assert trainer._main_metric_higher_is_better('ndcg@10') is True


def test_any_main_metric_improvement_updates_patience_bests():
    trainer = _trainer('loss|ndcg@10')
    best_metrics = {'loss': 0.8, 'ndcg@10': 0.3}

    improved = trainer._update_main_metric_bests(
        {'loss': 0.9, 'ndcg@10': 0.31},
        best_metrics,
    )

    assert improved == ['ndcg@10']
    assert best_metrics == {'loss': 0.8, 'ndcg@10': 0.31}
