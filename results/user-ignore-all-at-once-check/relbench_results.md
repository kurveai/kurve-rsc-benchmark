# Kurve-RSC RelBench Results

- Task type: `classification`
- Single train period: `False`
- Model backend: `catboost`
- Train all at once: `True`
- Submission directory: `disabled`
- Passed: `1`
- Failed: `0`

| Task | Status | Duration (s) | Highlights | Log |
| --- | --- | ---: | --- | --- |
| `relbench_event_user_ignore` | `passed` | 241.711 | frozen_execution_plan: split=train cutoff=2012-06-20T00:00:00 records=10 features=1439<br>training_mode: all_at_once<br>feature_count: 1432<br>validation_metrics: {'accuracy': 0.877297565822156, 'average_precision': 0.623613097391179, 'f1': 0.5348399246704332, 'roc_auc': 0.8897750331938773}<br>test_metrics: {'accuracy': 0.8707865168539326, 'average_precision': 0.5272422485025313, 'f1': 0.5048923679060665, 'roc_auc': 0.7981532013862503} | `results/user-ignore-all-at-once-check/logs/relbench_event_user_ignore.log` |
