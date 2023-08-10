hparams = [
    {
        "model": "LightGBM",
        "dataset": "mimic-iii",
        "task": "outcome",
        "max_depth": 5,
        "n_estimators": 50,
        "learning_rate": 0.1,
        "batch_size": 81920,
        "main_metric": "auprc",
    },
]