dl_best_hparams = [
    {
        "model": "BiLSTM",
        "dataset": "mimic-iii",
        "task": "outcome",
        "epochs": 2,
        "patience": 10,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 59,
        "hidden_dim": 128,
        "output_dim": 1,
    },
]