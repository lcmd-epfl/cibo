benchmark = [
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 10,  # 15
        "batch_size": 5,
        "max_batch_cost": 10.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "acq_func": "NEI",
        "buget_schedule": "adaptive_1",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 10,  # 15
        "batch_size": 5,
        "max_batch_cost": 10.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "acq_func": "NEI",
        "buget_schedule": "adaptive_2",
    },
]
