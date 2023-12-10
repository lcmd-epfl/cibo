benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15, #15
        "batch_size": 5,
        "max_batch_cost": 20.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "buget_schedule": "adaptive_2",
    }
]