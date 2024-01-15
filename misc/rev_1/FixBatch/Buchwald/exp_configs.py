benchmark = [
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "GP",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_COSTS_GP_NEI",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "GP",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_NORMAL_GP_NEI",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,  # 300
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_COSTS_GP_GIBBON",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_NORMAL_GP_GIBBON",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "RF",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_COSTS_RF_NEI",
    },
    {
        "dataset": "buchwald",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 10,
        "batch_size": 5,
        "max_batch_cost": 800.0,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "RF",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BUCHWALD_NORMAL_RF_NEI",
    },
]