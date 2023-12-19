benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 150.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BMS_COSTS_GP_NEI",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 1000.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BMS_NORMAL_GP_NEI",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 150.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "decreasing",
        "label": "BMS_COSTS_GP_GIBBON",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 1000.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "decreasing",
        "label": "BMS_NORMAL_GP_GIBBON",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 150.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "RF",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BMS_COSTS_RF_NEI",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 1000.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "RF",
        "acq_func": "NEI",
        "buget_schedule": "decreasing",
        "label": "BMS_NORMAL_RF_NEI",
    },
]
