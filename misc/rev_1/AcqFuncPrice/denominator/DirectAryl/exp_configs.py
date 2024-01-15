benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "NEI",
        "label": "BMS_COST_GP_NEI",
        "cost_mod": "denominator",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "NEI",
        "label": "BMS_NORMAL_GP_NEI",
        "cost_mod": "denominator",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "label": "BMS_COST_GP_GIBBON",
        "cost_mod": "denominator",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "label": "BMS_NORMAL_GP_GIBBON",
        "cost_mod": "denominator",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "RF",
        "acq_func": "NEI",
        "label": "BMS_COST_RF_NEI",
        "cost_mod": "denominator",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "RF",
        "acq_func": "NEI",
        "label": "BMS_NORMAL_RF_NEI",
        "cost_mod": "denominator",
    },
]