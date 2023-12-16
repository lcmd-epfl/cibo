benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 60,
        "batch_size": 5,
        "max_batch_cost": 20.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "adaptive_2",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 100.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "constant",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 100.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "increasing",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 1000.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "decreasing",
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
        "buget_schedule": "constant",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "max_batch_cost": 100.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "adaptive",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "max_batch_cost": 50.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "adaptive",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 30,
        "batch_size": 5,
        "max_batch_cost": 150.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "adaptive",
    },
    {
        "dataset": "BMS",
        "init_strategy": "worst_ligand",
        "cost_aware": False,
        "n_runs": 5,
        "n_iter": 15,
        "batch_size": 5,
        "max_batch_cost": 500000.0,
        "ntrain": 200,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "GIBBON",
        "buget_schedule": "constant",
    },
]
