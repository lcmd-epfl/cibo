benchmark = [
    {
        "dataset": "BMS",
        "init_strategy": "random_ligands",
        "cost_aware": False,
        "n_runs": 100,
        "n_iter": 20,
        "batch_size": 5,
        "ntrain": 144,
        "prices": "update_ligand_when_used",
        "surrogate": "GP",
        "acq_func": "NEI",
        "label": "BMS_BO_GP_NEI_1.0",
        "cost_mod": "minus",
        "cost_weight": 1.0,
    },
]
