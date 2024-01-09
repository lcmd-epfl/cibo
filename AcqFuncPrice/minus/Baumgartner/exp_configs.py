benchmark = [
    {
        "dataset": "baumgartner",
        "init_strategy": "worst_ligand_and_more",
        "cost_aware": True,
        "n_runs": 5,
        "n_iter": 20,
        "batch_size": 4,
        "ntrain": 200,
        "prices": "update_all_when_bought",
        "surrogate": "GP",
        "acq_func": "NEI",
        "label": "BAUM_COST_GP_NEI",
        "cost_mod": "minus",
    }
]