from typing import Any

from carte_ai import CARTEClassifier, CARTERegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor


def get_model(type: str,
              objective: str,
              device: str) -> Any:
    """
    device: cpu if cpu, cuda if gpu -> CARTE takes gpu as input
    """
    if objective == "classification":
        if type == "tabpfn":
            model = TabPFNClassifier(n_estimators=1,
                                     fit_mode='fit_preprocessors',
                                     device=device)
        elif type == "carte":
            if device == 'cuda':
                device = 'gpu'
            model = CARTEClassifier(disable_pbar=False,
                                    max_epoch=1,
                                    num_model=1,
                                    device=device)
        else:
            raise ValueError(f"Model type {type} not supported.")
    elif objective == "regression":
        if type == "tabpfn":
            model = TabPFNRegressor(n_estimators=1,
                                    fit_mode='fit_preprocessors',
                                    device=device)
        elif type == "carte":
            if device == 'cuda':
                device = 'gpu'
            model = CARTERegressor(disable_pbar=False,
                                   max_epoch=1,
                                   num_model=1,
                                   device=device)
        else:
            raise ValueError(f"Model type {type} not supported.")
    else:
        raise ValueError(f"Objective {objective} not supported.")

    return model
