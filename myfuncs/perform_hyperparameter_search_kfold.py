# perform_hyperparameter_search_kfold.py
# Hyper-parameter search using either Scikit-Optimize or BayesianOptimization
# Uses k-fold which makes it run out of memory

import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool
np.random.seed(42)

import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
from myfuncs.objective import objective

def perform_hyperparameter_search(train_gen, use_bayesian_optimization=False, n_calls=25, n_random_starts=5, n_splits=3):
    space = [
        Categorical(['Adam', 'SGD', 'RMSprop'], name='optimizer_type'),
        Real(1e-6, 1e-3, prior='log-uniform', name='learning_rate'),
        Real(1e-6, 1e-3, prior='log-uniform', name='weight_decay'),
        Real(0.3, 0.7, name='dropout'),
        Real(0.85, 0.95, name='beta_1'),
        Real(0.98, 0.999, name='beta_2'),
        Real(1e-6, 1e-4, prior='log-uniform', name='decay'),
        Real(0.0, 1.0, name='momentum'),
        Real(0.8, 0.99, name='rho')
    ]

    def objective_function(**params):
        return -objective(params, train_gen, n_splits=n_splits)

    if use_bayesian_optimization:
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds={
                'optimizer_type': space[0].categories,
                'learning_rate': (1e-6, 1e-3),
                'weight_decay': (1e-6, 1e-3),
                'dropout': (0.3, 0.7),
                'beta_1': (0.85, 0.95),
                'beta_2': (0.98, 0.999),
                'decay': (1e-6, 1e-4),
                'momentum': (0.0, 1.0),
                'rho': (0.8, 0.99)
            },
            random_state=0,
            verbose=2
        )
        optimizer.maximize(init_points=n_random_starts, n_iter=n_calls)
        best_params = optimizer.max['params']
        best_score = -optimizer.max['target']
    else:
        @use_named_args(space)
        def objective_fn(**params):
            return objective_function(**params)

        result = gp_minimize(func=objective_fn, dimensions=space, n_calls=n_calls, n_random_starts=n_random_starts, random_state=0)
        best_params = {
            'optimizer_type': result.x[0],
            'learning_rate': result.x[1],
            'weight_decay': result.x[2],
            'dropout': result.x[3],
            'beta_1': result.x[4],
            'beta_2': result.x[5],
            'decay': result.x[6],
            'momentum': result.x[7],
            'rho': result.x[8]
        }
        best_score = -result.fun

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    return best_params

