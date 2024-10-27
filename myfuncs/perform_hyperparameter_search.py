# perform_hyperparameter_search.py
# Hyper-parameter search using either Scikit-Optimize or BayesianOptimization

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

def perform_hyperparameter_search(train_gen, use_bayesian_optimization=False):
    '''
    Tune hyper-parameters using Scikit-Optimize or Bayesian Optimization.
    '''
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

    if use_bayesian_optimization:
        # Bayesian Optimization
        def objective_function(**params):
            return -objective(params, train_gen)

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
            random_state=0
        )

        optimizer.maximize(init_points=3, n_iter=5)

        best_params = optimizer.max['params']
        best_score = -optimizer.max['target']
    else:
        # Scikit-Optimize
        @use_named_args(space)
        def objective_fn(**params):
            return objective(params, train_gen)

        result = gp_minimize(func=objective_fn, dimensions=space, n_calls=5, n_random_starts=3, random_state=0)

        best_params = result.x
        best_score = -result.fun

        best_params = {
            'optimizer_type': best_params[0],
            'learning_rate': best_params[1],
            'weight_decay': best_params[2],
            'dropout': best_params[3],
            'beta_1': best_params[4],
            'beta_2': best_params[5],
            'decay': best_params[6],
            'momentum': best_params[7],
            'rho': best_params[8]
        }

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    return best_params
