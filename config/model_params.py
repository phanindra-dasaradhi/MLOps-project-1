from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    'num_leaves': randint(20, 100),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 500),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,
    'cv': 2,
    'scoring': 'accuracy',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 2
}