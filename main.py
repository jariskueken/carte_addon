import argparse
import logging
import openml
import timeit
import yaml
import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')

from typing import Any

from util.preprocessing import split, preprocess
from util.metric import root_mean_squared_error_metric, auc_metric
from util.model import get_model
from util.task import get_task, generate_train_test_splits


logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TabPFNMixModel on a dataset")
    parser.add_argument("-c", "--config",
                        type=str,
                        help="Path to the config file",
                        required=True)
    return parser.parse_args()


def read_config(cfg_path) -> dict[str, Any]:
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def main(cfg: dict[str, Any]):
    classification_tasks = cfg['tasks']['classification']
    regression_tasks = cfg['tasks']['regression']
    num_splits = cfg['cv']['num_splits']
    device = cfg['model']['device']
    # model_type = cfg['model']['type']

    if cfg['tasks']['task_type'] == 'classification':
        logger.info('Running classification tasks')
        for model_type in ['tabpfn', 'carte']:
            logger.info('\n' + '*' * 50 + '\n' + f'Running for {model_type} model' + '\n' + '*' * 50)
            for task_id in classification_tasks:
                data_df, dataset_name, target = get_task(task_id)
                logger.info('\n' + '#' * 50 + '\n' + f'Running for {dataset_name} dataset' + '\n' + '#' * 50)

                score_rows = []
                stat_rows = []

                for split_num in range(num_splits):
                    train_df, test_df = generate_train_test_splits(data_df, task_id, split_num)
                    # Do something with the train and test dataframes
                    train_X, train_y, test_X, test_y = split(train_df, test_df, target)

                    train_samples = train_X.shape[0]
                    test_samples = test_X.shape[0]

                    model = get_model(model_type, 'classification', device=device)

                    num_categoricals = len([col for col in train_X.columns if train_X[col].dtype == 'object' or train_X[col].dtype.name == 'category'])
                    num_numericals = len(list(train_X.columns)) - num_categoricals

                    logger.info(f"Number of categorical columns: {num_categoricals}")
                    logger.info(f"Number of numerical columns: {num_numericals}")
                    logger.info(f"Number of rows in training data: {train_X.shape[0]}")
                    logger.info(f"Number of rows in test data: {test_X.shape[0]}")

                    logger.info(f"Preprocessing for {model.__class__.__name__} model")
                    # preprocess the data
                    start = timeit.default_timer()
                    train_X, train_y, test_X, test_y = preprocess(train_X,
                                                                train_y,
                                                                test_X,
                                                                test_y,
                                                                model)
                    stop = timeit.default_timer()

                    preprocessing_time = round((stop - start), 2)
                    logger.info(f"Preprocessing took {preprocessing_time} seconds")

                    # fit the model
                    logger.info(f"Fitting {model.__class__.__name__} model")

                    start = timeit.default_timer()
                    model.fit(train_X, train_y)
                    stop = timeit.default_timer()

                    fitting_time = round((stop - start), 2)
                    logger.info(f"Training took {fitting_time} seconds")

                    # predict
                    logger.info(f"Predicting with {model.__class__.__name__} model")
                    start = timeit.default_timer()
                    preds = model.predict_proba(test_X)
                    stop = timeit.default_timer()

                    prediction_time = round((stop - start), 2)
                    logger.info(f"Prediction took {prediction_time} seconds")

                    # accuracy
                    score = auc_metric(test_y.to_numpy(), preds)
                    logger.info(f"AUC: {score.item()}")

                    # store the results
                    score_rows.append({'dataset': dataset_name,
                                    'cv_split': split_num,
                                    'score': score.item(),
                                    'model': model_type})
                    stat_rows.append({'dataset': dataset_name,
                                    'cv_split': split_num,
                                    'num_categoricals': num_categoricals,
                                    'num_numericals': num_numericals,
                                    'num_train_samples': train_samples,
                                    'num_test_samples': test_samples,
                                    'model': model_type,
                                    'preprocessing_time': preprocessing_time,
                                    'fitting_time': fitting_time,
                                    'prediction_time': prediction_time})

                score_df = pd.DataFrame(score_rows)
                stat_df = pd.DataFrame(stat_rows)

                if not os.path.isfile(cfg['storage']['classification_scores_path']):
                    score_df.to_csv(cfg['storage']['classification_scores_path'], index=False)
                else:
                    score_df.to_csv(cfg['storage']['classification_scores_path'], mode='a', header=False, index=False)

                if not os.path.isfile(cfg['storage']['classification_stats_path']):
                    stat_df.to_csv(cfg['storage']['classification_stats_path'], index=False)
                else:
                    stat_df.to_csv(cfg['storage']['classification_stats_path'], mode='a', header=False, index=False)

    elif cfg['tasks']['task_type'] == 'regression':
        logger.info('Running regression tasks')

        for model_type in ['tabpfn', 'carte']:
            logger.info('\n' + '*' * 50 + '\n' + f'Running for {model_type} model' + '\n' + '*' * 50)
            for task_id in regression_tasks:
                data_df, dataset_name, target = get_task(task_id)
                logger.info('\n' + '#' * 50 + '\n' + f'Running for {dataset_name} dataset' + '\n' + '#' * 50)

                score_rows = []
                stat_rows = []

                for split_num in range(num_splits):
                    train_df, test_df = generate_train_test_splits(data_df, task_id, split_num)
                    # Do something with the train and test dataframes
                    train_X, train_y, test_X, test_y = split(train_df, test_df, target)

                    train_samples = train_X.shape[0]
                    test_samples = test_X.shape[0]

                    model = get_model(model_type, 'regression', device=device)

                    num_categoricals = len([col for col in train_X.columns if train_X[col].dtype == 'object' or train_X[col].dtype.name == 'category'])
                    num_numericals = len(list(train_X.columns)) - num_categoricals

                    logger.info(f"Number of categorical columns: {num_categoricals}")
                    logger.info(f"Number of numerical columns: {num_numericals}")
                    logger.info(f"Number of rows in training data: {train_X.shape[0]}")
                    logger.info(f"Number of rows in test data: {test_X.shape[0]}")

                    logger.info(f"Preprocessing for {model.__class__.__name__} model")
                    # preprocess the data
                    start = timeit.default_timer()
                    train_X, train_y, test_X, test_y = preprocess(train_X,
                                                                train_y,
                                                                test_X,
                                                                test_y,
                                                                model)
                    stop = timeit.default_timer()

                    preprocessing_time = round((stop - start), 2)
                    logger.info(f"Preprocessing took {preprocessing_time} seconds")

                    # fit the model
                    logger.info(f"Fitting {model.__class__.__name__} model")

                    start = timeit.default_timer()
                    model.fit(train_X, train_y)
                    stop = timeit.default_timer()

                    fitting_time = round((stop - start), 2)
                    logger.info(f"Training took {fitting_time} seconds")

                    # predict
                    logger.info(f"Predicting with {model.__class__.__name__} model")
                    start = timeit.default_timer()
                    preds = model.predict(test_X)
                    stop = timeit.default_timer()

                    print(preds)
                    prediction_time = round((stop - start), 2)
                    logger.info(f"Prediction took {prediction_time} seconds")

                    print(test_y.to_numpy())
                    print(test_y.to_numpy().shape)
                    print(preds.shape)
                    # accuracy
                    score = root_mean_squared_error_metric(test_y.to_numpy(), preds)
                    logger.info(f"RMSE: {score}")

                    # store the results
                    score_rows.append({'dataset': dataset_name,
                                    'cv_split': split_num,
                                    'score': score.item(),
                                    'model': model_type})
                    stat_rows.append({'dataset': dataset_name,
                                    'cv_split': split_num,
                                    'num_categoricals': num_categoricals,
                                    'num_numericals': num_numericals,
                                    'num_train_samples': train_samples,
                                    'num_test_samples': test_samples,
                                    'model': model_type,
                                    'preprocessing_time': preprocessing_time,
                                    'fitting_time': fitting_time,
                                    'prediction_time': prediction_time})

                score_df = pd.DataFrame(score_rows)
                stat_df = pd.DataFrame(stat_rows)

                if not os.path.isfile(cfg['storage']['regression_scores_path']):
                    score_df.to_csv(cfg['storage']['regression_scores_path'], index=False)
                else:
                    score_df.to_csv(cfg['storage']['regression_scores_path'], mode='a', header=False, index=False)

                if not os.path.isfile(cfg['storage']['regression_stats_path']):
                    stat_df.to_csv(cfg['storage']['regression_stats_path'], index=False)
                else:
                    stat_df.to_csv(cfg['storage']['regression_stats_path'], mode='a', header=False, index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = read_config(args.config)
    main(cfg)