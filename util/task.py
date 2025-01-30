import pandas as pd
import openml


def get_task(task_id: int) -> tuple[pd.DataFrame, str, str]:
    """
    Get the task from OpenML and return the dataset, dataset name and target column
    """
    task = openml.tasks.get_task(task_id)
    data = task.get_dataset()
    data_df = data.get_data()[0]
    dataset_name = data.name
    target = task.target_name
    return data_df, dataset_name, target


def generate_train_test_splits(data_df: pd.DataFrame,
                               task_id: int,
                               split_num: int
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the train/test splits for a given dataset.
    Returns the train and test dataframes
    """
    task = openml.tasks.get_task(task_id)
    train_indices, test_indices = task.get_train_test_split_indices(split_num)
    return data_df.iloc[train_indices], data_df.iloc[test_indices]