import logging
import pandas as pd

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from carte_ai import CARTERegressor, CARTEClassifier, Table2GraphTransformer
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Any

logger = logging.getLogger(__name__)

def split(train_df: pd.DataFrame,
          test_df: pd.DataFrame,
          target: str,
          ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split and preprocess two dataframes into their respective feature and
    target vectors.

    Parameters
    ----------
    train_df : pd.DataFrame
        The training dataframe
    test_df : pd.DataFrame
        The test dataframe
    target : str
        The name of the target column
    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        The training features, training target, test features, test target,
        validation features, validation target (optional, only if get_val=True else, these will be none)
    """
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    return X_train, y_train, X_test, y_test


def preprocess(X_train: pd.DataFrame,
               y_train: pd.Series,
               X_test: pd.DataFrame,
               y_test: pd.Series,
               model: Any) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Run preprocessing strategy depending on the model type.
    """
    # basic general preprocessing
    pt = infer_problem_type(y_train)
    lc = LabelCleaner.construct(problem_type=pt, y=y_train)
    y_train = lc.transform(y_train)
    y_test = lc.transform(y_test)

    if isinstance(model, CARTERegressor) or isinstance(model, CARTEClassifier):
        return _preprocess_CARTE(X_train, y_train, X_test, y_test)

    elif isinstance(model, TabPFNClassifier) or isinstance(model, TabPFNRegressor):
        return _preprocess_TabPFN(X_train, y_train, X_test, y_test)

    else:
        raise ValueError(f"Model type {type(model)} not supported")


def _preprocess_CARTE(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocess data for CARTE models
    """
    model_path = hf_hub_download(repo_id='hi-paris/fastText', filename="cc.en.300.bin")
    transformer = Table2GraphTransformer(fasttext_model_path=model_path)
    for column in X_train.columns:
        if X_train[column].dtype == 'category':
            X_train[column] = X_train[column].astype('object')
            X_test[column] = X_test[column].astype('object')

    X_train = transformer.fit_transform(X_train, y=y_train)
    X_test = transformer.transform(X_test)

    return X_train, y_train, X_test, y_test


def _preprocess_TabPFN(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocess data for TabPFN models
    """
    # y will already be preprocessed
    preprocessor = AutoMLPipelineFeatureGenerator(**{"verbosity": 0})

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, y_train, X_test, y_test
