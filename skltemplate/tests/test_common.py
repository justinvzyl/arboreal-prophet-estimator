import pytest

from sklearn.utils.estimator_checks import check_estimator

from arborealprophet import TemplateEstimator
from arborealprophet import TemplateClassifier
from arborealprophet import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
