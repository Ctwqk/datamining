"""Execute the notebook and guard the boundary between EDA and model evaluation."""

import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1]


def test_cross_validation_imputes_only_from_training_rows(tmp_path):
    shutil.copyfile(HERE / "titanic.csv", tmp_path / "titanic.csv")
    notebook = nbformat.read(HERE / "titanic_visualization.ipynb", as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
from sklearn.base import clone

# The EDA may fill missing values, but evaluation must retain the original gaps.
raw = pd.read_csv('titanic.csv').dropna(subset=['survived'])
for column in ['Age', 'Fare']:
    assert features[column].isna().sum() == raw[column.lower()].isna().sum(), column

# Changing held-out values must not change training-fold imputation statistics.
training = features.iloc[:3].copy()
training['Age'] = [10., 30., np.nan]
training['Fare'] = [20., 40., np.nan]
fitted = clone(model).fit(training, [0, 1, 0])
held_out = features.iloc[:2].copy()
held_out['Age'] = [np.nan, 1000.]
held_out['Fare'] = [np.nan, 2000.]
imputed = fitted[0].transform(held_out)
assert imputed[0, features.columns.get_loc('Age')] == 20.
assert imputed[0, features.columns.get_loc('Fare')] == 30.
"""
        )
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
