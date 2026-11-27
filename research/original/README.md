# Original notebook

`Parkinsons.ipynb` is the original Google Colab notebook behind the ICOIICS 2025 paper,
kept exactly as written (outputs included) so the paper's numbers can be traced.

What it does (cells 13–38):

- drops `name` and `status` to get the 22 features;
- `train_test_split(X, Y, test_size=0.2, random_state=2)`: an **unstratified** 80/20 split;
- `StandardScaler` fitted on the training rows only;
- `svm.SVC(kernel='linear')` with default `C=1`: **no hyperparameter search**;
- prints 0.8846 training and 0.8718 test accuracy, the figures reported in the paper.

`research/reproduce_paper.py` reproduces this exactly (see
[`research/results/paper_reproduction.md`](../results/paper_reproduction.md)). The paper's
text describes a stratified split and an RBF SVM tuned by grid search; the report compares
that described protocol with what the notebook ran.

The notebook loads the dataset from a copy on GitHub; its SHA-256 matches the UCI file
pinned in `data/download.py`, so the data is byte-for-byte the same.
