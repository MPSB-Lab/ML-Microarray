import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from joblib import Parallel, delayed

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

def evaluate_model(X, y, model, loo):
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        y_true_all.append(y[test_idx][0])
        y_pred_all.append(y_pred[0])
        y_prob_all.append(y_prob[0])

    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true_all, y_pred_all)
    metrics["Precision"] = precision_score(y_true_all, y_pred_all, zero_division=0)
    metrics["Recall"] = recall_score(y_true_all, y_pred_all)
    metrics["F1"] = f1_score(y_true_all, y_pred_all)
    metrics["MCC"] = matthews_corrcoef(y_true_all, y_pred_all)
    metrics["AUC"] = roc_auc_score(y_true_all, y_prob_all)

    return metrics


def single_permutation(X, y, model, loo):
    y_perm = np.random.permutation(y)
    try:
        return evaluate_model(X, y_perm, model, loo)
    except Exception:
        return None


def permutation_test_parallel_all(X, y, model, n_permutations=200, n_jobs=-1):
    loo = LeaveOneOut()

    observed = evaluate_model(X, y, model, loo)

    perm_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(single_permutation)(X, y, model, loo) for _ in range(n_permutations)
    )
    perm_results = [res for res in perm_results if res is not None]

    perm_metrics = {m: [res[m] for res in perm_results if not np.isnan(res[m])] for m in observed.keys()}

    pvals = {}
    perm_means = {}
    for metric, obs_val in observed.items():
        null_scores = np.array(perm_metrics[metric])
        perm_means[metric] = np.mean(null_scores)
        pvals[metric] = (np.sum(null_scores >= obs_val) + 1) / (len(null_scores) + 1)

    return observed, perm_means, pvals

def run_permutation_pipeline(expr_file, gene_sets, n_permutations=200, n_jobs=-1, out_csv="/content/drive/MyDrive/Sensepermutation_results_as.csv"):
    results = []

    for gene_set_name in gene_sets:
        # Load data
        expr_df = pd.read_excel(expr_file, sheet_name=gene_set_name)
        gene_names = expr_df.iloc[:, 0].astype(str)
        expr_data = expr_df.iloc[:, 1:]

        control_cols = [col for col in expr_data.columns if "UNC" in col]
        infected_cols = [col for col in expr_data.columns if "UNC" not in col]

        X_all = expr_data[control_cols + infected_cols].T
        X_all.columns = gene_names
        y = np.array([0] * len(control_cols) + [1] * len(infected_cols))
        X = X_all.values

        # Run models
        for model_name, model in models.items():
            observed, perm_means, pvals = permutation_test_parallel_all(
                X, y, model, n_permutations=n_permutations, n_jobs=n_jobs
            )

            for metric in observed.keys():
                results.append({
                    "GeneSet": gene_set_name,
                    "Model": model_name,
                    "Metric": metric,
                    "Observed": round(observed[metric], 3),
                    "NullMean": round(perm_means[metric], 3),
                    "p-value": round(pvals[metric], 4)
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    return results_df

gene_sets = ["S_20","S_30","S_40","S_50","S_60","S_70","S_80","S_92"]
expr_file = "/content/drive/MyDrive/DATA.xlsx"
results_df = run_permutation_pipeline(expr_file, gene_sets, n_permutations=200, n_jobs=-1)
print(results_df)
