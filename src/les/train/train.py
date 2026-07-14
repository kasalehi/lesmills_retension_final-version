import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
f1_score,
balanced_accuracy_score,
roc_auc_score
)
from xgboost import XGBClassifier
from src.les.logger import logging
from src.les.exception import CustomException

@dataclass
class ModelConfig:
        base_dir: str = Path(__file__).resolve().parent.parent.parent / "artifacts"
        preprocessor_path: str = base_dir / "preprocessor.pkl"


        # ❌ RandomForest REMOVED
        param_grids: dict = field(default_factory=lambda: {
            "xgboost": {
                    "clf__n_estimators": [200,300,400,500],
                    "clf__max_depth": [2,3, 5, 7],
                    "clf__learning_rate": [0.03, 0.05, 0.1, 0.01, 0.001],
                },
        })

        models: dict = field(default_factory=lambda: {
            "xgboost":XGBClassifier(
                random_state=42,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                subsample=0.8,
                colsample_bytree=0.8
            ),
        })


class ModelTraing:
    def __init__(self):
        self.data = ModelConfig()


# ✅ FIXED INDENTATION (VERY IMPORTANT)
    def trainingModel(self, x_train, y_train, x_test, y_test, preprocessor):
        try:
            results = {}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # ================================
            # HANDLE IMBALANCE
            # ================================
            class_counts = Counter(y_train)
            total = sum(class_counts.values())

            class_weights = {0: 3, 1: 3, 2: 1}
            import numpy as np
            sample_weights = np.array([class_weights[y] for y in y_train])

            for name, model in self.data.models.items():
                logging.info(f"🚀 Training model: {name}")

                param_grid = self.data.param_grids.get(name, {})
                if not param_grid:
                    logging.warning(f"No param grid found for model '{name}'. Skipping.")
                    continue

                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("clf", model),
                ])

                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=2,
                    scoring="f1_macro",
                    verbose=1,
                )

                fit_params = {}

                # Apply weights
                if name in ["gradient_boosting", "xgboost"]:
                    fit_params["clf__sample_weight"] = sample_weights

                grid.fit(x_train, y_train, **fit_params)

                best_estimator = grid.best_estimator_
                best_params = grid.best_params_
                cv_score = grid.best_score_

                y_pred = best_estimator.predict(x_test)

                test_acc = accuracy_score(y_test, y_pred)
                test_bal_acc = balanced_accuracy_score(y_test, y_pred)
                test_f1_macro = f1_score(y_test, y_pred, average="macro")

                # ROC-AUC
                try:
                    y_prob = best_estimator.predict_proba(x_test)
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except Exception:
                    roc_auc = None

                logging.info(
                    f"Model: {name} | Best Params: {best_params} | "
                    f"CV Score: {cv_score:.4f} | "
                    f"Test Acc: {test_acc:.4f} | "
                    f"Test BalAcc: {test_bal_acc:.4f} | "
                    f"Test F1_macro: {test_f1_macro:.4f} | "
                    f"ROC-AUC: {roc_auc}"
                )

                logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
                logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

                results[name] = {
                    "best_estimator": best_estimator,
                    "best_params": best_params,
                    "cv_score": cv_score,
                    "test_accuracy": test_acc,
                    "test_balanced_accuracy": test_bal_acc,
                    "test_f1_macro": test_f1_macro,
                    "roc_auc": roc_auc,
                }

            if not results:
                raise CustomException("No models were successfully trained.", sys)

            best_model_name = max(results, key=lambda m: results[m]["test_f1_macro"])
            best_info = results[best_model_name]

            logging.info(
                f"🏆 Best model is '{best_model_name}' "
                f"with F1_macro {best_info['test_f1_macro']:.4f}"
            )

            return best_info["best_estimator"], best_model_name, results

        except Exception as e:
            raise CustomException(e, sys)

