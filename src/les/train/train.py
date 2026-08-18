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
roc_auc_score,
precision_score,
recall_score,
make_scorer
)
from xgboost import XGBClassifier
from src.les.logger import logging
from src.les.exception import CustomException

@dataclass
class ModelConfig:
        base_dir: str = Path(__file__).resolve().parent.parent.parent / "artifacts"
        preprocessor_path: str = base_dir / "preprocessor.pkl"


        # ✅ MINIMAL TUNING - Only tune 2-3 key params (2-3 min training, best results)
        param_grids: dict = field(default_factory=lambda: {
            "xgboost": {
                    "clf__n_estimators": [400, 500],  # Trees
                    "clf__max_depth": [5, 6],  # Depth
                    "clf__learning_rate": [0.05, 0.1],  # Learning rate
                },
        })

        models: dict = field(default_factory=lambda: {
            "xgboost": XGBClassifier(
                random_state=42,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                subsample=0.8,
                colsample_bytree=0.8,
                # Note: scale_pos_weight not used for multi-class; using sample_weight instead
                verbosity=0  # Suppress XGBoost warnings
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

            # ✅ PHASE 1: AGGRESSIVE class weights (2x boost to catch more churners)
            class_weights = {0: 10, 1: 10, 2: 1}
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

                # ✅ Use GridSearchCV for simple exhaustive search
                # Only 8 combinations = very fast
                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=2,
                    scoring="f1_weighted",  # Better for imbalanced classes
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
                y_prob = best_estimator.predict_proba(x_test)

                test_acc = accuracy_score(y_test, y_pred)
                test_bal_acc = balanced_accuracy_score(y_test, y_pred)
                test_f1_macro = f1_score(y_test, y_pred, average="macro")
                test_f1_weighted = f1_score(y_test, y_pred, average="weighted")

                # ✅ PER-CLASS METRICS FOR CLASSES 0 & 1
                precision_0 = precision_score(y_test, y_pred, labels=[0], average='micro', zero_division=0)
                recall_0 = recall_score(y_test, y_pred, labels=[0], average='micro', zero_division=0)
                f1_0 = f1_score(y_test, y_pred, labels=[0], average='micro', zero_division=0)

                precision_1 = precision_score(y_test, y_pred, labels=[1], average='micro', zero_division=0)
                recall_1 = recall_score(y_test, y_pred, labels=[1], average='micro', zero_division=0)
                f1_1 = f1_score(y_test, y_pred, labels=[1], average='micro', zero_division=0)

                # ROC-AUC
                try:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except Exception:
                    roc_auc = None

                logging.info(
                    f"Model: {name} | Best Params: {best_params} | "
                    f"CV Score: {cv_score:.4f} | "
                    f"Test Acc: {test_acc:.4f} | "
                    f"Test BalAcc: {test_bal_acc:.4f} | "
                    f"Test F1_weighted: {test_f1_weighted:.4f} | "
                    f"ROC-AUC: {roc_auc}"
                )

                # ✅ CLASS 0 METRICS
                logging.info(
                    f"CLASS 0 (Churn 0-3mo) | "
                    f"Precision: {precision_0:.4f} | "
                    f"Recall: {recall_0:.4f} | "
                    f"F1: {f1_0:.4f}"
                )

                # ✅ CLASS 1 METRICS
                logging.info(
                    f"CLASS 1 (Churn 3-6mo) | "
                    f"Precision: {precision_1:.4f} | "
                    f"Recall: {recall_1:.4f} | "
                    f"F1: {f1_1:.4f}"
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
                    "test_f1_weighted": test_f1_weighted,
                    "roc_auc": roc_auc,
                    # ✅ Per-class metrics for Classes 0 & 1
                    "class_0_precision": precision_0,
                    "class_0_recall": recall_0,
                    "class_0_f1": f1_0,
                    "class_1_precision": precision_1,
                    "class_1_recall": recall_1,
                    "class_1_f1": f1_1,
                    "y_prob": y_prob,  # Store probabilities for threshold adjustment
                }

            if not results:
                raise CustomException("No models were successfully trained.", sys)

            # ✅ Use F1_weighted for better class balance
            best_model_name = max(results, key=lambda m: results[m]["test_f1_weighted"])
            best_info = results[best_model_name]

            logging.info(
                f"🏆 Best model is '{best_model_name}' "
                f"with F1_weighted {best_info['test_f1_weighted']:.4f}"
            )

            # ✅ Display Class 0 & 1 performance summary
            logging.info(
                f"📊 Churn Classes Performance:\n"
                f"   Class 0 - Precision: {best_info['class_0_precision']:.4f}, Recall: {best_info['class_0_recall']:.4f}, F1: {best_info['class_0_f1']:.4f}\n"
                f"   Class 1 - Precision: {best_info['class_1_precision']:.4f}, Recall: {best_info['class_1_recall']:.4f}, F1: {best_info['class_1_f1']:.4f}"
            )

            return best_info["best_estimator"], best_model_name, results

        except Exception as e:
            raise CustomException(e, sys)

