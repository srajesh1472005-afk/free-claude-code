"""Machine learning component using scikit-learn for predicting processing times."""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

from free_claude_code.scheduler.models import HistoricalDataRecord


class ProcessingTimePredictor:
    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None
        self.part_encoder = LabelEncoder()
        self.material_encoder = LabelEncoder()
        self.operation_encoder = LabelEncoder()
        self.feature_names: list[str] = [
            "part_encoded",
            "material_encoded",
            "operation_encoded",
            "cutting_speed",
            "feed_rate",
            "depth_of_cut",
            "tool_condition",
        ]
        self.mae: float = 0.0
        self.r2: float = 0.0
        self.feature_importances: dict[str, float] = {}
        self.is_trained: bool = False

    def train(self, records: list[HistoricalDataRecord]) -> dict[str, Any]:
        if not records or len(records) < 2:
            self.is_trained = False
            return {"status": "insufficient_data", "mae": 0.0, "r2": 0.0}

        df = pd.DataFrame(
            [
                {
                    "part": r.part,
                    "material": r.material,
                    "operation": r.operation,
                    "cutting_speed": r.cutting_speed,
                    "feed_rate": r.feed_rate,
                    "depth_of_cut": r.depth_of_cut,
                    "tool_condition": r.tool_condition,
                    "target": r.actual_processing_time,
                }
                for r in records
            ]
        )

        # Fit encoders
        df["part_encoded"] = self.part_encoder.fit_transform(df["part"])
        df["material_encoded"] = self.material_encoder.fit_transform(df["material"])
        df["operation_encoded"] = self.operation_encoder.fit_transform(df["operation"])

        X = df[self.feature_names]
        y = df["target"]

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        self.mae = float(mean_absolute_error(y, y_pred))
        self.r2 = float(r2_score(y, y_pred))

        importances = self.model.feature_importances_
        self.feature_importances = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances, strict=True)
        }
        self.is_trained = True

        return {
            "status": "success",
            "mae": self.mae,
            "r2": self.r2,
            "feature_importances": self.feature_importances,
        }

    def predict_processing_time(
        self,
        part: str,
        material: str,
        operation: str,
        fallback_time: float,
        cutting_speed: float = 250.0,
        feed_rate: float = 0.25,
        depth_of_cut: float = 2.0,
        tool_condition: float = 0.9,
    ) -> float:
        if not self.is_trained or self.model is None:
            return fallback_time

        try:
            # Handle unseen categories gracefully
            part_val = (
                self.part_encoder.transform([part])[0]
                if part in self.part_encoder.classes_
                else 0
            )
            mat_val = (
                self.material_encoder.transform([material])[0]
                if material in self.material_encoder.classes_
                else 0
            )
            op_val = (
                self.operation_encoder.transform([operation])[0]
                if operation in self.operation_encoder.classes_
                else 0
            )

            features = pd.DataFrame(
                [
                    [
                        part_val,
                        mat_val,
                        op_val,
                        cutting_speed,
                        feed_rate,
                        depth_of_cut,
                        tool_condition,
                    ]
                ],
                columns=self.feature_names,
            )

            pred = float(self.model.predict(features)[0])
            return max(0.1, pred)
        except Exception:
            return fallback_time
