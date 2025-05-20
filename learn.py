# # # # # # # # #!/usr/bin/env python3
# # # # # # # # """
# # # # # # # # learn.py

# # # # # # # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # # # # # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # # # # # # – Defines a BayesianModel structure
# # # # # # # # – Learns CPTs via Maximum Likelihood Estimation
# # # # # # # # – Validates normalization, pickles CPDs to cpds.pkl
# # # # # # # # """

# # # # # # # # import os
# # # # # # # # import pickle
# # # # # # # # import pandas as pd
# # # # # # # # from pgmpy.models import BayesianModel
# # # # # # # # from pgmpy.estimators import MaximumLikelihoodEstimator

# # # # # # # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # # # # # # PICKLE_PATH = "cpds.pkl"

# # # # # # # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # # # # # # #     # Discretize Income and Experience
# # # # # # # #     df["Income_Level"] = pd.cut(
# # # # # # # #         df["Income"],
# # # # # # # #         bins=[-1, 50000, 100000, df["Income"].max()],
# # # # # # # #         labels=["Low", "Medium", "High"]
# # # # # # # #     )
# # # # # # # #     df["Experience_Level"] = pd.cut(
# # # # # # # #         df["Experience"],
# # # # # # # #         bins=[-1, 2, 5, df["Experience"].max()],
# # # # # # # #         labels=["Junior", "Mid", "Senior"]
# # # # # # # #     )

# # # # # # # #     # Map binary flags to strings
# # # # # # # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"})
# # # # # # # #     df["Car_Ownership"]   = df["Car_Ownership"].map({0: "No", 1: "Yes"})

# # # # # # # #     # Map target to categorical
# # # # # # # #     df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"})

# # # # # # # #     # Keep only the columns we model
# # # # # # # #     cols = [
# # # # # # # #         "Income_Level",
# # # # # # # #         "Experience_Level",
# # # # # # # #         "House_Ownership",
# # # # # # # #         "Car_Ownership",
# # # # # # # #         "Risk_Flag",
# # # # # # # #     ]
# # # # # # # #     return df[cols].dropna()

# # # # # # # # def build_and_learn(df: pd.DataFrame) -> BayesianModel:
# # # # # # # #     # Define network structure
# # # # # # # #     structure = [
# # # # # # # #         ("Income_Level",    "Risk_Flag"),
# # # # # # # #         ("Experience_Level","Risk_Flag"),
# # # # # # # #         ("House_Ownership", "Risk_Flag"),
# # # # # # # #         ("Car_Ownership",   "Risk_Flag"),
# # # # # # # #     ]
# # # # # # # #     model = BayesianModel(structure)
# # # # # # # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # # # # # # #     return model

# # # # # # # # def validate_cpds(model: BayesianModel):
# # # # # # # #     for cpd in model.get_cpds():
# # # # # # # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# # # # # # # #         for idx, row in enumerate(values):
# # # # # # # #             if not abs(row.sum() - 1.0) < 1e-6:
# # # # # # # #                 raise ValueError(
# # # # # # # #                     f"Normalization error in CPT for {cpd.variable}, row {idx}"
# # # # # # # #                 )

# # # # # # # # def main():
# # # # # # # #     if not os.path.exists(RAW_CSV):
# # # # # # # #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# # # # # # # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # # # # # # #     raw_df = pd.read_csv(RAW_CSV)

# # # # # # # #     print("[learn.py] Preprocessing data...")
# # # # # # # #     data = preprocess(raw_df)

# # # # # # # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # # # # # # #     model = build_and_learn(data)

# # # # # # # #     print("[learn.py] Validating learned CPDs...")
# # # # # # # #     validate_cpds(model)

# # # # # # # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # # # # # # #     with open(PICKLE_PATH, "wb") as f:
# # # # # # # #         pickle.dump(model.get_cpds(), f)

# # # # # # # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # # # # # # if __name__ == "__main__":
# # # # # # # #     main()


# # # # # # # #!/usr/bin/env python3
# # # # # # # """
# # # # # # # learn.py

# # # # # # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # # # # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # # # # # – Defines a BayesianModel structure
# # # # # # # – Learns CPDs via Maximum Likelihood Estimation
# # # # # # # – Validates normalization, pickles CPDs to cpds.pkl
# # # # # # # """

# # # # # # # import os
# # # # # # # import pickle
# # # # # # # import pandas as pd
# # # # # # # from pgmpy.models import BayesianModel
# # # # # # # from pgmpy.estimators import MaximumLikelihoodEstimator
# # # # # # # from typing import List

# # # # # # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # # # # # PICKLE_PATH = "cpds.pkl"

# # # # # # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # # # # # #     """
# # # # # # #     Preprocesses raw data:
# # # # # # #     - Discretizes 'Income' and 'Experience'
# # # # # # #     - Maps binary flags to categorical strings
# # # # # # #     - Selects relevant columns for modeling
# # # # # # #     """
# # # # # # #     df["Income_Level"] = pd.cut(
# # # # # # #         df["Income"],
# # # # # # #         bins=[-1, 50000, 100000, df["Income"].max()],
# # # # # # #         labels=["Low", "Medium", "High"]
# # # # # # #     )

# # # # # # #     df["Experience_Level"] = pd.cut(
# # # # # # #         df["Experience"],
# # # # # # #         bins=[-1, 2, 5, df["Experience"].max()],
# # # # # # #         labels=["Junior", "Mid", "Senior"]
# # # # # # #     )

# # # # # # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"})
# # # # # # #     df["Car_Ownership"]   = df["Car_Ownership"].map({0: "No", 1: "Yes"})
# # # # # # #     df["Risk_Flag"]       = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"})

# # # # # # #     columns = [
# # # # # # #         "Income_Level",
# # # # # # #         "Experience_Level",
# # # # # # #         "House_Ownership",
# # # # # # #         "Car_Ownership",
# # # # # # #         "Risk_Flag"
# # # # # # #     ]
# # # # # # #     return df[columns].dropna()

# # # # # # # def get_model_structure() -> List[tuple]:
# # # # # # #     """
# # # # # # #     Defines and returns the Bayesian network structure.
# # # # # # #     """
# # # # # # #     return [
# # # # # # #         ("Income_Level",    "Risk_Flag"),
# # # # # # #         ("Experience_Level","Risk_Flag"),
# # # # # # #         ("House_Ownership", "Risk_Flag"),
# # # # # # #         ("Car_Ownership",   "Risk_Flag")
# # # # # # #     ]

# # # # # # # def build_and_learn(df: pd.DataFrame) -> BayesianModel:
# # # # # # #     """
# # # # # # #     Builds and fits a BayesianModel using Maximum Likelihood Estimation.
# # # # # # #     """
# # # # # # #     structure = get_model_structure()
# # # # # # #     model = BayesianModel(structure)
# # # # # # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # # # # # #     return model

# # # # # # # def validate_cpds(model: BayesianModel) -> None:
# # # # # # #     """
# # # # # # #     Validates that all CPDs are normalized.
# # # # # # #     """
# # # # # # #     for cpd in model.get_cpds():
# # # # # # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# # # # # # #         for idx, row in enumerate(values):
# # # # # # #             if not abs(row.sum() - 1.0) < 1e-6:
# # # # # # #                 raise ValueError(
# # # # # # #                     f"Normalization error in CPT for {cpd.variable}, row {idx}"
# # # # # # #                 )

# # # # # # # def main():
# # # # # # #     if not os.path.exists(RAW_CSV):
# # # # # # #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# # # # # # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # # # # # #     raw_df = pd.read_csv(RAW_CSV)

# # # # # # #     print("[learn.py] Preprocessing data...")
# # # # # # #     data = preprocess(raw_df)

# # # # # # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # # # # # #     model = build_and_learn(data)

# # # # # # #     print("[learn.py] Validating learned CPDs...")
# # # # # # #     validate_cpds(model)

# # # # # # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # # # # # #     with open(PICKLE_PATH, "wb") as f:
# # # # # # #         pickle.dump(model.get_cpds(), f)

# # # # # # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # # # # # if __name__ == "__main__":
# # # # # # #     main()

# # # # # # """
# # # # # # learn.py

# # # # # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # # # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # # # # – Defines BayesianModel structure
# # # # # # – Learns CPDs via Maximum Likelihood Estimation
# # # # # # – Validates normalization, pickles CPDs to cpds.pkl
# # # # # # """

# # # # # # import os
# # # # # # import pickle
# # # # # # import pandas as pd
# # # # # # from pgmpy.models import BayesianModel
# # # # # # from pgmpy.estimators import MaximumLikelihoodEstimator
# # # # # # from typing import List

# # # # # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # # # # PICKLE_PATH = "cpds.pkl"

# # # # # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # # # # #     """
# # # # # #     Preprocesses raw data:
# # # # # #     - Discretizes 'Income' and 'Experience'
# # # # # #     - Maps binary flags to categorical strings
# # # # # #     - Selects relevant columns for modeling
# # # # # #     """
# # # # # #     # Improved binning with explicit thresholds
# # # # # #     df["Income_Level"] = pd.cut(
# # # # # #         df["Income"],
# # # # # #         bins=[0, 50000, 100000, df["Income"].max()],
# # # # # #         labels=["Low", "Medium", "High"],
# # # # # #         right=False  # [0-50k), [50k-100k), [100k+)
# # # # # #     )

# # # # # #     df["Experience_Level"] = pd.cut(
# # # # # #         df["Experience"],
# # # # # #         bins=[0, 2, 5, df["Experience"].max()],
# # # # # #         labels=["Junior", "Mid", "Senior"],
# # # # # #         right=False
# # # # # #     )

# # # # # #     # Safer mapping with fillna
# # # # # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # # # #     df["Car_Ownership"] = df["Car_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # # # #     df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"}).fillna("Unknown")

# # # # # #     columns = [
# # # # # #         "Income_Level",
# # # # # #         "Experience_Level",
# # # # # #         "House_Ownership",
# # # # # #         "Car_Ownership",
# # # # # #         "Risk_Flag"
# # # # # #     ]
# # # # # #     return df[columns].dropna()

# # # # # # def get_model_structure() -> List[tuple]:
# # # # # #     """
# # # # # #     Defines and returns the Bayesian network structure.
# # # # # #     """
# # # # # #     return [
# # # # # #         ("Income_Level",    "Risk_Flag"),
# # # # # #         ("Experience_Level", "Risk_Flag"),
# # # # # #         ("House_Ownership", "Risk_Flag"),
# # # # # #         ("Car_Ownership",   "Risk_Flag")
# # # # # #     ]

# # # # # # def build_and_learn(df: pd.DataFrame) -> BayesianModel:
# # # # # #     """
# # # # # #     Builds and fits a BayesianModel using Maximum Likelihood Estimation.
# # # # # #     """
# # # # # #     structure = get_model_structure()
# # # # # #     model = BayesianModel(structure)
# # # # # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # # # # #     return model

# # # # # # def validate_cpds(model: BayesianModel) -> None:
# # # # # #     """
# # # # # #     Validates that all CPDs are properly normalized (columns sum to 1).
# # # # # #     """
# # # # # #     for cpd in model.get_cpds():
# # # # # #         # Reshape to (num_states, num_parent_combinations)
# # # # # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
        
# # # # # #         # Check each parent combination (columns)
# # # # # #         column_sums = values.sum(axis=0)
# # # # # #         for col_idx, col_sum in enumerate(column_sums):
# # # # # #             if not abs(col_sum - 1.0) < 1e-6:
# # # # # #                 raise ValueError(
# # # # # #                     f"CPD normalization error in {cpd.variable} | "
# # # # # #                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
# # # # # #                 )

# # # # # # def main():
# # # # # #     if not os.path.exists(RAW_CSV):
# # # # # #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# # # # # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # # # # #     raw_df = pd.read_csv(RAW_CSV)

# # # # # #     print("[learn.py] Preprocessing data...")
# # # # # #     data = preprocess(raw_df)

# # # # # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # # # # #     model = build_and_learn(data)

# # # # # #     print("[learn.py] Validating learned CPDs...")
# # # # # #     validate_cpds(model)

# # # # # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # # # # #     with open(PICKLE_PATH, "wb") as f:
# # # # # #         pickle.dump(model.get_cpds(), f)

# # # # # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # # # # if __name__ == "__main__":
# # # # # #     main()

# # # # # """
# # # # # learn.py

# # # # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # # # – Defines BayesianModel structure
# # # # # – Learns CPDs via Maximum Likelihood Estimation
# # # # # – Validates normalization, pickles CPDs to cpds.pkl
# # # # # """

# # # # # import os
# # # # # import pickle
# # # # # import pandas as pd
# # # # # from pgmpy.models import BayesianModel
# # # # # from pgmpy.estimators import MaximumLikelihoodEstimator
# # # # # from typing import List

# # # # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # # # PICKLE_PATH = "cpds.pkl"

# # # # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # # # #     """
# # # # #     Preprocesses raw data:
# # # # #     - Discretizes 'Income' and 'Experience'
# # # # #     - Maps binary flags to categorical strings
# # # # #     - Selects relevant columns for modeling
# # # # #     """
# # # # #     df["Income_Level"] = pd.cut(
# # # # #         df["Income"],
# # # # #         bins=[0, 50000, 100000, df["Income"].max()],
# # # # #         labels=["Low", "Medium", "High"],
# # # # #         right=False
# # # # #     )

# # # # #     df["Experience_Level"] = pd.cut(
# # # # #         df["Experience"],
# # # # #         bins=[0, 2, 5, df["Experience"].max()],
# # # # #         labels=["Junior", "Mid", "Senior"],
# # # # #         right=False
# # # # #     )

# # # # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # # #     df["Car_Ownership"] = df["Car_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # # #     df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"}).fillna("Unknown")

# # # # #     columns = [
# # # # #         "Income_Level",
# # # # #         "Experience_Level",
# # # # #         "House_Ownership",
# # # # #         "Car_Ownership",
# # # # #         "Risk_Flag"
# # # # #     ]
# # # # #     return df[columns].dropna()

# # # # # def get_model_structure() -> List[tuple]:
# # # # #     return [
# # # # #         ("Income_Level",    "Risk_Flag"),
# # # # #         ("Experience_Level", "Risk_Flag"),
# # # # #         ("House_Ownership", "Risk_Flag"),
# # # # #         ("Car_Ownership",   "Risk_Flag")
# # # # #     ]

# # # # # def build_and_learn(df: pd.DataFrame) -> BayesianModel:
# # # # #     structure = get_model_structure()
# # # # #     model = BayesianModel(structure)
# # # # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # # # #     return model

# # # # # def validate_cpds(model: BayesianModel) -> None:
# # # # #     for cpd in model.get_cpds():
# # # # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# # # # #         column_sums = values.sum(axis=0)
# # # # #         for col_idx, col_sum in enumerate(column_sums):
# # # # #             if not abs(col_sum - 1.0) < 1e-6:
# # # # #                 raise ValueError(
# # # # #                     f"CPD normalization error in {cpd.variable} | "
# # # # #                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
# # # # #                 )

# # # # # def main():
# # # # #     if not os.path.exists(RAW_CSV):
# # # # #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# # # # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # # # #     raw_df = pd.read_csv(RAW_CSV)

# # # # #     print("[learn.py] Preprocessing data...")
# # # # #     data = preprocess(raw_df)

# # # # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # # # #     model = build_and_learn(data)

# # # # #     print("[learn.py] Validating learned CPDs...")
# # # # #     validate_cpds(model)

# # # # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # # # #     with open(PICKLE_PATH, "wb") as f:
# # # # #         pickle.dump(model.get_cpds(), f)

# # # # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # #### learn.py

# # # # #!/usr/bin/env python3
# # # # """
# # # # learn.py

# # # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # # – Defines BayesianNetwork structure
# # # # – Learns CPDs via Maximum Likelihood Estimation
# # # # – Validates normalization, pickles CPDs to cpds.pkl
# # # # """

# # # # import os
# # # # import pickle
# # # # import pandas as pd
# # # # from pgmpy.models import BayesianNetwork
# # # # from pgmpy.estimators import MaximumLikelihoodEstimator
# # # # from typing import List

# # # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # # PICKLE_PATH = "cpds.pkl"

# # # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # # #     """
# # # #     Preprocesses raw data:
# # # #     - Discretizes 'Income' and 'Experience'
# # # #     - Maps binary flags to categorical strings
# # # #     - Selects relevant columns for modeling
# # # #     """
# # # #     df["Income_Level"] = pd.cut(
# # # #         df["Income"],
# # # #         bins=[0, 50000, 100000, df["Income"].max()],
# # # #         labels=["Low", "Medium", "High"],
# # # #         right=False
# # # #     )

# # # #     df["Experience_Level"] = pd.cut(
# # # #         df["Experience"],
# # # #         bins=[0, 2, 5, df["Experience"].max()],
# # # #         labels=["Junior", "Mid", "Senior"],
# # # #         right=False
# # # #     )

# # # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # #     df["Car_Ownership"] = df["Car_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # # #     df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"}).fillna("Unknown")

# # # #     columns = [
# # # #         "Income_Level",
# # # #         "Experience_Level",
# # # #         "House_Ownership",
# # # #         "Car_Ownership",
# # # #         "Risk_Flag"
# # # #     ]
# # # #     return df[columns].dropna()

# # # # def get_model_structure() -> List[tuple]:
# # # #     return [
# # # #         ("Income_Level",    "Risk_Flag"),
# # # #         ("Experience_Level", "Risk_Flag"),
# # # #         ("House_Ownership", "Risk_Flag"),
# # # #         ("Car_Ownership",   "Risk_Flag")
# # # #     ]

# # # # def build_and_learn(df: pd.DataFrame) -> BayesianNetwork:
# # # #     structure = get_model_structure()
# # # #     model = BayesianNetwork(structure)
# # # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # # #     return model

# # # # def validate_cpds(model: BayesianNetwork) -> None:
# # # #     for cpd in model.get_cpds():
# # # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# # # #         column_sums = values.sum(axis=0)
# # # #         for col_idx, col_sum in enumerate(column_sums):
# # # #             if not abs(col_sum - 1.0) < 1e-6:
# # # #                 raise ValueError(
# # # #                     f"CPD normalization error in {cpd.variable} | "
# # # #                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
# # # #                 )

# # # # def main():
# # # #     if not os.path.exists(RAW_CSV):
# # # #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# # # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # # #     raw_df = pd.read_csv(RAW_CSV)

# # # #     print("[learn.py] Preprocessing data...")
# # # #     data = preprocess(raw_df)

# # # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # # #     model = build_and_learn(data)

# # # #     print("[learn.py] Validating learned CPDs...")
# # # #     validate_cpds(model)

# # # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # # #     with open(PICKLE_PATH, "wb") as f:
# # # #         pickle.dump(model.get_cpds(), f)

# # # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # # if __name__ == "__main__":
# # # #     main()


# # # #!/usr/bin/env python3
# # # """
# # # learn.py

# # # – Loads raw credit data from data/credit_risk_dataset.csv
# # # – Preprocesses (discretizes continuous features, maps binary flags)
# # # – Defines BayesianNetwork structure
# # # – Learns CPDs via Maximum Likelihood Estimation
# # # – Validates normalization, pickles CPDs to cpds.pkl
# # # """

# # # import os
# # # import pickle
# # # import pandas as pd
# # # from pgmpy.models import BayesianNetwork
# # # from pgmpy.estimators import MaximumLikelihoodEstimator
# # # from typing import List

# # # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # # PICKLE_PATH = "cpds.pkl"

# # # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# # #     """
# # #     Preprocesses raw data:
# # #     - Discretizes 'Income' and 'Experience'
# # #     - Maps binary flags to categorical strings
# # #     - Selects relevant columns for modeling
# # #     """
# # #     df["Income_Level"] = pd.cut(
# # #         df["Income"],
# # #         bins=[0, 50000, 100000, df["Income"].max()],
# # #         labels=["Low", "Medium", "High"],
# # #         right=False
# # #     )

# # #     df["Experience_Level"] = pd.cut(
# # #         df["Experience"],
# # #         bins=[0, 2, 5, df["Experience"].max()],
# # #         labels=["Junior", "Mid", "Senior"],
# # #         right=False
# # #     )

# # #     df["House_Ownership"] = df["House_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # #     df["Car_Ownership"] = df["Car_Ownership"].map({0: "No", 1: "Yes"}).fillna("Unknown")
# # #     df["Risk_Flag"] = df["Risk_Flag"].map({0: "LowRisk", 1: "HighRisk"}).fillna("Unknown")

# # #     columns = [
# # #         "Income_Level",
# # #         "Experience_Level",
# # #         "House_Ownership",
# # #         "Car_Ownership",
# # #         "Risk_Flag"
# # #     ]
# # #     return df[columns].dropna()

# # # def get_model_structure() -> List[tuple]:
# # #     return [
# # #         ("Income_Level", "Risk_Flag"),
# # #         ("Experience_Level", "Risk_Flag"),
# # #         ("House_Ownership", "Risk_Flag"),
# # #         ("Car_Ownership", "Risk_Flag")
# # #     ]

# # # def build_and_learn(df: pd.DataFrame) -> BayesianNetwork:
# # #     structure = get_model_structure()
# # #     model = BayesianNetwork(structure)
# # #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# # #     return model

# # # def validate_cpds(model: BayesianNetwork) -> None:
# # #     for cpd in model.get_cpds():
# # #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# # #         column_sums = values.sum(axis=0)
# # #         for col_idx, col_sum in enumerate(column_sums):
# # #             if not abs(col_sum - 1.0) < 1e-6:
# # #                 raise ValueError(
# # #                     f"CPD normalization error in {cpd.variable} | "
# # #                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
# # #                 )

# # # def main():
# # #     if not os.path.exists(RAW_CSV):
# # #         raise FileNotFoundError(f"Raw data file not found at path: {RAW_CSV}")

# # #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# # #     raw_df = pd.read_csv(RAW_CSV)

# # #     print("[learn.py] Preprocessing data...")
# # #     data = preprocess(raw_df)

# # #     print("[learn.py] Learning Bayesian Network CPDs...")
# # #     model = build_and_learn(data)

# # #     print("[learn.py] Validating learned CPDs...")
# # #     validate_cpds(model)

# # #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# # #     with open(PICKLE_PATH, "wb") as f:
# # #         pickle.dump(model.get_cpds(), f)

# # #     print("[learn.py] All CPDs learned and validated successfully.")

# # # if __name__ == "__main__":
# # #     main()


# # #!/usr/bin/env python3
# # """
# # learn.py

# # – Loads raw credit data from data/credit_risk_dataset.csv
# # – Preprocesses (discretizes continuous features, maps binary flags)
# # – Defines BayesianNetwork structure
# # – Learns CPDs via Maximum Likelihood Estimation
# # – Validates normalization, pickles CPDs to cpds.pkl
# # """

# # import os
# # import pickle
# # import pandas as pd
# # from pgmpy.models import DiscreteBayesianNetwork
# # from pgmpy.estimators import MaximumLikelihoodEstimator
# # from typing import List

# # RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# # PICKLE_PATH = "cpds.pkl"

# # def preprocess(df: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     Preprocesses raw data:
# #     - Discretizes 'person_income' and 'person_emp_length'
# #     - Maps ownership and default flags to categorical strings
# #     - Selects relevant columns for modeling
# #     """
# #     df["Income_Level"] = pd.cut(
# #         df["person_income"],
# #         bins=[0, 50000, 100000, df["person_income"].max()],
# #         labels=["Low", "Medium", "High"],
# #         right=False
# #     )

# #     df["Experience_Level"] = pd.cut(
# #         df["person_emp_length"],
# #         bins=[0, 2, 5, df["person_emp_length"].max()],
# #         labels=["Junior", "Mid", "Senior"],
# #         right=False
# #     )

# #     df["House_Ownership"] = df["person_home_ownership"].fillna("Unknown")

# #     df["Risk_Flag"] = df["cb_person_default_on_file"].map({
# #         "N": "LowRisk",
# #         "Y": "HighRisk"
# #     }).fillna("Unknown")

# #     columns = [
# #         "Income_Level",
# #         "Experience_Level",
# #         "House_Ownership",
# #         "Risk_Flag"
# #     ]
# #     return df[columns].dropna()

# # def get_model_structure() -> List[tuple]:
# #     return [
# #         ("Income_Level",    "Risk_Flag"),
# #         ("Experience_Level", "Risk_Flag"),
# #         ("House_Ownership", "Risk_Flag")
# #     ]

# # def build_and_learn(df: pd.DataFrame) -> BayesianNetwork:
# #     structure = get_model_structure()
# #     model = BayesianNetwork(structure)
# #     model.fit(df, estimator=MaximumLikelihoodEstimator)
# #     return model

# # def validate_cpds(model: BayesianNetwork) -> None:
# #     for cpd in model.get_cpds():
# #         values = cpd.get_values().reshape(cpd.cardinality, -1)
# #         column_sums = values.sum(axis=0)
# #         for col_idx, col_sum in enumerate(column_sums):
# #             if not abs(col_sum - 1.0) < 1e-6:
# #                 raise ValueError(
# #                     f"CPD normalization error in {cpd.variable} | "
# #                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
# #                 )

# # def main():
# #     if not os.path.exists(RAW_CSV):
# #         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

# #     print(f"[learn.py] Loading raw data from {RAW_CSV}")
# #     raw_df = pd.read_csv(RAW_CSV)

# #     print("[learn.py] Preprocessing data...")
# #     data = preprocess(raw_df)

# #     print("[learn.py] Learning Bayesian Network CPDs...")
# #     model = build_and_learn(data)

# #     print("[learn.py] Validating learned CPDs...")
# #     validate_cpds(model)

# #     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
# #     with open(PICKLE_PATH, "wb") as f:
# #         pickle.dump(model.get_cpds(), f)

# #     print("[learn.py] All CPDs learned and validated successfully.")

# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# """
# learn.py

# – Loads raw credit data from data/credit_risk_dataset.csv
# – Preprocesses (discretizes continuous features, maps binary flags)
# – Defines BayesianNetwork structure
# – Learns CPDs via Maximum Likelihood Estimation
# – Validates normalization, pickles CPDs to cpds.pkl
# """

# import os
# import pickle
# import pandas as pd
# from pgmpy.models import DiscreteBayesianNetwork
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from typing import List

# RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
# PICKLE_PATH = "cpds.pkl"

# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
#     df["Income_Level"] = pd.cut(
#         df["person_income"],
#         bins=[0, 50000, 100000, df["person_income"].max()],
#         labels=["Low", "Medium", "High"],
#         right=False
#     )

#     df["Experience_Level"] = pd.cut(
#         df["person_emp_length"],
#         bins=[0, 2, 5, df["person_emp_length"].max()],
#         labels=["Junior", "Mid", "Senior"],
#         right=False
#     )

#     df["House_Ownership"] = df["person_home_ownership"].map({
#         "RENT": "No",
#         "OWN": "Yes",
#         "MORTGAGE": "Yes"
#     }).fillna("Unknown")

#     df["Car_Ownership"] = df["cb_person_default_on_file"].map({
#         "N": "No",
#         "Y": "Yes"
#     }).fillna("Unknown")

#     df["Risk_Flag"] = df["loan_status"].map({
#         0: "LowRisk",
#         1: "HighRisk"
#     }).fillna("Unknown")

#     columns = [
#         "Income_Level",
#         "Experience_Level",
#         "House_Ownership",
#         "Car_Ownership",
#         "Risk_Flag"
#     ]
#     return df[columns].dropna()

# def get_model_structure() -> List[tuple]:
#     return [
#         ("Income_Level",    "Risk_Flag"),
#         ("Experience_Level", "Risk_Flag"),
#         ("House_Ownership", "Risk_Flag"),
#         ("Car_Ownership",   "Risk_Flag")
#     ]

# def build_and_learn(df: pd.DataFrame) -> DiscreteBayesianNetwork:
#     structure = get_model_structure()
#     model = DiscreteBayesianNetwork(structure)
#     model.fit(df, estimator=MaximumLikelihoodEstimator)
#     return model

# def validate_cpds(model: DiscreteBayesianNetwork) -> None:
#     for cpd in model.get_cpds():
#         values = cpd.get_values().reshape(cpd.cardinality, -1)
#         column_sums = values.sum(axis=0)
#         for col_idx, col_sum in enumerate(column_sums):
#             if not abs(col_sum - 1.0) < 1e-6:
#                 raise ValueError(
#                     f"CPD normalization error in {cpd.variable} | "
#                     f"Parent combination {col_idx}: sum={col_sum:.4f}"
#                 )

# def main():
#     if not os.path.exists(RAW_CSV):
#         raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

#     print(f"[learn.py] Loading raw data from {RAW_CSV}")
#     raw_df = pd.read_csv(RAW_CSV)

#     print("[learn.py] Preprocessing data...")
#     data = preprocess(raw_df)

#     print("[learn.py] Learning Bayesian Network CPDs...")
#     model = build_and_learn(data)

#     print("[learn.py] Validating learned CPDs...")
#     validate_cpds(model)

#     print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
#     with open(PICKLE_PATH, "wb") as f:
#         pickle.dump(model.get_cpds(), f)

#     print("[learn.py] All CPDs learned and validated successfully.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
learn.py

– Loads raw credit data from data/credit_risk_dataset.csv
– Preprocesses (discretizes continuous features, maps binary flags)
– Defines BayesianNetwork structure
– Learns CPDs via Maximum Likelihood Estimation
– Validates normalization, pickles CPDs to cpds.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from typing import List

RAW_CSV = os.path.join("data", "credit_risk_dataset.csv")
PICKLE_PATH = "cpds.pkl"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses raw data:
    - Discretizes 'person_income' and 'person_emp_length'
    - Maps home ownership to binary categories
    - Maps loan_status to risk categories
    - Selects relevant columns for modeling
    """
    df["Income_Level"] = pd.cut(
        df["person_income"],
        bins=[0, 30000, 70000, df["person_income"].max()],
        labels=["Low", "Medium", "High"],
        right=False
    )

    df["Experience_Level"] = pd.cut(
        df["person_emp_length"],
        bins=[0, 3, 7, df["person_emp_length"].max()],
        labels=["Junior", "Mid", "Senior"],
        right=False
    )

    df["House_Ownership"] = df["person_home_ownership"].map(
        lambda x: "Yes" if x in ["OWN", "MORTGAGE"] else "No"
    )

    df["Car_Ownership"] = df["cb_person_default_on_file"].map({"N": "No", "Y": "Yes"}).fillna("Unknown")
    df["Risk_Flag"] = df["loan_status"].map({0: "LowRisk", 1: "HighRisk"}).fillna("Unknown")

    columns = [
        "Income_Level",
        "Experience_Level",
        "House_Ownership",
        "Car_Ownership",
        "Risk_Flag"
    ]
    return df[columns].dropna()


def get_model_structure() -> List[tuple]:
    return [
        ("Income_Level",    "Risk_Flag"),
        ("Experience_Level", "Risk_Flag"),
        ("House_Ownership", "Risk_Flag"),
        ("Car_Ownership",   "Risk_Flag")
    ]


def build_and_learn(df: pd.DataFrame) -> DiscreteBayesianNetwork:
    structure = get_model_structure()
    model = DiscreteBayesianNetwork(structure)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model


def validate_cpds(model: DiscreteBayesianNetwork) -> None:
    for cpd in model.get_cpds():
        card_product = int(np.prod(cpd.cardinality[1:]))  # parent combinations
        values = cpd.get_values().reshape((cpd.cardinality[0], card_product))
        column_sums = values.sum(axis=0)
        for col_idx, col_sum in enumerate(column_sums):
            if not abs(col_sum - 1.0) < 1e-6:
                raise ValueError(
                    f"CPD normalization error in {cpd.variable} | "
                    f"Parent combination {col_idx}: sum={col_sum:.4f}"
                )


def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw data file not found: {RAW_CSV}")

    print(f"[learn.py] Loading raw data from {RAW_CSV}")
    raw_df = pd.read_csv(RAW_CSV)

    print("[learn.py] Preprocessing data...")
    data = preprocess(raw_df)

    print("[learn.py] Learning Bayesian Network CPDs...")
    model = build_and_learn(data)

    print("[learn.py] Validating learned CPDs...")
    validate_cpds(model)

    print(f"[learn.py] Pickling CPDs to {PICKLE_PATH}")
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(model.get_cpds(), f)

    print("[learn.py] All CPDs learned and validated successfully.")


if __name__ == "__main__":
    main()
