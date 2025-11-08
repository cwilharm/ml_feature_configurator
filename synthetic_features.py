"""
Synthetic Feature Engineering Module
"""
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from gplearn.genetic import SymbolicTransformer


class SyntheticFeatureEngineering:
    """Synthetic Feature Generation using various techniques"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_features = list(df.columns)

    def add_polynomial_features(self, degree: int = 2, selected_features: List[str] = None):
        """
        Generate polynomial features (degree 2 or 3)
        Example: For features [A, B], degree 2 generates: A^2, A*B, B^2
        """
        if selected_features is None:
            # Use numeric columns only
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.df[selected_features])

        # Get feature names
        feature_names = poly.get_feature_names_out(selected_features)

        # Add to dataframe (skip the original features, only add new combinations)
        start_idx = len(selected_features)
        for idx, name in enumerate(feature_names[start_idx:], start=start_idx):
            # Clean up feature name
            clean_name = name.replace(' ', '*').replace('^', '_pow_')
            self.df[f'Poly_{clean_name}'] = poly_features[:, idx]

        return self

    def add_feature_interactions(self, selected_features: List[str] = None, max_interactions: int = 20):
        """
        Generate feature interactions (multiplication, division, addition, subtraction)
        """
        if selected_features is None:
            # Use numeric columns only, limit to avoid explosion
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = numeric_cols[:10]  # Limit to first 10 to avoid explosion

        if len(selected_features) < 2:
            return self

        # Generate pairs
        feature_pairs = list(combinations(selected_features, 2))

        # Limit number of interactions
        feature_pairs = feature_pairs[:max_interactions]

        for feat1, feat2 in feature_pairs:
            # Multiplication
            self.df[f'Interact_Mult_{feat1}_x_{feat2}'] = self.df[feat1] * self.df[feat2]

            # Division (avoid division by zero)
            self.df[f'Interact_Div_{feat1}_by_{feat2}'] = self.df[feat1] / (self.df[feat2].replace(0, np.nan))

            # Addition
            self.df[f'Interact_Add_{feat1}_plus_{feat2}'] = self.df[feat1] + self.df[feat2]

            # Subtraction
            self.df[f'Interact_Sub_{feat1}_minus_{feat2}'] = self.df[feat1] - self.df[feat2]

            # Ratio
            self.df[f'Interact_Ratio_{feat1}_to_{feat2}'] = self.df[feat1] / (self.df[feat2].replace(0, np.nan))

        return self

    def add_lag_features(self, selected_features: List[str] = None, lags: List[int] = [1, 2, 3, 5, 10]):
        """
        Generate lag features (time-shifted features)
        """
        if selected_features is None:
            # Use numeric columns
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            for lag in lags:
                self.df[f'Lag_{lag}_{feature}'] = self.df[feature].shift(lag)

        return self

    def add_rolling_statistics_on_features(self, selected_features: List[str] = None,
                                           windows: List[int] = [3, 5, 10]):
        """
        Apply rolling statistics on existing features
        """
        if selected_features is None:
            # Use numeric columns
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            for window in windows:
                self.df[f'Rolling_Mean_{window}_{feature}'] = self.df[feature].rolling(window=window).mean()
                self.df[f'Rolling_Std_{window}_{feature}'] = self.df[feature].rolling(window=window).std()
                self.df[f'Rolling_Min_{window}_{feature}'] = self.df[feature].rolling(window=window).min()
                self.df[f'Rolling_Max_{window}_{feature}'] = self.df[feature].rolling(window=window).max()

        return self

    def add_diff_features(self, selected_features: List[str] = None, periods: List[int] = [1, 2, 5]):
        """
        Generate difference features (change over time)
        """
        if selected_features is None:
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            for period in periods:
                self.df[f'Diff_{period}_{feature}'] = self.df[feature].diff(period)
                self.df[f'Pct_Change_{period}_{feature}'] = self.df[feature].pct_change(period)

        return self

    def add_ratio_features(self, base_features: List[str] = None):
        """
        Generate ratio features against Close price
        """
        if base_features is None:
            base_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if 'Close' not in self.df.columns or not base_features:
            return self

        close = self.df['Close']

        for feature in base_features:
            if feature != 'Close':
                self.df[f'Ratio_{feature}_to_Close'] = self.df[feature] / close.replace(0, np.nan)

        return self

    def add_mathematical_transforms(self, selected_features: List[str] = None):
        """
        Apply mathematical transformations (log, sqrt, square, etc.)
        """
        if selected_features is None:
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            # Only apply to positive values
            if (self.df[feature] > 0).all():
                self.df[f'Log_{feature}'] = np.log(self.df[feature])
                self.df[f'Sqrt_{feature}'] = np.sqrt(self.df[feature])

            # Square
            self.df[f'Square_{feature}'] = self.df[feature] ** 2

            # Cube
            self.df[f'Cube_{feature}'] = self.df[feature] ** 3

            # Inverse (avoid division by zero)
            self.df[f'Inverse_{feature}'] = 1 / self.df[feature].replace(0, np.nan)

        return self

    def add_cumulative_features(self, selected_features: List[str] = None):
        """
        Generate cumulative sum and product features
        """
        if selected_features is None:
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            self.df[f'CumSum_{feature}'] = self.df[feature].cumsum()
            self.df[f'CumProd_{feature}'] = (1 + self.df[feature]).cumprod()

        return self

    def add_binning_features(self, selected_features: List[str] = None, bins: int = 5):
        """
        Create binned/categorical versions of continuous features
        """
        if selected_features is None:
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features:
            return self

        for feature in selected_features:
            try:
                self.df[f'Binned_{bins}_{feature}'] = pd.qcut(
                    self.df[feature],
                    q=bins,
                    labels=False,
                    duplicates='drop'
                )
            except:
                # If qcut fails (e.g., too many duplicates), use cut
                try:
                    self.df[f'Binned_{bins}_{feature}'] = pd.cut(
                        self.df[feature],
                        bins=bins,
                        labels=False
                    )
                except:
                    pass  # Skip if binning fails

        return self

    def add_gplearn_features(self, selected_features: List[str] = None,
                            n_components: int = 10,
                            generations: int = 20,
                            population_size: int = 1000,
                            tournament_size: int = 20,
                            metric: str = 'pearson'):
        """
        Generate features using Genetic Programming (gplearn)
        This creates evolved mathematical combinations of features
        """
        if selected_features is None:
            selected_features = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not selected_features or len(selected_features) < 2:
            return self

        # Prepare data - remove NaN rows for gplearn
        X = self.df[selected_features].copy()
        X_clean = X.dropna()

        if len(X_clean) < 10:  # Need minimum data for evolution
            return self

        # Check if target variable exists - use it for supervised feature generation
        if 'target' in self.df.columns:
            # SUPERVISED: Use actual target for better feature evolution
            y_clean = self.df.loc[X_clean.index, 'target'].values

            # Remove rows where target is NaN
            valid_mask = ~np.isnan(y_clean)
            X_clean = X_clean[valid_mask]
            y_clean = y_clean[valid_mask]

            if len(X_clean) < 10:
                return self

            y_target = y_clean
            mode = "SUPERVISED (using target variable)"
        else:
            # UNSUPERVISED: Create a dummy target using variance of features
            y_target = X_clean.std(axis=1).values
            mode = "UNSUPERVISED (no target found)"

        try:
            # Initialize SymbolicTransformer
            gp = SymbolicTransformer(
                generations=generations,
                population_size=population_size,
                hall_of_fame=max(10, n_components),
                n_components=n_components,
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'max', 'min'],
                tournament_size=tournament_size,
                const_range=(-1.0, 1.0),
                init_depth=(2, 6),
                init_method='half and half',
                metric=metric,
                parsimony_coefficient=0.001,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                random_state=42,
                n_jobs=1
            )

            # Fit and transform with target (supervised if available)
            gp_features = gp.fit_transform(X_clean.values, y_target)

            # Store the mode info for user feedback
            self.gplearn_mode = mode

            # Add evolved features back to dataframe
            for i in range(gp_features.shape[1]):
                feature_name = f'GP_Evolved_{i+1}'
                # Create full-length series with NaN for dropped rows
                full_feature = pd.Series(index=self.df.index, dtype=float)
                full_feature.loc[X_clean.index] = gp_features[:, i]
                self.df[feature_name] = full_feature

                # Also add the formula/expression as metadata if possible
                try:
                    program = gp._programs[0][i]
                    formula = str(program)
                    # Store formula in column name or as comment
                    self.df.rename(columns={feature_name: f'{feature_name}'}, inplace=True)
                except:
                    pass

        except Exception as e:
            # If gplearn fails, continue without error
            pass

        return self

    def get_dataframe(self):
        """Return the dataframe with all synthetic features"""
        return self.df
