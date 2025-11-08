"""
Utility functions for ML Feature Configurator
"""
import pandas as pd
from config import COLUMN_VARIATIONS


def auto_map_columns(df_columns, target_col):
   df_columns_lower = {col.lower(): col for col in df_columns}
   target_lower = target_col.lower()

   # Direct match
   if target_lower in df_columns_lower:
       return df_columns_lower[target_lower]
   
   # Check for common variations
   if target_lower in COLUMN_VARIATIONS:
      for variant in COLUMN_VARIATIONS[target_lower]:
         if variant in df_columns_lower:
            return df_columns_lower[variant]
           
   return ''


def create_mapped_dataframe(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
   mapped_df = pd.DataFrame()

   for target_col, source_col in mapping.items():
      if source_col:
         mapped_df[target_col] = df[source_col]

   # Handle date column
   if mapping.get('Date'):
      try:
         mapped_df['Date'] = pd.to_datetime(mapped_df['Date'])
         mapped_df = mapped_df.set_index('Date')
      except Exception:
         pass  # If parsing fails, keep as is

   return mapped_df


def validate_ohlc_data(df: pd.DataFrame) -> tuple[bool, str]:
   required_cols = ['Open', 'High', 'Low', 'Close']

   # Check if all required columns exist
   missing_cols = [col for col in required_cols if col not in df.columns]
   if missing_cols:
      return False, f"Missing required columns: {', '.join(missing_cols)}"

   # Check for empty DataFrame
   if df.empty:
     return False, "DataFrame is empty"

   # Check for numeric data types
   for col in required_cols:
      if not pd.api.types.is_numeric_dtype(df[col]):
         return False, f"Column '{col}' must be numeric"

   return True, ""


def get_feature_count_by_category(feature_categories: dict, selected_features: dict) -> dict:
   counts = {}
   for category, selected in selected_features.items():
      if selected and category in feature_categories:
         counts[category] = len(feature_categories[category]['features'])

   return counts