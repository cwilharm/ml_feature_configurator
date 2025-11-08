"""
Configuration and Session State Management for ML Feature Configurator
"""
import streamlit as st


def init_session_state():
   """Initialize all session state variables"""

   # Main data storage
   if 'df' not in st.session_state:
       st.session_state.df = None
   if 'mapped_df' not in st.session_state:
       st.session_state.mapped_df = None
   if 'feature_df' not in st.session_state:
       st.session_state.feature_df = None
   if 'column_mapping' not in st.session_state:
       st.session_state.column_mapping = {}

   # Version history for undo/reset
   if 'original_df' not in st.session_state:
       st.session_state.original_df = None
   if 'mapped_df_backup' not in st.session_state:
       st.session_state.mapped_df_backup = None
   if 'feature_df_backup' not in st.session_state:
       st.session_state.feature_df_backup = None
   if 'scaling_applied' not in st.session_state:
       st.session_state.scaling_applied = False

   # Step completion tracking
   if 'step_1_completed' not in st.session_state:
       st.session_state.step_1_completed = False
   if 'step_2_completed' not in st.session_state:
       st.session_state.step_2_completed = False
   if 'step_3_completed' not in st.session_state:
       st.session_state.step_3_completed = False
   if 'step_4_completed' not in st.session_state:
       st.session_state.step_4_completed = False
   if 'step_5_completed' not in st.session_state:
       st.session_state.step_5_completed = False
   if 'step_6_completed' not in st.session_state:
       st.session_state.step_6_completed = False
   if 'step_6_5_completed' not in st.session_state:
       st.session_state.step_6_5_completed = False
   if 'step_7_completed' not in st.session_state:
       st.session_state.step_7_completed = False


# App configuration
APP_CONFIG = {
   'page_title': 'ML Feature Configurator',
   'page_icon': '📊',
   'layout': 'wide',
   'version': '1.2.0',
   'about_text': """
   # 👨‍💻 About This Tool

   **ML Feature Configurator**

   Transform simple OHLC data into ML-ready feature sets with 24+ technical indicators and synthetic feature generation using genetic programming (gplearn).

   **Features:**
   - 🔧 Auto Column Mapping
   - 📊 24+ Technical Indicators
   - 🧬 Genetic Programming (gplearn)
   - 🎯 Synthetic Features
   - 🎯 Target Variable Generation
   - 📥 CSV Export

   Built with ❤️ using Streamlit, pandas, scikit-learn & gplearn.

   Version: 1.2.0
   """
}

# Required and optional columns for OHLC data
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close']
OPTIONAL_COLS = ['Volume', 'Date']

# Column mapping variations for auto-detection
COLUMN_VARIATIONS = {
   'open': ['open', 'o', 'open_price', 'opening'],
   'high': ['high', 'h', 'high_price', 'highest'],
   'low': ['low', 'l', 'low_price', 'lowest'],
   'close': ['close', 'c', 'close_price', 'closing', 'last'],
   'volume': ['volume', 'vol', 'v', 'tick_volume', 'amount'],
   'date': ['date', 'd', 'time', 'datetime', 'timestamp', 'dt', 'date_time']
}
