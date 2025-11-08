"""
ML Feature Configurator - Refactored Version

Transform OHLC data into ML-ready features with 24+ technical indicators,
synthetic feature generation, and target variable creation.

Version: 1.1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Import from new modules
from config import init_session_state, APP_CONFIG, REQUIRED_COLS, OPTIONAL_COLS
from utils import auto_map_columns, create_mapped_dataframe, validate_ohlc_data
from feature_engineering import FeatureEngineering
from synthetic_features import SyntheticFeatureEngineering
from charts import create_data_overview_charts, create_feature_overview_charts, create_feature_target_analysis_charts


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title=APP_CONFIG['page_title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    menu_items={
        'About': APP_CONFIG['about_text']
    }
)


# ========== INITIALIZE SESSION STATE ==========
init_session_state()


# ========== MAIN FUNCTION ==========
def main():
    st.title("📊 ML Feature Generator")
    st.markdown("Transform your OHLC data into ML-ready features with full control!")

    # ========== AUTO-CHECK EXISTING DATA ==========
    # Automatically check and update completion status based on session state
    if st.session_state.df is not None:
        st.session_state.step_1_completed = True

    if st.session_state.mapped_df is not None:
        st.session_state.step_2_completed = True
        st.session_state.step_4_completed = True  # Auto-complete optional config

    if st.session_state.feature_df is not None:
        st.session_state.step_6_completed = True
        st.session_state.step_7_completed = True

    # ========== SIDEBAR: STEP-BY-STEP GUIDE ==========
    st.sidebar.header("📋 Step-by-Step Guide")

    st.sidebar.markdown("""
    **1️⃣ Upload CSV**

    **2️⃣ Column Mapping**

    **3️⃣ Data Scaling** *(Optional)*

    **4️⃣ Window Configuration** *(Optional)*

    **5️⃣ Feature Selection**
    - Choose from 24+ categories

    **6️⃣ Generate Features**

    **7️⃣ Synthetic Features** *(Optional, Advanced)*

    **8️⃣ Target Variable** *(Optional)*

    **9️⃣ Download**

    ---

    ### 💡 Quick Tips
    - Start with 5-10 features
    - Use Feature Importance later
    - Lag features reduce row count
    - Synthetic features can explode!
    - Target defines your ML problem!
    """)

    st.sidebar.markdown("---")

    # ========== README LINK ==========
    st.sidebar.markdown("### 📚 Documentation")
    readme_url = "https://github.com/yourusername/TFT/blob/main/README_FEATURE_CONFIGURATOR.md"
    st.sidebar.markdown(f"[📖 Read Full Documentation]({readme_url})")

    st.sidebar.markdown("---")

    # ========== ABOUT ME ==========
    st.sidebar.info(APP_CONFIG['about_text'])

    st.sidebar.markdown("---")

    # Global Reset All button
    st.sidebar.markdown("### 🔄 Reset Application")
    if st.sidebar.button("🔄 Reset All to Start", type="secondary", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("✅ Application reset! Please upload CSV to start.")
        st.rerun()


    # ========== STEP 1: CSV UPLOAD ==========
    st.header("1️⃣ Upload CSV")
    uploaded_file = st.file_uploader("Choose your OHLC CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.original_df = df.copy()  # Backup original for reset
        st.session_state.step_1_completed = True  # Mark Step 1 as completed

        st.success(f"✅ File uploaded! Shape: {df.shape}")

        with st.expander("📋 Preview original data"):
            st.dataframe(df.head(10))

        # ========== STEP 2: COLUMN MAPPING ==========
        st.header("2️⃣ Column Mapping")
        st.markdown("Map your CSV columns to standard OHLC format")

        with st.spinner("Running auto-detection of columns..."):
            # Auto-detect columns
            auto_mapping = {}
            all_cols = REQUIRED_COLS + OPTIONAL_COLS

            detected_any = []
            for col in all_cols:
                detected = auto_map_columns(df.columns, col)
                auto_mapping[col] = detected
                if detected:
                    detected_any.append(col)

        if not detected_any:
            st.warning("⚠️ Could not auto-detect any columns. Please map them manually below.")
        else:
            st.success(f"✓ Auto-detected {len(detected_any)} columns")

        st.markdown("**Adjust mappings if needed:**")

        col1, col2 = st.columns(2)

        mapping = {}
        with col1:
            st.subheader("Required Columns")
            for col in REQUIRED_COLS:
                # Use auto-detected value as default index
                default_idx = 0
                detected_col = auto_mapping.get(col, '')
                if detected_col and detected_col in df.columns:
                    default_idx = list(df.columns).index(detected_col) + 1  # +1 because of empty option

                mapping[col] = st.selectbox(
                    f"Select {col} column",
                    options=[''] + list(df.columns),
                    index=default_idx,
                    key=f"map_{col}"
                )

        with col2:
            st.subheader("Optional Columns")
            for col in OPTIONAL_COLS:
                # Use auto-detected value as default index
                default_idx = 0
                detected_col = auto_mapping.get(col, '')
                if detected_col and detected_col in df.columns:
                    default_idx = list(df.columns).index(detected_col) + 1  # +1 because of empty option

                mapping[col] = st.selectbox(
                    f"Select {col} column (optional)",
                    options=[''] + list(df.columns),
                    index=default_idx,
                    key=f"map_{col}"
                )

        # Validate mapping
        if all(mapping.get(col) for col in REQUIRED_COLS):
            # Create mapped dataframe
            mapped_df = create_mapped_dataframe(df, mapping)

            st.session_state.mapped_df = mapped_df
            st.session_state.mapped_df_backup = mapped_df.copy()  # Backup for reset
            st.session_state.column_mapping = mapping
            st.session_state.step_2_completed = True  # Mark Step 2 as completed
            st.session_state.step_4_completed = True  # Mark Step 4 as completed (optional config)
            st.session_state.scaling_applied = False  # Reset scaling flag

            st.success("✅ Column mapping successful!")

            with st.expander("📋 Preview mapped data"):
                st.dataframe(mapped_df.head(10))

            # Data overview charts
            with st.expander("📊 Data Overview Charts", expanded=True):
                create_data_overview_charts(mapped_df)

            # Reset button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Reset to Original CSV"):
                    st.session_state.mapped_df = None
                    st.session_state.feature_df = None
                    st.session_state.step_2_completed = False
                    st.session_state.step_3_completed = False
                    st.session_state.step_4_completed = False
                    st.session_state.step_5_completed = False
                    st.session_state.step_6_completed = False
                    st.session_state.step_6_5_completed = False
                    st.session_state.step_7_completed = False
                    st.session_state.scaling_applied = False
                    st.rerun()

            # ========== STEP 3: SCALING OPTIONS ==========
            st.header("3️⃣ Data Scaling (Optional)")
            st.markdown("Scale your OHLC data before feature calculation")

            apply_scaling = st.checkbox("Apply scaling to OHLC data")

            if apply_scaling:
                scaler_type = st.selectbox(
                    "Select scaling method",
                    options=['StandardScaler', 'MinMaxScaler', 'RobustScaler'],
                    help="StandardScaler: (x-mean)/std für normalverteilte Daten" \
                    "MinMaxScaler: (x-min)/(max-min) für [0,1] Range" \
                    "RobustScaler: (x-median)/IQR robust gegen Outliers"
                )

                cols_to_scale = st.multiselect(
                    "Select columns to scale",
                    options=['Open', 'High', 'Low', 'Close', 'Volume'],
                    default=['Open', 'High', 'Low', 'Close']
                )

                if st.button("Apply Scaling"):
                    scaled_df = mapped_df.copy()

                    if scaler_type == 'StandardScaler':
                        scaler = StandardScaler()
                    elif scaler_type == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()

                    # Only scale columns that exist
                    cols_to_scale_filtered = [col for col in cols_to_scale if col in scaled_df.columns]

                    scaled_df[cols_to_scale_filtered] = scaler.fit_transform(
                        scaled_df[cols_to_scale_filtered]
                    )

                    st.session_state.mapped_df = scaled_df
                    st.session_state.step_3_completed = True  # Mark Step 3 as completed
                    st.session_state.scaling_applied = True
                    st.success(f"✅ Applied {scaler_type} to {cols_to_scale_filtered}")

                    with st.expander("📋 Preview scaled data"):
                        st.dataframe(scaled_df.head(10))

            # Undo Scaling button
            if st.session_state.scaling_applied and st.session_state.mapped_df_backup is not None:
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("↩️ Undo Scaling"):
                        st.session_state.mapped_df = st.session_state.mapped_df_backup.copy()
                        st.session_state.scaling_applied = False
                        st.session_state.step_3_completed = False
                        st.success("✅ Scaling removed - back to mapped data!")
                        st.rerun()

            # ========== STEP 4: WINDOW LENGTH CONFIGURATION ==========
            st.header("4️⃣ Window Length Configuration (Optional)")
            st.markdown("Customize window lengths for rolling calculations")

            with st.expander("🔧 Configure Window Lengths", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Moving Averages")
                    ma_windows_input = st.text_input(
                        "MA Windows (comma-separated)",
                        value="5,10,20,50,100,200",
                        help="Perioden für gleitende Durchschnitte - kleine Windows (5,10) reagieren schnell, große Windows (100,200) glätten Trends"
                    )
                    ma_windows = [int(x.strip()) for x in ma_windows_input.split(',')]

                with col2:
                    st.subheader("Momentum Indicators")
                    momentum_windows_input = st.text_input(
                        "Momentum Windows",
                        value="5,10,20",
                        help="Perioden für Momentum und ROC - misst Preisänderung über N Perioden (5=kurzfristig, 20=mittelfristig)"
                    )
                    momentum_windows = [int(x.strip()) for x in momentum_windows_input.split(',')]

                with col3:
                    st.subheader("Statistical Features")
                    stat_windows_input = st.text_input(
                        "Statistical Windows",
                        value="5,10,20,50",
                        help="Perioden für Rolling Statistics (Std, Skewness, Kurtosis, Z-Score) - berechnet Volatilität und Verteilungs-Eigenschaften"
                    )
                    stat_windows = [int(x.strip()) for x in stat_windows_input.split(',')]

                # Store in session state
                window_config = {
                    'ma_windows': ma_windows,
                    'momentum_windows': momentum_windows,
                    'stat_windows': stat_windows
                }
                st.session_state.window_config = window_config

            # ========== STEP 5: FEATURE SELECTION ==========
            st.header("5️⃣ Feature Selection")
            st.markdown("Select the feature categories you want to add")

            # Feature categories with descriptions
            feature_categories = {
                "Price-Based Features": {
                    "description": "Range, Close/Open Ratio, High/Low Ratio, Mid-Price",
                    "features": ["Range", "Close/Open Ratio", "High/Low Ratio", "Mid-Price"]
                },
                "Returns": {
                    "description": "Daily Return, Log Return, High-Low Return, Open-Close Return",
                    "features": ["Daily Return", "Log Return", "High-Low Return", "Open-Close Return"]
                },
                "Moving Averages (SMA)": {
                    "description": f"Simple Moving Averages für windows: {ma_windows_input}",
                    "features": [f"SMA_{w}" for w in ma_windows]
                },
                "Moving Averages (EMA)": {
                    "description": f"Exponential Moving Averages für windows: {ma_windows_input}",
                    "features": [f"EMA_{w}" for w in ma_windows]
                },
                "Moving Averages (WMA)": {
                    "description": "Weighted Moving Averages - linear gewichtete Durchschnitte",
                    "features": ["WMA"]
                },
                "Hull Moving Average (HMA)": {
                    "description": "Hull Moving Average - schnellster und glattester MA",
                    "features": ["HMA"]
                },
                "Volatility": {
                    "description": "Rolling Std, ATR - misst Preisschwankung und Marktrisiko",
                    "features": ["Volatility", "ATR"]
                },
                "Momentum": {
                    "description": f"Momentum und ROC für windows: {momentum_windows_input}",
                    "features": ["Momentum", "ROC"]
                },
                "RSI": {
                    "description": "Relative Strength Index (14, 21) - überkauft/überverkauft Indikator",
                    "features": ["RSI_14", "RSI_21"]
                },
                "Stochastic Oscillator": {
                    "description": "Stochastic %K und %D - Momentum Oszillator",
                    "features": ["Stoch_%K", "Stoch_%D"]
                },
                "Williams %R": {
                    "description": "Williams %R (14) - Momentum Indikator für überkauft/überverkauft",
                    "features": ["Williams_%R_14"]
                },
                "CCI": {
                    "description": "Commodity Channel Index (20) - misst Abweichung vom Durchschnitt",
                    "features": ["CCI_20"]
                },
                "CMO": {
                    "description": "Chande Momentum Oscillator (14) - Momentum Stärke",
                    "features": ["CMO_14"]
                },
                "MACD": {
                    "description": "MACD (12,26,9) - Trend-Following Momentum Indikator",
                    "features": ["MACD", "MACD_Signal", "MACD_Hist"]
                },
                "Bollinger Bands": {
                    "description": "Bollinger Bands (20,2) mit %B und Bandwidth",
                    "features": ["BB_Upper", "BB_Middle", "BB_Lower", "BB_%B", "BB_Bandwidth"]
                },
                "Keltner Channel": {
                    "description": "Keltner Channel (20,10,2) - ATR-basierte Volatilitätsbänder",
                    "features": ["KC_Upper", "KC_Middle", "KC_Lower"]
                },
                "Candlestick Features": {
                    "description": "Body Length, Shadows, Ratios, Direction",
                    "features": ["Body", "Upper_Shadow", "Lower_Shadow", "Body/Range", "Direction"]
                },
                "Ichimoku Cloud": {
                    "description": "Ichimoku (9,26,52) - Tenkan, Kijun, Senkou Spans ohne Future Bias",
                    "features": ["Ichimoku_Tenkan", "Ichimoku_Kijun", "Senkou_A", "Senkou_B"]
                },
                "ADX & DMI": {
                    "description": "ADX (14) und +DI/-DI - Trend Stärke und Richtung",
                    "features": ["ADX", "+DI", "-DI"]
                },
                "Parabolic SAR": {
                    "description": "Parabolic SAR (0.02, 0.2) - Stop and Reverse Punkte",
                    "features": ["PSAR"]
                },
                "SuperTrend": {
                    "description": "SuperTrend (10,3) - ATR-basierter Trend Indikator",
                    "features": ["SuperTrend", "SuperTrend_Direction"]
                },
                "Awesome Oscillator": {
                    "description": "Awesome Oscillator (5,34) - Midprice-basierter Momentum",
                    "features": ["AO"]
                },
                "Volume Features": {
                    "description": "OBV, VWAP, MFI, Volume Change und SMA",
                    "features": ["OBV", "VWAP", "MFI", "Volume_Change"]
                },
                "Statistical Features": {
                    "description": f"Rolling Mean/Median/Std/Skewness/Kurtosis/Z-Score für windows: {stat_windows_input}",
                    "features": ["Rolling_Mean", "Rolling_Std", "Z-Score", "Skewness", "Kurtosis"]
                },
                "Time Features": {
                    "description": "Day of Week/Month, Week, Month, Quarter mit cyclical encoding",
                    "features": ["DayOfWeek", "DayOfMonth", "Month", "Quarter", "sin/cos encoding"]
                },
                "Cumulative Returns": {
                    "description": f"Cumulative Returns für windows: {stat_windows_input}",
                    "features": ["Cumulative_Return"]
                },
                "MA Crossovers": {
                    "description": "Moving Average Crossover Signals (Fast/Slow)",
                    "features": ["MA_Cross_Signal"]
                },
            }

            # Select/Deselect All buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Select All Features"):
                    for cat in feature_categories.keys():
                        st.session_state[f"feature_{cat}"] = True
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All Features"):
                    for cat in feature_categories.keys():
                        st.session_state[f"feature_{cat}"] = False
                    st.rerun()

            # Feature selection checkboxes in 3 columns
            selected_features = {}

            # Split features into 3 columns for better layout
            categories_list = list(feature_categories.items())
            cols = st.columns(3)

            for idx, (category, info) in enumerate(categories_list):
                col_idx = idx % 3

                with cols[col_idx]:
                    # Initialize session state if not exists
                    if f"feature_{category}" not in st.session_state:
                        st.session_state[f"feature_{category}"] = False

                    # Checkbox with description
                    selected = st.checkbox(
                        category,
                        value=st.session_state[f"feature_{category}"],
                        help=info["description"],
                        key=f"checkbox_{category}"
                    )
                    selected_features[category] = selected
                    st.session_state[f"feature_{category}"] = selected

            # ========== STEP 6: GENERATE FEATURES ==========
            st.markdown("---")
            st.header("6️⃣ Generate Features")

            if st.button("🚀 Generate Features", type="primary"):
                # Check if any features selected
                if not any(selected_features.values()):
                    st.error("❌ Please select at least one feature category!")
                else:
                    with st.spinner("🔬 Generating features... This may take a moment."):
                        # Initialize feature engineering
                        fe = FeatureEngineering(st.session_state.mapped_df)

                        # Get window configurations
                        window_config = st.session_state.window_config
                        ma_windows = window_config['ma_windows']
                        momentum_windows = window_config['momentum_windows']
                        stat_windows = window_config['stat_windows']

                        # Generate each selected feature type
                        if selected_features.get("Price-Based Features"):
                            fe.add_range()
                            fe.add_close_open_ratio()
                            fe.add_high_low_ratio()
                            fe.add_mid_price()

                        if selected_features.get("Returns"):
                            fe.add_daily_return()
                            fe.add_log_return()
                            fe.add_high_low_return()
                            fe.add_open_close_return()

                        if selected_features.get("Moving Averages (SMA)"):
                            fe.add_sma(ma_windows)

                        if selected_features.get("Moving Averages (EMA)"):
                            fe.add_ema(ma_windows)

                        if selected_features.get("Moving Averages (WMA)"):
                            fe.add_wma(ma_windows)

                        if selected_features.get("Hull Moving Average (HMA)"):
                            fe.add_hma(ma_windows)

                        if selected_features.get("Volatility"):
                            fe.add_volatility(stat_windows)
                            fe.add_atr([14, 21])

                        if selected_features.get("Momentum"):
                            fe.add_momentum(momentum_windows)
                            fe.add_roc(momentum_windows)

                        if selected_features.get("RSI"):
                            fe.add_rsi([14, 21])

                        if selected_features.get("Stochastic Oscillator"):
                            fe.add_stochastic(k_window=14, d_window=3)

                        if selected_features.get("Williams %R"):
                            fe.add_williams_r([14])

                        if selected_features.get("CCI"):
                            fe.add_cci([20])

                        if selected_features.get("CMO"):
                            fe.add_cmo([14])

                        if selected_features.get("MACD"):
                            fe.add_macd(fast=12, slow=26, signal=9)

                        if selected_features.get("Bollinger Bands"):
                            fe.add_bollinger_bands(window=20, num_std=2.0)

                        if selected_features.get("Keltner Channel"):
                            fe.add_keltner_channel(window=20, atr_window=10, multiplier=2.0)

                        if selected_features.get("Candlestick Features"):
                            fe.add_candlestick_features()

                        if selected_features.get("Ichimoku Cloud"):
                            fe.add_ichimoku(tenkan=9, kijun=26, senkou_b=52)

                        if selected_features.get("ADX & DMI"):
                            fe.add_adx_dmi(window=14)

                        if selected_features.get("Parabolic SAR"):
                            fe.add_parabolic_sar(acceleration=0.02, maximum=0.2)

                        if selected_features.get("SuperTrend"):
                            fe.add_supertrend(period=10, multiplier=3.0)

                        if selected_features.get("Awesome Oscillator"):
                            fe.add_awesome_oscillator(fast=5, slow=34)

                        if selected_features.get("Volume Features"):
                            if 'Volume' in st.session_state.mapped_df.columns:
                                fe.add_volume_features(windows=[5, 10, 20])
                            else:
                                st.warning("⚠️ Volume column not found - skipping volume features")

                        if selected_features.get("Statistical Features"):
                            fe.add_statistical_features(stat_windows)

                        if selected_features.get("Time Features"):
                            # Check if index is datetime
                            if isinstance(st.session_state.mapped_df.index, pd.DatetimeIndex):
                                fe.add_time_features()
                            else:
                                st.warning("⚠️ Date index not found - skipping time features")

                        if selected_features.get("Cumulative Returns"):
                            fe.add_cumulative_returns([5, 10, 20])

                        if selected_features.get("MA Crossovers"):
                            fe.add_ma_crossovers()

                        # Get the feature dataframe
                        feature_df = fe.get_dataframe()

                        # Drop NaN rows
                        original_rows = len(feature_df)
                        feature_df = feature_df.dropna()
                        rows_dropped = original_rows - len(feature_df)

                        # Store in session state
                        st.session_state.feature_df = feature_df
                        st.session_state.step_6_completed = True

                        st.success(f"✅ Features generated successfully!")
                        st.info(f"📊 Total features: {feature_df.shape[1]} | Rows: {feature_df.shape[0]} | Dropped {rows_dropped} rows with NaN")

                        # Show feature summary
                        st.subheader("Generated Features Summary")
                        feature_counts = {}
                        for cat, info in feature_categories.items():
                            if selected_features.get(cat):
                                # Count actual features in dataframe
                                matching_cols = [col for col in feature_df.columns if any(keyword.split('_')[0] in col for keyword in info["features"])]
                                if matching_cols:
                                    feature_counts[cat] = len(matching_cols)

                        if feature_counts:
                            for cat, count in feature_counts.items():
                                st.write(f"- **{cat}**: {count} features")

                        # Show preview
                        with st.expander("Feature DataFrame Preview", expanded=False):
                            st.dataframe(feature_df.head(10))

                        # Feature overview charts
                        with st.expander("📊 Feature Analysis Charts", expanded=True):
                            create_feature_overview_charts(feature_df)

            st.markdown("Next after that: Create target variable (optional) and synthetic features (advanced)")

            st.markdown("---")

            # ========== STEP 7: TARGET VARIABLE GENERATION ==========
            if st.session_state.feature_df is not None:
                st.header("7️⃣ Target Variable (Optional)")

                st.info("""
                **Target Variable** is what your ML model is supposed to predict.
                Choose the type based on your ML problem:
                - **Regression**: Predicting continuous values (e.g., future price)
                - **Classification**: Predicting categories (e.g., trend direction)
                - **Binary**: Predicting yes/no outcomes (e.g., will the price go up?)
                """)

                current_df = st.session_state.feature_df

                # Target Type Selection
                target_type = st.radio(
                    "Choose your ML problem type:",
                    options=["Continuous (Regression)", "Categorical (Multi-Class)", "Binary (Classification)"],
                    help="Continuous: predict exact values | Categorical: predict categories | Binary: predict yes/no"
                )

                # Initialize variables for all paths
                target_method = None
                horizon = 5
                use_log_transform = False  # Default for non-continuous targets

                # ===== CONTINUOUS / REGRESSION TARGETS =====
                if target_type == "Continuous (Regression)":
                    target_method = st.selectbox(
                        "Select Target Method",
                        options=[
                            "Future Return (%)",
                            "Future Price Change (Absolute)",
                            "Future High (Max in Period)",
                            "Future Low (Min in Period)",
                            "Future Volatility (Std)",
                            "Future ATR",
                            "Future Volume Change (%)"
                        ],
                        help="Future Return: % change | Price Change: absolute | High/Low: max/min in period | Volatility: std | ATR: average true range | Volume Change: %"
                    )

                    horizon = st.slider("Prediction Horizon (periods ahead)", 1, 50, 5,
                        help="Wie weit in die Zukunft? Größerer Horizon = leichter zu predicten, aber weniger Trading-Opportunities")

                    # Log transformation option for regression
                    use_log_transform = st.checkbox(
                        "Apply Log Transformation to Target",
                        value=False,
                        help="Für Returns: log(1 + target/100) macht Verteilung symmetrischer und stabilisiert Varianz - Standard in Finance!"
                    )

                # ===== CATEGORICAL / CLASSIFICATION TARGETS =====
                elif target_type == "Categorical (Multi-Class)":

                    target_method = st.selectbox(
                        "Select Target Method",
                        options=[
                            "Trend Direction (Up/Down/Sideways)",
                            "Return Bins (Strong Down/Down/Neutral/Up/Strong Up)",
                            "Volatility Regime (Low/Medium/High)",
                            "Price vs MA (Below/At/Above)",
                            "Custom Quantile Bins"
                        ]
                    )

                    horizon = st.slider("Prediction Horizon (periods ahead)", 1, 50, 5)

                    # Initialize all variables
                    up_threshold = 1.0
                    down_threshold = -1.0
                    threshold_1 = -2.0
                    threshold_2 = -0.5
                    threshold_3 = 0.5
                    threshold_4 = 2.0
                    window = 20
                    low_percentile = 33
                    high_percentile = 67
                    ma_period = 20
                    tolerance = 0.5
                    n_bins = 5

                    if target_method == "Trend Direction (Up/Down/Sideways)":
                        col1, col2 = st.columns(2)
                        with col1:
                            up_threshold = st.number_input("Up Threshold (%)", 0.0, 10.0, 1.0, 0.1)
                        with col2:
                            down_threshold = st.number_input("Down Threshold (%)", -10.0, 0.0, -1.0, 0.1)

                    elif target_method == "Return Bins (Strong Down/Down/Neutral/Up/Strong Up)":
                        col1, col2 = st.columns(2)
                        with col1:
                            threshold_1 = st.number_input("Strong Down Threshold (%)", value=-2.0, step=0.1)
                            threshold_2 = st.number_input("Down Threshold (%)", value=-0.5, step=0.1)
                        with col2:
                            threshold_3 = st.number_input("Up Threshold (%)", value=0.5, step=0.1)
                            threshold_4 = st.number_input("Strong Up Threshold (%)", value=2.0, step=0.1)

                    elif target_method == "Volatility Regime (Low/Medium/High)":
                        window = st.slider("Volatility Window", 5, 50, 20)
                        col1, col2 = st.columns(2)
                        with col1:
                            low_percentile = st.slider("Low Percentile", 0, 50, 33)
                        with col2:
                            high_percentile = st.slider("High Percentile", 50, 100, 67)

                    elif target_method == "Price vs MA (Below/At/Above)":
                        ma_period = st.slider("MA Period", 5, 200, 20)
                        tolerance = st.slider("Tolerance (%)", 0.0, 5.0, 0.5, 0.1)

                    elif target_method == "Custom Quantile Bins":
                        n_bins = st.slider("Number of Bins", 3, 10, 5)

                # ===== BINARY / CLASSIFICATION TARGETS =====
                else:  # Binary

                    target_method = st.selectbox(
                        "Select Target Method",
                        options=[
                            "Price Up/Down (Simple Binary)",
                            "Return Exceeds Threshold",
                            "Breakout (Price > MA)",
                            "Breakdown (Price < MA)",
                            "Take Profit Hit",
                            "Stop Loss Hit",
                            "High Volatility Event"
                        ]
                    )

                    horizon = st.slider("Prediction Horizon (periods ahead)", 1, 50, 5)

                    # Initialize all variables
                    threshold = 1.0
                    ma_period = 20
                    tp_pct = 2.0
                    sl_pct = 2.0
                    vol_window = 20
                    vol_percentile = 80

                    if target_method == "Return Exceeds Threshold":
                        threshold = st.number_input("Return Threshold (%)", 0.0, 10.0, 1.0, 0.1)

                    elif target_method in ["Breakout (Price > MA)", "Breakdown (Price < MA)"]:
                        ma_period = st.slider("MA Period", 5, 200, 20)

                    elif target_method == "Take Profit Hit":
                        tp_pct = st.number_input("Take Profit (%)", 0.1, 20.0, 2.0, 0.1)

                    elif target_method == "Stop Loss Hit":
                        sl_pct = st.number_input("Stop Loss (%)", 0.1, 20.0, 2.0, 0.1)

                    elif target_method == "High Volatility Event":
                        vol_window = st.slider("Volatility Window", 5, 50, 20)
                        vol_percentile = st.slider("High Vol Percentile", 50, 99, 80)


                # Generate Target Button
                if st.button("🎯 Generate Target Variable", type="primary"):
                    with st.spinner("Generating target variable..."):
                        target_df = current_df.copy()

                        # CONTINUOUS TARGETS
                        if target_type == "Continuous (Regression)":
                            if target_method == "Future Return (%)":
                                target_df['target'] = (target_df['Close'].shift(-horizon) - target_df['Close']) / target_df['Close'] * 100
                            elif target_method == "Future Price Change (Absolute)":
                                target_df['target'] = target_df['Close'].shift(-horizon) - target_df['Close']
                            elif target_method == "Future High (Max in Period)":
                                target_df['target'] = target_df['High'].shift(-1).rolling(window=horizon).max()
                            elif target_method == "Future Low (Min in Period)":
                                target_df['target'] = target_df['Low'].shift(-1).rolling(window=horizon).min()
                            elif target_method == "Future Volatility (Std)":
                                target_df['target'] = target_df['Close'].pct_change().shift(-1).rolling(window=horizon).std() * 100
                            elif target_method == "Future ATR":
                                high_low = target_df['High'] - target_df['Low']
                                high_close = np.abs(target_df['High'] - target_df['Close'].shift(1))
                                low_close = np.abs(target_df['Low'] - target_df['Close'].shift(1))
                                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                                target_df['target'] = true_range.shift(-1).rolling(window=horizon).mean()
                            elif target_method == "Future Volume Change (%)":
                                if 'Volume' in target_df.columns:
                                    target_df['target'] = (target_df['Volume'].shift(-horizon) - target_df['Volume']) / target_df['Volume'] * 100
                                else:
                                    st.error("❌ Volume column not available!")
                                    st.stop()

                            # Apply log transformation for continuous targets
                            if use_log_transform:
                                # Check if target represents percentage returns (like Future Return %)
                                is_percentage_return = target_method in ["Future Return (%)", "Future Volume Change (%)"]

                                if is_percentage_return:
                                    # Convert percentage to decimal: 2.5% → 0.025, then apply log(1 + r)
                                    # This is the CORRECT way for returns!
                                    decimal_return = target_df['target'] / 100.0  # 2.5 → 0.025
                                    target_df['target_log'] = np.log1p(decimal_return)  # log(1 + 0.025)
                                    st.info("ℹ️ Applied log(1 + r) transformation for returns (r = target/100)")
                                    st.info(f"📊 Example: 2.5% → log(1 + 0.025) = {np.log1p(0.025):.6f}")
                                else:
                                    # For non-percentage targets (price changes, volatility, ATR)
                                    has_negative = (target_df['target'] < 0).any()

                                    if has_negative:
                                        # For price changes that can be negative
                                        # Shift to make all positive, then log
                                        min_val = target_df['target'].min()
                                        shift_val = abs(min_val) + 1
                                        target_df['target_log'] = np.log(target_df['target'] + shift_val)
                                        st.info(f"ℹ️ Applied log(target + {shift_val:.2f}) - shifted to positive range")
                                    else:
                                        # Simple log for positive values (volatility, ATR, etc.)
                                        has_zero = (target_df['target'] == 0).any()
                                        if has_zero or (target_df['target'].min() < 1):
                                            target_df['target_log'] = np.log1p(target_df['target'])
                                            st.info("ℹ️ Applied log(1 + target) transformation")
                                        else:
                                            target_df['target_log'] = np.log(target_df['target'])
                                            st.info("ℹ️ Applied log(target) transformation")

                                # Keep both original and log-transformed target
                                target_df['target_original'] = target_df['target']
                                target_df['target'] = target_df['target_log']

                        # CATEGORICAL TARGETS
                        elif target_type == "Categorical (Multi-Class)":
                            future_return = (target_df['Close'].shift(-horizon) - target_df['Close']) / target_df['Close'] * 100

                            if target_method == "Trend Direction (Up/Down/Sideways)":
                                target_df['target'] = pd.cut(future_return, bins=[-np.inf, down_threshold, up_threshold, np.inf], labels=[0, 1, 2]).astype(int)
                            elif target_method == "Return Bins (Strong Down/Down/Neutral/Up/Strong Up)":
                                target_df['target'] = pd.cut(future_return, bins=[-np.inf, threshold_1, threshold_2, threshold_3, threshold_4, np.inf], labels=[0, 1, 2, 3, 4]).astype(int)
                            elif target_method == "Volatility Regime (Low/Medium/High)":
                                vol = target_df['Close'].pct_change().shift(-1).rolling(window=horizon).std() * 100
                                vol_low = vol.quantile(low_percentile / 100)
                                vol_high = vol.quantile(high_percentile / 100)
                                target_df['target'] = pd.cut(vol, bins=[-np.inf, vol_low, vol_high, np.inf], labels=[0, 1, 2]).astype(int)
                            elif target_method == "Price vs MA (Below/At/Above)":
                                ma = target_df['Close'].rolling(window=ma_period).mean()
                                future_close = target_df['Close'].shift(-horizon)
                                pct_diff = (future_close - ma) / ma * 100
                                target_df['target'] = pd.cut(pct_diff, bins=[-np.inf, -tolerance, tolerance, np.inf], labels=[0, 1, 2]).astype(int)
                            elif target_method == "Custom Quantile Bins":
                                target_df['target'] = pd.qcut(future_return, q=n_bins, labels=list(range(n_bins)), duplicates='drop').astype(int)

                        # BINARY TARGETS
                        else:  # Binary
                            if target_method == "Price Up/Down (Simple Binary)":
                                future_return = (target_df['Close'].shift(-horizon) - target_df['Close']) / target_df['Close'] * 100
                                target_df['target'] = (future_return > 0).astype(int)
                            elif target_method == "Return Exceeds Threshold":
                                future_return = (target_df['Close'].shift(-horizon) - target_df['Close']) / target_df['Close'] * 100
                                target_df['target'] = (np.abs(future_return) > threshold).astype(int)
                            elif target_method == "Breakout (Price > MA)":
                                ma = target_df['Close'].rolling(window=ma_period).mean()
                                future_close = target_df['Close'].shift(-horizon)
                                target_df['target'] = (future_close > ma).astype(int)
                            elif target_method == "Breakdown (Price < MA)":
                                ma = target_df['Close'].rolling(window=ma_period).mean()
                                future_close = target_df['Close'].shift(-horizon)
                                target_df['target'] = (future_close < ma).astype(int)
                            elif target_method == "Take Profit Hit":
                                target_series = pd.Series(0, index=target_df.index)
                                for i in range(1, horizon + 1):
                                    future_return = (target_df['Close'].shift(-i) - target_df['Close']) / target_df['Close'] * 100
                                    target_series = target_series | (future_return >= tp_pct)
                                target_df['target'] = target_series.astype(int)
                            elif target_method == "Stop Loss Hit":
                                target_series = pd.Series(0, index=target_df.index)
                                for i in range(1, horizon + 1):
                                    future_return = (target_df['Close'].shift(-i) - target_df['Close']) / target_df['Close'] * 100
                                    target_series = target_series | (future_return <= -sl_pct)
                                target_df['target'] = target_series.astype(int)
                            elif target_method == "High Volatility Event":
                                vol = target_df['Close'].pct_change().shift(-1).rolling(window=horizon).std() * 100
                                vol_threshold_val = vol.quantile(vol_percentile / 100)
                                target_df['target'] = (vol > vol_threshold_val).astype(int)

                        # Remove rows with NaN targets
                        original_len = len(target_df)
                        target_df = target_df.dropna(subset=['target'])
                        rows_removed = original_len - len(target_df)

                        st.session_state.feature_df = target_df
                        st.success(f"✅ Target variable '{target_method}' generated!")
                        st.info(f"Rows removed (future lookahead): {rows_removed} ({rows_removed/original_len*100:.1f}%)")

                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(target_df))

                        if target_type == "Continuous (Regression)":
                            with col2:
                                if use_log_transform and 'target_original' in target_df.columns:
                                    st.metric("Mean Target (Log)", f"{target_df['target'].mean():.4f}")
                                else:
                                    st.metric("Mean Target", f"{target_df['target'].mean():.4f}")
                            with col3:
                                if use_log_transform and 'target_original' in target_df.columns:
                                    st.metric("Std Target (Log)", f"{target_df['target'].std():.4f}")
                                else:
                                    st.metric("Std Target", f"{target_df['target'].std():.4f}")

                            # Show original target statistics if log-transformed
                            if use_log_transform and 'target_original' in target_df.columns:
                                st.markdown("**Original Target Statistics (before log):**")
                                col1_orig, col2_orig = st.columns(2)
                                with col1_orig:
                                    st.metric("Mean Original", f"{target_df['target_original'].mean():.4f}")
                                with col2_orig:
                                    st.metric("Std Original", f"{target_df['target_original'].std():.4f}")
                        else:
                            value_counts = target_df['target'].value_counts().sort_index()
                            with col2:
                                st.metric("Unique Classes", len(value_counts))
                            with col3:
                                balance = (value_counts.min() / value_counts.max() * 100)
                                st.metric("Class Balance", f"{balance:.1f}%")
                            if balance < 30:
                                st.warning(f"⚠️ Class Imbalance: {balance:.1f}% - Consider SMOTE or class weights!")

                        with st.expander("📋 Preview Data with Target"):
                            if use_log_transform and 'target_original' in target_df.columns and target_type == "Continuous (Regression)":
                                # Show both original and log-transformed
                                preview_cols = ['Close', 'target_original', 'target']
                                preview_df = target_df[preview_cols].tail(20).copy()
                                preview_df.columns = ['Close', 'Target (Original)', 'Target (Log)']
                                st.dataframe(preview_df)
                            else:
                                st.dataframe(target_df[['Close', 'target']].tail(20))


            # ========== STEP 8: SYNTHETIC FEATURES ==========
            if st.session_state.feature_df is not None and st.session_state.step_6_completed:
                st.header("8️⃣ Synthetic Features (Advanced)")
                st.markdown("**Synthetic features** automatically combine existing features - can significantly improve ML Performance through non-linear relationships")

                # Get current feature_df
                current_df = st.session_state.feature_df

                # Feature selection for synthetic generation
                st.subheader("Select Base Features")
                st.markdown("Choose which features to use as base for synthetic feature generation (too many = explosion!)")

                available_features = current_df.select_dtypes(include=[np.number]).columns.tolist()

                # Smart defaults: select key features
                default_features = [col for col in available_features if any(
                    key in col for key in ['Close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume']
                )][:15]  # Limit to 15

                selected_base_features = st.multiselect(
                    "Select features for synthetic generation",
                    options=available_features,
                    default=default_features if default_features else available_features[:10],
                    help="Wähle 5-20 Features - zu viele führen zu Feature-Explosion!"
                )

                # Synthetic feature categories
                st.subheader("Select Synthetic Feature Types")

                col1, col2, col3 = st.columns(3)

                # Initialize variables
                poly_degree = 2
                max_interactions = 20
                lag_periods = [1, 2, 3, 5, 10]
                rolling_windows = [3, 5, 10]
                diff_periods = [1, 2, 5]
                n_bins = 5

                with col1:
                    use_polynomial = st.checkbox(
                        "Polynomial Features",
                        value=False,
                        help="Polynomial combinations (A², A*B, B²) - Degree 2 oder 3"
                    )
                    if use_polynomial:
                        poly_degree = st.slider("Polynomial Degree", 2, 3, 2,
                            help="Degree 2: quadratic | Degree 3: cubic (mehr features, langsamer)")

                    use_interactions = st.checkbox(
                        "Feature Interactions",
                        value=False,
                        help="Feature interactions (multiplication, division, addition, subtraction)"
                    )
                    if use_interactions:
                        max_interactions = st.slider("Max Interaction Pairs", 10, 100, 20)

                    use_lag = st.checkbox(
                        "Lag Features",
                        value=False,
                        help="Time-shifted features (lag 1, 2, 3, 5, 10)"
                    )
                    if use_lag:
                        lag_periods_input = st.text_input("Lag Periods (comma-separated)", value="1,2,3,5,10")
                        lag_periods = [int(x.strip()) for x in lag_periods_input.split(',')]

                with col2:
                    use_rolling_stats = st.checkbox(
                        "Rolling Statistics",
                        value=False,
                        help="Apply rolling mean/std/min/max auf features"
                    )
                    if use_rolling_stats:
                        rolling_windows_input = st.text_input("Rolling Windows", value="3,5,10")
                        rolling_windows = [int(x.strip()) for x in rolling_windows_input.split(',')]

                    use_diff = st.checkbox(
                        "Difference Features",
                        value=False,
                        help="Generate difference und percentage change features"
                    )
                    if use_diff:
                        diff_periods_input = st.text_input("Diff Periods", value="1,2,5")
                        diff_periods = [int(x.strip()) for x in diff_periods_input.split(',')]

                    use_ratio = st.checkbox(
                        "Ratio to Close",
                        value=False,
                        help="Create ratios of all features to Close price"
                    )

                with col3:
                    use_math_transforms = st.checkbox(
                        "Mathematical Transforms",
                        value=False,
                        help="Log, Sqrt, Square, Cube, Inverse transformations"
                    )

                    use_cumulative = st.checkbox(
                        "Cumulative Features",
                        value=False,
                        help="Cumulative sum and product"
                    )

                    use_binning = st.checkbox(
                        "Binning Features",
                        value=False,
                        help="Create binned/categorical versions (quantile-based)"
                    )
                    if use_binning:
                        n_bins = st.slider("Number of Bins", 3, 10, 5)

                st.markdown("**Advanced**: Genetic algorithms automatically develop complex mathematical feature combinations (30 seconds to 2 minutes depending on settings):")

                # gplearn Section
                use_gplearn = st.checkbox(
                    "Genetic Programming Features (with gplearn)",
                    value=False,
                    help="Use genetic algorithms to evolve new features from mathematical combinations"
                )
            

                if use_gplearn:
                    col_gp1, col_gp2, col_gp3 = st.columns(3)
                    with col_gp1:
                        gp_n_components = st.slider(
                            "Number of GP Features",
                            5, 50, 10,
                            help="Wie viele evolved Features? 10=Standard, mehr=langsamer"
                        )
                    with col_gp2:
                        gp_generations = st.slider(
                            "Generations",
                            10, 50, 20,
                            help="Evolution-Runden: mehr=bessere Features aber langsamer (20=optimal)"
                        )
                    with col_gp3:
                        gp_population = st.slider(
                            "Population Size",
                            500, 2000, 1000,
                            step=100,
                            help="Größe des genetischen Pools - größer=mehr Diversity aber langsamer (1000=optimal)"
                        )

                if st.button("🧬 Generate Synthetic Features", type="primary"):
                    if not selected_base_features:
                        st.error("❌ Please select at least one base feature!")
                    else:
                        with st.spinner("🔬 Generating synthetic features... This may take a moment."):

                            # Backup feature_df before adding synthetic features
                            st.session_state.feature_df_backup = current_df.copy()

                            # Initialize synthetic feature engineering
                            sfe = SyntheticFeatureEngineering(current_df)

                            # Apply selected synthetic feature types
                            if use_polynomial:
                                sfe.add_polynomial_features(degree=poly_degree, selected_features=selected_base_features)

                            if use_interactions:
                                sfe.add_feature_interactions(selected_features=selected_base_features, max_interactions=max_interactions)

                            if use_lag:
                                sfe.add_lag_features(selected_features=selected_base_features, lags=lag_periods)

                            if use_rolling_stats:
                                sfe.add_rolling_statistics_on_features(selected_features=selected_base_features, windows=rolling_windows)

                            if use_diff:
                                sfe.add_diff_features(selected_features=selected_base_features, periods=diff_periods)

                            if use_ratio:
                                sfe.add_ratio_features(base_features=selected_base_features)

                            if use_math_transforms:
                                sfe.add_mathematical_transforms(selected_features=selected_base_features)

                            if use_cumulative:
                                sfe.add_cumulative_features(selected_features=selected_base_features)

                            if use_binning:
                                sfe.add_binning_features(selected_features=selected_base_features, bins=n_bins)

                            # Apply gplearn if selected
                            if use_gplearn:
                                sfe.add_gplearn_features(
                                    selected_features=selected_base_features,
                                    n_components=gp_n_components,
                                    generations=gp_generations,
                                    population_size=gp_population
                                )
                                # Show mode (supervised vs unsupervised)
                                if hasattr(sfe, 'gplearn_mode'):
                                    if 'SUPERVISED' in sfe.gplearn_mode:
                                        st.success(f"✅ gplearn used SUPERVISED mode with target variable for better features!")
                                    else:
                                        st.info(f"ℹ️ gplearn used UNSUPERVISED mode (no target found). Generate target first for better results!")

                            # Get synthetic features dataframe
                            synthetic_df = sfe.get_dataframe()

                            # Smart NaN handling for synthetic features
                            original_shape = synthetic_df.shape

                            # Step 1: Drop rows where OHLC data is missing (critical)
                            ohlc_cols = ['Open', 'High', 'Low', 'Close']
                            synthetic_df = synthetic_df.dropna(subset=ohlc_cols)

                            # Step 2: Forward-fill feature columns
                            feature_cols = [col for col in synthetic_df.columns if col not in ohlc_cols + ['Volume']]
                            if feature_cols:
                                synthetic_df[feature_cols] = synthetic_df[feature_cols].ffill()

                            # Step 3: Backward-fill remaining NaNs
                            if feature_cols:
                                synthetic_df[feature_cols] = synthetic_df[feature_cols].bfill()

                            # Step 4: Drop rows with >50% NaNs
                            threshold = int(len(synthetic_df.columns) * 0.5)
                            synthetic_df = synthetic_df.dropna(thresh=threshold)

                            # Update session state
                            st.session_state.feature_df = synthetic_df
                            st.session_state.step_6_5_completed = True  # Mark Step 6.5 as completed

                            # Calculate new features added
                            new_features_count = synthetic_df.shape[1] - current_df.shape[1]
                            rows_removed = original_shape[0] - synthetic_df.shape[0]

                            st.success(f"✅ Synthetic features generated successfully!")
                            st.info(f"Added {new_features_count} new synthetic features, Original Features: {current_df.shape[1]} | New Features: {new_features_count} | Total: {synthetic_df.shape[1]} ")

                            # Show what was added - NO NESTED EXPANDERS
                            st.markdown("### 🔍 Newly Added Synthetic Features")
                            new_cols = [col for col in synthetic_df.columns if col not in current_df.columns]

                            if new_cols:
                                col1, col2 = st.columns(2)
                                half = len(new_cols) // 2

                                with col1:
                                    for col in new_cols[:half]:
                                        st.text(f"  • {col}")

                                with col2:
                                    for col in new_cols[half:]:
                                        st.text(f"  • {col}")
                            else:
                                st.write("No new features added (possibly all removed due to NaN)")

                # Show results in separate expanders (outside the button click)
                if 'synthetic_features_generated' not in st.session_state:
                    st.session_state.synthetic_features_generated = False

                # After generation, show preview and stats
                if st.session_state.feature_df is not None and st.session_state.feature_df.shape[1] > len(current_df.columns):
                    with st.expander("📋 Preview Current Features"):
                        st.dataframe(st.session_state.feature_df.head(10))

                    with st.expander("📊 Synthetic Feature Analysis", expanded=False):
                        create_feature_overview_charts(st.session_state.feature_df)

                    with st.expander("📈 Current Statistics"):
                        st.dataframe(st.session_state.feature_df.describe())

                # Undo Synthetic Features button
                if st.session_state.step_6_5_completed and st.session_state.feature_df_backup is not None:
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("↩️ Undo Synthetic Features"):
                            st.session_state.feature_df = st.session_state.feature_df_backup.copy()
                            st.session_state.step_6_5_completed = False
                            st.success("✅ Synthetic features removed - back to base features!")
                            st.rerun()

                st.markdown("---")


            # ========== STEP 9: DOWNLOAD ==========
            if st.session_state.feature_df is not None:
                st.header("9️⃣ Download Features")
                st.session_state.step_7_completed = True

                            # ========== FEATURE-TARGET ANALYSIS CHARTS ==========
                if st.session_state.feature_df is not None and 'target' in st.session_state.feature_df.columns:
                    with st.expander("📊 Feature-Target Analysis Charts", expanded=False):
                        st.info("""
                    **Feature-Target Analysis** helps you understand which features are most predictive of your target variable.
                    - **Correlation Chart**: Shows which features have the strongest relationship with your target
                    - **Distribution Analysis**: Visualizes how feature values differ across target values
                    - **Scatter Plots**: Detailed view of top features vs target with trend lines
                    """)
                        create_feature_target_analysis_charts(st.session_state.feature_df, target_col='target', top_n=10)

                feature_df = st.session_state.feature_df

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Features", feature_df.shape[1])

                with col2:
                    st.metric("Total Rows", feature_df.shape[0])

                with col3:
                    missing_pct = (feature_df.isnull().sum().sum() / (feature_df.shape[0] * feature_df.shape[1]) * 100)
                    st.metric("Missing Values %", f"{missing_pct:.2f}%")

                # Download options
                st.markdown("### Download Options")

                filename = st.text_input("Filename", value="ml_features.csv")

                include_index = st.checkbox("Include index (date/time)", value=True)

                # Convert to CSV
                csv_buffer = io.StringIO()
                feature_df.to_csv(csv_buffer, index=include_index)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download full CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    type="primary"
                )

                st.success("✅ Your ML-ready dataset is ready for download! Thanks for using ;)")

                # Final preview
                with st.expander("🔍 Final Preview"):
                    st.dataframe(feature_df.head(20))

    else:
        st.info("👆 Upload a CSV file to get started!")

        # Show example format
        st.markdown("### 📝 Expected CSV Format")
        st.code("Date,Open,High,Low,Close,Volume\n2024-01-01,100.0,105.0,99.0,103.0,1000000\n2024-01-02,103.0,106.0,102.0,105.0,1200000", language="csv")


if __name__ == "__main__":
    main()
