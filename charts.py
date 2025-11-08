"""
Visualization functions for ML Feature Configurator
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List


def create_ohlc_chart(df: pd.DataFrame, max_points: int = 500) -> go.Figure:
    """Create a compact OHLC candlestick chart"""
    # Sample data if too large
    if len(df) > max_points:
        df = df.iloc[-max_points:]

    fig = go.Figure(data=[go.Candlestick(
        x=df.index if not df.index.name else df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    )])

    fig.update_layout(
        title='Price Chart (OHLC)',
        xaxis_title='Time',
        yaxis_title='Price',
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    return fig


def create_volume_chart(df: pd.DataFrame, max_points: int = 500) -> go.Figure:
    """Create a compact volume bar chart"""
    if 'Volume' not in df.columns:
        return None

    # Sample data if too large
    if len(df) > max_points:
        df = df.iloc[-max_points:]

    fig = go.Figure(data=[go.Bar(
        x=df.index if not df.index.name else df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    )])

    fig.update_layout(
        title='Volume',
        xaxis_title='Time',
        yaxis_title='Volume',
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    return fig


def create_feature_correlation_heatmap(df: pd.DataFrame, max_features: int = 20) -> go.Figure:
    """Create a compact correlation heatmap for features"""
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Limit to max_features
    if len(numeric_cols) > max_features:
        numeric_cols = numeric_cols[:max_features]

    corr = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f'Feature Correlation Heatmap (Top {len(numeric_cols)} features)',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_feature_distribution_chart(df: pd.DataFrame, columns: List[str] = None, max_cols: int = 6) -> go.Figure:
    """Create distribution histograms for selected features"""
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols[:max_cols]

    # Create subplots
    n_cols = min(len(columns), max_cols)
    rows = (n_cols + 2) // 3  # 3 columns per row
    cols = min(n_cols, 3)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=columns[:n_cols]
    )

    for idx, col in enumerate(columns[:n_cols]):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1

        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False, nbinsx=30),
            row=row, col=col_pos
        )

    fig.update_layout(
        title='Feature Distributions',
        height=200 * rows,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    return fig


def create_data_overview_charts(df: pd.DataFrame):
    """Display overview charts for the data"""
    # OHLC Chart
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig_ohlc = create_ohlc_chart(df)
        st.plotly_chart(fig_ohlc, use_container_width=True)

    # Volume Chart
    if 'Volume' in df.columns:
        fig_volume = create_volume_chart(df)
        if fig_volume:
            st.plotly_chart(fig_volume, use_container_width=True)


def create_feature_overview_charts(df: pd.DataFrame):
    """Display overview charts for features"""
    col1, col2 = st.columns(2)

    with col1:
        # Distribution of first few features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        if numeric_cols:
            fig_dist = create_feature_distribution_chart(df, numeric_cols)
            st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Correlation heatmap
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            fig_corr = create_feature_correlation_heatmap(df)
            st.plotly_chart(fig_corr, use_container_width=True)


def create_feature_target_analysis_charts(df: pd.DataFrame, target_col: str = 'target', top_n: int = 10):
    """
    Create comprehensive feature-target analysis charts.

    Args:
        df: DataFrame with features and target
        target_col: Name of target column (default: 'target')
        top_n: Number of top features to display (default: 10)
    """
    # Check if target exists
    if target_col not in df.columns:
        st.warning(f"⚠️ Target column '{target_col}' not found in dataframe. Skipping analysis.")
        return

    # Get numeric features (exclude OHLC, Volume, and target itself)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', target_col, 'target_original', 'target_log']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

    if not numeric_cols:
        st.warning("⚠️ No features found for analysis.")
        return

    # Calculate correlations with target
    correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    abs_correlations = correlations.abs().sort_values(ascending=False)
    top_features = abs_correlations.head(top_n).index.tolist()

    # Determine if target is continuous or categorical
    unique_target_values = df[target_col].nunique()
    is_categorical = unique_target_values < 10  # Assume categorical if < 10 unique values

    st.markdown(f"**Target Type:** {'Classification' if is_categorical else 'Regression'}")
    st.markdown(f"**Analyzing Top {len(top_features)} Features (by correlation with target)**")

    # === CHART 1: Feature-Target Correlation Bar Chart ===
    st.markdown("### 1️⃣ Feature-Target Correlation")

    # Sort by absolute correlation (strongest correlations at top)
    top_correlations = correlations[top_features]
    top_correlations_sorted = top_correlations.reindex(
        top_correlations.abs().sort_values(ascending=True).index
    )
    colors = ['red' if x < 0 else 'green' for x in top_correlations_sorted]

    fig_corr = go.Figure(data=[
        go.Bar(
            y=top_correlations_sorted.index,
            x=top_correlations_sorted.values,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{val:.3f}' for val in top_correlations_sorted.values],
            textposition='auto',
        )
    ])

    fig_corr.update_layout(
        title=f'Top {len(top_features)} Features by Absolute Correlation with Target',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        height=max(400, len(top_features) * 25),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # === CHART 2: Feature Distributions by Target ===
    st.markdown("### 2️⃣ Feature Distributions by Target")

    if is_categorical:
        # Box plots for categorical targets
        n_features_to_plot = min(len(top_features), 6)
        rows = (n_features_to_plot + 1) // 2
        cols = 2

        fig_dist = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{feat}" for feat in top_features[:n_features_to_plot]]
        )

        for idx, feature in enumerate(top_features[:n_features_to_plot]):
            row = idx // 2 + 1
            col_pos = idx % 2 + 1

            for target_val in sorted(df[target_col].unique()):
                feature_data = df[df[target_col] == target_val][feature]
                fig_dist.add_trace(
                    go.Box(y=feature_data, name=f'Target={target_val}',
                           showlegend=(idx == 0)),
                    row=row, col=col_pos
                )

        fig_dist.update_layout(
            title='Feature Distributions by Target Class',
            height=300 * rows,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        # Scatter plots for continuous targets (color-coded by target value)
        n_features_to_plot = min(len(top_features), 6)
        rows = (n_features_to_plot + 1) // 2
        cols = 2

        fig_dist = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{feat}" for feat in top_features[:n_features_to_plot]]
        )

        for idx, feature in enumerate(top_features[:n_features_to_plot]):
            row = idx // 2 + 1
            col_pos = idx % 2 + 1

            # Sample data if too large
            plot_df = df[[feature, target_col]].dropna()
            if len(plot_df) > 1000:
                plot_df = plot_df.sample(1000, random_state=42)

            fig_dist.add_trace(
                go.Scatter(
                    x=plot_df[feature],
                    y=plot_df[target_col],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=plot_df[target_col],
                        colorscale='Viridis',
                        showscale=(idx == 0)
                    ),
                    showlegend=False
                ),
                row=row, col=col_pos
            )

            fig_dist.update_xaxes(title_text=feature, row=row, col=col_pos)
            fig_dist.update_yaxes(title_text='Target' if col_pos == 1 else '', row=row, col=col_pos)

        fig_dist.update_layout(
            title='Feature vs Target (Scatter Plots)',
            height=300 * rows,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # === CHART 3: Top Features vs Target Scatter Plots (Detail) ===
    st.markdown("### 3️⃣ Detailed Feature vs Target Analysis")

    n_features_to_plot = min(len(top_features), 4)

    fig_scatter = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{feat} (r={correlations[feat]:.3f})"
                       for feat in top_features[:n_features_to_plot]]
    )

    for idx, feature in enumerate(top_features[:n_features_to_plot]):
        row = idx // 2 + 1
        col_pos = idx % 2 + 1

        # Sample data if too large
        plot_df = df[[feature, target_col]].dropna()
        if len(plot_df) > 1000:
            plot_df = plot_df.sample(1000, random_state=42)

        if is_categorical:
            # Color by target class
            for target_val in sorted(plot_df[target_col].unique()):
                class_data = plot_df[plot_df[target_col] == target_val]
                fig_scatter.add_trace(
                    go.Scatter(
                        x=class_data.index,
                        y=class_data[feature],
                        mode='markers',
                        name=f'Target={target_val}',
                        marker=dict(size=4),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col_pos
                )
        else:
            # Scatter with trend line for regression
            fig_scatter.add_trace(
                go.Scatter(
                    x=plot_df[feature],
                    y=plot_df[target_col],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=plot_df[target_col],
                        colorscale='Viridis',
                        showscale=(idx == 0),
                        colorbar=dict(title="Target") if idx == 0 else None
                    ),
                    showlegend=False
                ),
                row=row, col=col_pos
            )

            # Add trend line
            z = np.polyfit(plot_df[feature].values, plot_df[target_col].values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(plot_df[feature].min(), plot_df[feature].max(), 100)

            fig_scatter.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col_pos
            )

        fig_scatter.update_xaxes(title_text=feature, row=row, col=col_pos)
        fig_scatter.update_yaxes(title_text='Target' if col_pos == 1 else '', row=row, col=col_pos)

    fig_scatter.update_layout(
        title=f'Top {n_features_to_plot} Features vs Target (with trend lines)' if not is_categorical else f'Top {n_features_to_plot} Features by Target Class',
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # === SUMMARY TABLE ===
    st.markdown("### 📊 Feature-Target Statistics Summary")

    summary_data = []
    for feature in top_features:
        corr_val = correlations[feature]
        feature_mean = df[feature].mean()
        feature_std = df[feature].std()

        summary_data.append({
            'Feature': feature,
            'Correlation': f'{corr_val:.4f}',
            'Abs. Correlation': f'{abs(corr_val):.4f}',
            'Mean': f'{feature_mean:.4f}',
            'Std': f'{feature_std:.4f}',
            'Missing %': f'{(df[feature].isna().sum() / len(df) * 100):.2f}%'
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
