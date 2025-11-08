# 📊 ML Finance Feature Configurator

A comprehensive Streamlit tool for transforming OHLC data into ML-ready feature sets with full control over feature selection, scaling, and parameters.

Find a demo on https://ml-feature-generator.streamlit.app/

## 🚀 Features

### 1. CSV Upload & Column Mapping
- Flexible column mapping for various CSV formats
- Support for OHLC + Volume + Date
- Automatic date parsing

### 2. Data Scaling (Optional)
- **StandardScaler**: (x - mean) / std
- **MinMaxScaler**: (x - min) / (max - min)
- **RobustScaler**: Uses median and IQR (robust against outliers)

### 3. Comprehensive Feature Categories

#### Price-Based Features
- Range (High - Low)
- Close/Open Ratio
- High/Low Ratio
- Mid-Price

#### Returns
- Daily Return
- Log Return
- High-Low Return
- Open-Close Return

#### Moving Averages
- **SMA**: Simple Moving Average (configurable windows)
- **EMA**: Exponential Moving Average
- **WMA**: Weighted Moving Average
- **HMA**: Hull Moving Average

#### Volatility
- Rolling Standard Deviation
- **ATR**: Average True Range

#### Momentum Indicators
- Momentum (Price difference)
- **ROC**: Rate of Change
- **RSI**: Relative Strength Index (14, 21)
- **Stochastic Oscillator**: %K and %D
- **Williams %R**: Momentum indicator
- **CCI**: Commodity Channel Index
- **CMO**: Chande Momentum Oscillator

#### Trend Indicators
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: With %B and Bandwidth
- **Keltner Channel**: ATR-based
- **ADX & DMI**: Average Directional Index
- **Parabolic SAR**: Stop and Reverse
- **SuperTrend**: With Direction Signal

#### Japanese Indicators
- **Ichimoku Cloud**: Tenkan, Kijun, Senkou Spans (no future bias!)
- **Awesome Oscillator**: Midprice-based

#### Candlestick Features
- Body Length
- Upper/Lower Shadow
- Body/Range Ratios
- Candle Direction

#### Volume Features (when available)
- Volume Change
- Volume SMA
- **OBV**: On-Balance Volume
- **VWAP**: Volume Weighted Average Price
- **MFI**: Money Flow Index

#### Statistical Features
- Rolling Mean/Median/Std
- **Z-Score**: Standardized deviation
- Skewness & Kurtosis

#### Time Features
- Day of Week/Month
- Week of Month
- Month/Quarter
- Cyclical Encoding (sin/cos)

#### Additional Features
- Cumulative Returns
- MA Crossovers

### 4. 🧬 Synthetic Features (Advanced)

**NEW!** Automatic generation of synthetic features through combinations and transformations:

#### Polynomial Features
- Quadratic and cubic combinations
- Example: For features [A, B] → A², A*B, B²
- Degree 2 or 3 selectable

#### Feature Interactions
- **Multiplication**: A * B
- **Division**: A / B
- **Addition**: A + B
- **Subtraction**: A - B
- **Ratios**: A / B
- Up to 100 interaction pairs configurable

#### Lag Features
- Time-shifted features (historical values)
- Standard lags: 1, 2, 3, 5, 10 (customizable)
- Important for time series ML models

#### Rolling Statistics on Features
- Rolling Mean/Std/Min/Max on existing features
- Windows: 3, 5, 10 (customizable)
- Smoothing and trend detection

#### Difference Features
- Difference: Feature_t - Feature_(t-n)
- Percentage Change: (Feature_t - Feature_(t-n)) / Feature_(t-n)
- Periods: 1, 2, 5 (customizable)

#### Ratio Features
- All features relative to Close price
- Normalization and relative strength

#### Mathematical Transforms
- **Log**: np.log(x) - for positive values
- **Sqrt**: np.sqrt(x) - square root
- **Square**: x²
- **Cube**: x³
- **Inverse**: 1/x

#### Cumulative Features
- Cumulative Sum: Accumulated values
- Cumulative Product: (1 + x).cumprod()
- Trend tracking over time

#### Binning Features
- Quantile-based categorization
- 3-10 bins selectable
- Discretization for ensemble models

#### 🧬 Genetic Programming (gplearn) **NEW!**
- **Automatic feature evolution** through genetic algorithms
- Discovers complex mathematical combinations automatically
- Uses SymbolicTransformer for unsupervised feature generation
- Function set: add, sub, mul, div, sqrt, log, abs, max, min
- Configurable:
  - **N Components**: 5-50 evolved features
  - **Generations**: 10-50 evolution generations
  - **Population Size**: 500-2000 individuals
- ⚠️ **Computationally intensive**: 30 seconds to 2 minutes depending on settings
- Creates features like: `GP_Evolved_1 = sqrt(SMA_20 * log(abs(RSI_14 - Close)))`
- **Best for**: Non-linear patterns, complex feature interactions

**💡 When to use gplearn:**
- When classic features are not sufficient
- With non-linear ML models (XGBoost, Random Forest, Neural Networks)
- For automatic feature discovery
- When you have time (slower than other methods)

**⚠️ Warning**: Synthetic features can lead to feature explosion! Choose 5-20 base features for optimal results.

### 5. 🎯 Target Variable Generation (Optional)

Create target variables for supervised learning:

#### Regression Targets (Continuous)
- **Future Return (%)**: Percentage price change
- **Future Price Change (Absolute)**: Absolute price difference
- **Future High**: Maximum price in period
- **Future Low**: Minimum price in period
- **Future Volatility**: Standard deviation of returns
- **Future ATR**: Average True Range
- **Future Volume Change (%)**: Volume percentage change
- **Log Transformation**: Optional log(1+r) transformation for returns

#### Classification Targets (Categorical)
- **Trend Direction**: Up/Down/Sideways (3 classes)
- **Return Bins**: Strong Down/Down/Neutral/Up/Strong Up (5 classes)
- **Volatility Regime**: Low/Medium/High (3 classes)
- **Price vs MA**: Below/At/Above moving average (3 classes)
- **Custom Quantile Bins**: User-defined number of classes

#### Binary Classification Targets
- **Price Up/Down**: Simple binary direction
- **Return Exceeds Threshold**: Absolute return > threshold
- **Breakout**: Price breaks above MA
- **Breakdown**: Price breaks below MA
- **Take Profit Hit**: Price reaches TP level within horizon
- **Stop Loss Hit**: Price hits SL level within horizon
- **High Volatility Event**: Volatility exceeds percentile

**Features:**
- Configurable prediction horizon (1-50 periods)
- Customizable thresholds for all target types
- Automatic handling of class imbalance detection
- Shows target statistics and distribution

### 6. 📊 Feature-Target Analysis Charts **NEW!**

**Comprehensive analysis of feature-target relationships before download:**

#### 1. Feature-Target Correlation Chart
- Shows top N features ranked by absolute correlation strength
- Color-coded: Green (positive) / Red (negative)
- Helps identify most predictive features
- Sorted by correlation strength for easy interpretation

#### 2. Feature Distributions by Target
- **For Classification**: Box plots showing feature distributions across target classes
- **For Regression**: Scatter plots with color-coded target values
- Visualizes how features separate different target values
- Displays top 6 most correlated features

#### 3. Detailed Feature vs Target Analysis
- 2x2 grid of detailed scatter plots
- **For Regression**: Includes trend lines to show relationship direction
- **For Classification**: Color-coded by target class
- Shows correlation coefficient for each feature
- Top 4 most predictive features

#### 4. Feature-Target Statistics Summary
- Tabular summary of correlations, means, and standard deviations
- Missing value percentages
- Absolute correlation for ranking
- Quick overview of top predictive features

**Benefits:**
- Understand which features are most important before training
- Detect non-linear relationships
- Identify features that separate target classes well
- Make informed decisions about feature selection
- Validate target variable quality

### 7. Variable Window Lengths
- Configurable windows for MA, Momentum and Statistical features
- Default: 5, 10, 20, 50, 100, 200 (customizable)

### 8. Feature Summary & Statistics
- Automatic feature summary
- Descriptive statistics
- Missing values report

### 9. Download
- CSV export with all features
- Optional index inclusion (for time series)
- Customizable filename

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

```bash
# Start the Streamlit app
streamlit run main.py
```

## 📝 Workflow

### Step 1: CSV Upload
Upload your OHLC CSV file. Format example:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,106.0,102.0,105.0,1200000
```

### Step 2: Column Mapping
Map your CSV columns to standard OHLC format:
- **Required**: Open, High, Low, Close
- **Optional**: Volume, Date

### Step 3: Data Scaling (Optional)
Choose a scaling method and columns to scale.

### Step 4: Window Length Configuration
Adjust window lengths for rolling calculations:
- Moving Averages: e.g., 5,10,20,50,100,200
- Momentum: e.g., 5,10,20
- Statistical: e.g., 5,10,20,50

### Step 5: Feature Selection
Choose from 24+ feature categories:
- ✅ Select All / ❌ Deselect All
- Checkbox for each category with description

### Step 6: Generate Features
Click "🚀 Generate Features" and wait for calculation.

### Step 7: Synthetic Features (Optional)
**NEW!** Generate advanced synthetic features:
1. Select 5-20 base features from generated features
2. Choose synthetic feature types:
   - Polynomial Features (Degree 2 or 3)
   - Feature Interactions (Mult, Div, Add, Sub, Ratio)
   - Lag Features (historical values)
   - Rolling Statistics (Mean, Std, Min, Max)
   - Difference Features (Diff, Pct Change)
   - Ratio to Close
   - Mathematical Transforms (Log, Sqrt, Square, Cube, Inverse)
   - Cumulative Features (CumSum, CumProd)
   - Binning Features (Quantile-based)
   - Genetic Programming (gplearn) - Automatic feature evolution
3. Click "🧬 Generate Synthetic Features"
4. Get hundreds of additional features!

### Step 8: Target Variable (Optional)
**NEW!** Create target variables for supervised learning:
1. Choose your ML problem type:
   - **Continuous (Regression)**: Predict exact values
   - **Categorical (Multi-Class)**: Predict categories
   - **Binary (Classification)**: Predict yes/no outcomes
2. Select target method (7-8 options per type)
3. Configure prediction horizon and thresholds
4. Optional: Apply log transformation for regression targets
5. Generate target and view statistics

### Step 9: Feature-Target Analysis (Optional)
**NEW!** Analyze feature-target relationships:
- View correlation rankings of top 10 features
- Examine feature distributions across target values
- Inspect scatter plots with trend lines
- Review detailed statistics table
- **Only available when target variable exists**

### Step 10: Download
Download your ML-ready CSV with all features!

## 🎨 Features in Detail

### No Future Bias

All features are implemented without **future bias**:
- Ichimoku: No forward-shifting of spans
- All Rolling Calculations: Only past data
- Safe for backtesting and training

### NaN Handling

After feature generation, all rows with NaN values are automatically removed (rolling windows create NaNs at the beginning).

For synthetic features, smart NaN handling is applied:
- Drop rows where OHLC data is missing (critical)
- Forward-fill feature columns
- Backward-fill remaining NaNs
- Drop rows with >50% NaNs

## 🧪 Example Feature Output

After generation, you'll get a CSV with columns like:

```
Open, High, Low, Close, Volume,
Range, Close_Open_Ratio,
SMA_20, EMA_50,
RSI_14, Stochastic_%K_14,
MACD_12_26, MACD_Signal_12_26_9, MACD_Hist_12_26_9,
BB_Upper_20, BB_Middle_20, BB_Lower_20, BB_PctB_20,
ATR_14, Volatility_20,
Ichimoku_Tenkan_9, Ichimoku_Kijun_26,
ADX_14, Plus_DI_14, Minus_DI_14,
OBV, VWAP, MFI_14,
target  # if generated
...
```

## 🧬 Synthetic Features - Best Practices

### When to Use Synthetic Features?

**Use synthetic features when:**
- Working with non-linear ML models (Random Forest, XGBoost, Neural Networks)
- Want to automatically discover feature interactions
- Need to capture temporal dependencies (Lag Features)
- Need more data (Data Augmentation)

**Avoid synthetic features when:**
- Working with linear models (can lead to multicollinearity)
- Dataset already has many features (>1000)
- Need fast inference (more features = slower prediction)

### Avoiding Feature Explosion

🔥 **Warning**: Synthetic features can grow exponentially!

**Example**:
- 10 base features with Polynomial Degree 2 → ~55 new features
- 10 base features with Interactions → ~200 new features
- 10 base features with Lag 5 → 50 new features

**Best Practice**:
1. Start with 5-10 important base features (e.g., SMA_20, RSI_14, MACD)
2. Test individual synthetic feature types
3. Use Feature Importance after training to remove irrelevant features
4. Combine max. 2-3 synthetic types simultaneously

### Recommended Combinations

**For Time Series Prediction:**
```
Base Features: Close, SMA_20, EMA_50, RSI_14, Volume
Synthetic: Lag Features (1,2,3,5) + Difference Features (1,2)
```

**For Trend Detection:**
```
Base Features: SMA_20, EMA_50, MACD, ADX_14
Synthetic: Feature Interactions + Ratio to Close
```

**For Volatility Trading:**
```
Base Features: ATR_14, BB_Width_20, Volatility_20
Synthetic: Rolling Statistics (3,5,10) + Mathematical Transforms
```

**For Automatic Feature Discovery (Advanced):**
```
Base Features: Close, Volume, SMA_20, RSI_14, MACD_12_26, ATR_14
Synthetic: gplearn (10-20 components, 20 generations)
→ Let genetic algorithms find optimal combinations!
```

## 🎯 Target Variable - Best Practices

### Choosing the Right Target Type

**Regression (Continuous):**
- Best for: Price prediction, volatility forecasting, return prediction
- Use log transformation for returns to stabilize variance
- Examples: Future Return %, Future ATR, Future Volatility

**Multi-Class Classification:**
- Best for: Trend direction, regime detection, strategy selection
- Ensure balanced classes (check class balance metric)
- Use SMOTE or class weights if imbalanced (<30%)
- Examples: Trend Direction (Up/Down/Sideways), Volatility Regime

**Binary Classification:**
- Best for: Simple trading signals, event detection
- Easier to train than multi-class
- Can handle class imbalance better
- Examples: Price Up/Down, Take Profit Hit, High Volatility Event

### Prediction Horizon Selection

- **Short horizon (1-5 periods)**: Harder to predict, more noise, more trading opportunities
- **Medium horizon (5-20 periods)**: Good balance of predictability and actionability
- **Long horizon (20-50 periods)**: Easier to predict, smoother trends, fewer opportunities

### Log Transformation for Returns

For regression targets representing returns:
- Transforms: `target = log(1 + return/100)`
- Makes distribution more symmetric
- Stabilizes variance
- Standard practice in quantitative finance
- To reverse: `return = (exp(target) - 1) * 100`

## 💡 Tips

1. **Start Small**: Choose few features initially for quick tests
2. **Use Scaling**: Scaling before feature calculation can be beneficial
3. **Window Tuning**: Adjust window lengths to your timeframe
4. **Check Statistics**: Review feature statistics for outliers
5. **Volume Check**: Only enable volume features if volume data is available
6. **Synthetic Features**: Start with 5-10 base features, test feature importance
7. **Lag Features**: Especially important for LSTM/RNN models
8. **Feature Selection**: Use correlation matrix to remove redundant features
9. **gplearn**: Use as last step for automatic feature discovery (slow!)
10. **gplearn Settings**: Start with 10 components, 20 generations. Increase if needed
11. **Target Variable**: Choose prediction horizon based on your trading timeframe
12. **Log Transform**: Always use for return-based regression targets
13. **Feature-Target Analysis**: Review correlation charts before training to validate features
14. **Class Imbalance**: Address if balance <30% using SMOTE or class weights

## 🔧 Customization

The tool is modular. You can:
- Add new features to the `FeatureEngineering` class
- Add new synthetic features to the `SyntheticFeatureEngineering` class
- Add new target methods in the target generation section
- Modify window defaults
- Extend scaler options
- Customize UI layout
- Add new chart types to `charts.py`

## 📊 Output Metrics

The tool shows:
- Total Features (number of columns)
- Total Rows (after NaN removal)
- Missing Values Percentage
- Feature summary grouped by categories
- Target statistics (when target exists)
- Class balance (for classification targets)

## 🎯 Use Cases

- **ML Training**: Create feature sets for price prediction
- **Feature Engineering Exploration**: Test different feature combinations
- **Backtesting**: Prepare data for strategy backtests
- **Data Analysis**: Extend OHLC data with technical indicators
- **Research**: Discover new feature combinations with gplearn
- **Production ML**: Generate consistent features for model deployment

## 📚 Technical Indicators & Features Included

### Standard Technical Indicators
- **Moving Averages**: SMA, EMA, WMA, HMA
- **Oscillators**: RSI, Stochastic, Williams %R, CCI, CMO, AO
- **Trend Followers**: MACD, ADX, DMI, Parabolic SAR, SuperTrend
- **Volatility**: ATR, Bollinger Bands, Keltner Channel
- **Volume**: OBV, VWAP, MFI
- **Japanese**: Ichimoku Cloud
- **Statistics**: Z-Score, Skewness, Kurtosis
- **Candlestick**: Body/Shadow Ratios

### Synthetic Features (NEW!)
- **Polynomial**: Quadratic & Cubic combinations
- **Interactions**: Mult, Div, Add, Sub, Ratios
- **Temporal**: Lag Features, Difference Features
- **Statistical**: Rolling aggregations on features
- **Mathematical**: Log, Sqrt, Square, Cube, Inverse
- **Cumulative**: CumSum, CumProd
- **Discretization**: Quantile-based Binning
- **🧬 Genetic Programming (gplearn)**: Automatically evolved features using genetic algorithms

### Target Variables (NEW!)
- **Regression**: 7 continuous target types
- **Multi-Class**: 5 categorical target types
- **Binary**: 7 binary classification target types
- **Log Transformation**: For return-based targets
- **Customizable**: Horizons, thresholds, windows

### Analysis Charts (NEW!)
- **Correlation Analysis**: Feature-target correlation rankings
- **Distribution Analysis**: Box plots and scatter plots
- **Trend Analysis**: Scatter plots with trend lines
- **Statistics Summary**: Comprehensive feature statistics table

## ⚠️ Notes

- Rolling calculations require warm-up period (first N rows will be NaN)
- More features = longer calculation time
- Very long windows (e.g., SMA_200) require at least 200+ data points
- Volume features require Volume column in data
- **Synthetic Features**: Can lead to 100+ additional features - feature selection recommended!
- **Lag Features**: Reduce number of available data points (Lag 10 = 10 fewer rows)
- **Polynomial Features**: Degree 3 can be very computationally intensive with many features
- **Target Generation**: Removes last N rows where target cannot be calculated (N = horizon)
- **Feature-Target Analysis**: Only available when target variable exists

## 🤝 Contributing

Feel free to extend this tool with:
- New technical indicators
- Additional scaling methods
- Custom feature combinations
- Export formats (Parquet, Excel, etc.)
- New target variable types
- Additional analysis charts

Feel free to contact me: constantinwilharm@gmail.com!

## 📄 License

Open Source - Use freely for your ML projects!

---

**Happy Feature Engineering! 🚀📊🤖**
