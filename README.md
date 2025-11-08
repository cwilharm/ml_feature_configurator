# 📊 ML Feature Configurator

Ein umfassendes Streamlit-Tool zur Transformation von OHLC-Daten in ML-ready Feature-Sets mit voller Kontrolle über Feature-Auswahl, Skalierung und Parameter.

## 🚀 Features

### 1. CSV Upload & Column Mapping
- Flexibles Column-Mapping für verschiedene CSV-Formate
- Unterstützung für OHLC + Volume + Date
- Automatische Datums-Parsing

### 2. Data Scaling (Optional)
- **StandardScaler**: (x - mean) / std
- **MinMaxScaler**: (x - min) / (max - min)
- **RobustScaler**: Verwendet Median und IQR (robust gegen Outliers)

### 3. Umfassende Feature-Kategorien

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
- **SMA**: Simple Moving Average (konfigurierbare Windows)
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
- **Stochastic Oscillator**: %K und %D
- **Williams %R**: Momentum Indikator
- **CCI**: Commodity Channel Index
- **CMO**: Chande Momentum Oscillator

#### Trend Indicators
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Mit %B und Bandwidth
- **Keltner Channel**: ATR-basiert
- **ADX & DMI**: Average Directional Index
- **Parabolic SAR**: Stop and Reverse
- **SuperTrend**: Mit Direction Signal

#### Japanese Indicators
- **Ichimoku Cloud**: Tenkan, Kijun, Senkou Spans (ohne Future Bias!)
- **Awesome Oscillator**: Midprice-basiert

#### Candlestick Features
- Body Length
- Upper/Lower Shadow
- Body/Range Ratios
- Candle Direction

#### Volume Features (wenn verfügbar)
- Volume Change
- Volume SMA
- **OBV**: On-Balance Volume
- **VWAP**: Volume Weighted Average Price
- **MFI**: Money Flow Index

#### Statistical Features
- Rolling Mean/Median/Std
- **Z-Score**: Standardisierte Abweichung
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

**NEU!** Automatische Generierung synthetischer Features durch Kombinationen und Transformationen:

#### Polynomial Features
- Quadratische und kubische Kombinationen
- Beispiel: Für Features [A, B] → A², A*B, B²
- Degree 2 oder 3 wählbar

#### Feature Interactions
- **Multiplication**: A * B
- **Division**: A / B
- **Addition**: A + B
- **Subtraction**: A - B
- **Ratios**: A / B
- Bis zu 100 Interaktions-Paare konfigurierbar

#### Lag Features
- Time-shifted Features (historische Werte)
- Standard Lags: 1, 2, 3, 5, 10 (anpassbar)
- Wichtig für Zeitreihen-ML-Modelle

#### Rolling Statistics on Features
- Rolling Mean/Std/Min/Max auf bestehenden Features
- Windows: 3, 5, 10 (anpassbar)
- Glättung und Trend-Erkennung

#### Difference Features
- Difference: Feature_t - Feature_(t-n)
- Percentage Change: (Feature_t - Feature_(t-n)) / Feature_(t-n)
- Periods: 1, 2, 5 (anpassbar)

#### Ratio Features
- Alle Features im Verhältnis zu Close Price
- Normalisierung und Relative Stärke

#### Mathematical Transforms
- **Log**: np.log(x) - für positive Werte
- **Sqrt**: np.sqrt(x) - Quadratwurzel
- **Square**: x²
- **Cube**: x³
- **Inverse**: 1/x

#### Cumulative Features
- Cumulative Sum: Aufsummierte Werte
- Cumulative Product: (1 + x).cumprod()
- Trend-Tracking über Zeit

#### Binning Features
- Quantile-basierte Kategorisierung
- 3-10 Bins wählbar
- Diskretisierung für Ensemble-Modelle

#### 🧬 Genetic Programming (gplearn) **NEU!**
- **Automatische Feature-Evolution** durch genetische Algorithmen
- Entdeckt komplexe mathematische Kombinationen automatisch
- Verwendet SymbolicTransformer für unsupervised Feature Generation
- Funktions-Set: add, sub, mul, div, sqrt, log, abs, max, min
- Konfiguierbar:
  - **N Components**: 5-50 evolved Features
  - **Generations**: 10-50 Evolution-Generationen
  - **Population Size**: 500-2000 Individuen
- ⚠️ **Rechenintensiv**: 30 Sekunden bis 2 Minuten je nach Settings
- Erstellt Features wie: `GP_Evolved_1 = sqrt(SMA_20 * log(abs(RSI_14 - Close)))`
- **Best for**: Nicht-lineare Patterns, komplexe Feature-Interaktionen

**💡 Wann gplearn nutzen:**
- Wenn klassische Features nicht ausreichen
- Bei nicht-linearen ML-Modellen (XGBoost, Random Forest, Neural Networks)
- Für automatisches Feature Discovery
- Wenn du Zeit hast (ist langsamer als andere Methoden)

**⚠️ Warnung**: Synthetische Features können zu Feature-Explosion führen! Wähle 5-20 Base Features für optimale Ergebnisse.

### 5. Distance to Close Features
Für jedes Feature wird automatisch die "Distance to Close" berechnet:
- Absolute Differenz
- Prozentuale Differenz
- Dies hilft ML-Modellen, relative Positionen besser zu verstehen

### 6. Variable Window Lengths
- Konfigurierbare Windows für MA, Momentum und Statistical Features
- Standard: 5, 10, 20, 50, 100, 200 (anpassbar)

### 7. Feature Summary & Statistics
- Automatische Feature-Zusammenfassung
- Deskriptive Statistiken
- Missing Values Report

### 8. Download
- CSV-Export mit allen Features
- Optionale Index-Inclusion (für Zeitreihen)
- Customizable Filename

## 📦 Installation

```bash
# Installiere Dependencies
pip install -r requirements.txt
```

## 🎯 Usage

```bash
# Starte die Streamlit App
streamlit run feature_configurator.py
```

## 📝 Workflow

### Step 1: CSV Upload
Lade deine OHLC CSV-Datei hoch. Format Beispiel:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.0,105.0,99.0,103.0,1000000
2024-01-02,103.0,106.0,102.0,105.0,1200000
```

### Step 2: Column Mapping
Mappe deine CSV-Spalten zu Standard OHLC-Format:
- **Required**: Open, High, Low, Close
- **Optional**: Volume, Date

### Step 3: Data Scaling (Optional)
Wähle eine Skalierungsmethode und Spalten zum Skalieren.

### Step 4: Window Length Configuration
Passe die Window Lengths für Rolling Calculations an:
- Moving Averages: z.B. 5,10,20,50,100,200
- Momentum: z.B. 5,10,20
- Statistical: z.B. 5,10,20,50

### Step 5: Feature Selection
Wähle aus 24+ Feature-Kategorien:
- ✅ Select All / ❌ Deselect All
- Checkbox für jede Kategorie mit Beschreibung

### Step 6: Generate Features
Klicke auf "🚀 Generate Features" und warte auf die Berechnung.

### Step 6.5: Synthetic Features (Optional)
**NEU!** Generiere fortgeschrittene synthetische Features:
1. Wähle 5-20 Base Features aus den generierten Features
2. Wähle synthetische Feature-Typen:
   - Polynomial Features (Degree 2 oder 3)
   - Feature Interactions (Mult, Div, Add, Sub, Ratio)
   - Lag Features (historische Werte)
   - Rolling Statistics (Mean, Std, Min, Max)
   - Difference Features (Diff, Pct Change)
   - Ratio to Close
   - Mathematical Transforms (Log, Sqrt, Square, Cube, Inverse)
   - Cumulative Features (CumSum, CumProd)
   - Binning Features (Quantile-based)
3. Klicke auf "🧬 Generate Synthetic Features"
4. Erhalte hunderte zusätzliche Features!

### Step 7: Download
Lade deine ML-ready CSV mit allen Features herunter!

## 🎨 Features im Detail

### Distance to Close Berechnungen

Für fast alle Features wird eine "dist_close" Variante berechnet:

```python
# Beispiel: SMA
SMA_20_dist_close = (SMA_20 - Close) / Close

# Beispiel: RSI
RSI_14_dist_50 = (RSI_14 - 50) / 50  # Distance zu Neutral Level
```

Dies hilft ML-Modellen:
- Relative Positionen zu verstehen
- Trendinformationen besser zu erfassen
- Feature-Scale zu normalisieren

### No Future Bias

Alle Features sind so implementiert, dass sie **keinen Future Bias** haben:
- Ichimoku: Keine Vorverschiebung der Spans
- Alle Rolling Calculations: Nur vergangene Daten
- Safe für Backtesting und Training

### NaN Handling

Nach Feature-Generation werden alle Zeilen mit NaN-Werten automatisch entfernt (Rolling Windows erzeugen am Anfang NaNs).

## 🧪 Beispiel Feature Output

Nach der Generierung erhältst du eine CSV mit z.B. folgenden Spalten:

```
Open, High, Low, Close, Volume,
Range, Range_dist_close,
Close_Open_Ratio, Close_Open_Ratio_dist_close,
SMA_20, SMA_20_dist_close,
EMA_50, EMA_50_dist_close,
RSI_14, RSI_14_dist_50,
MACD_12_26, MACD_Signal_12_26_9, MACD_Hist_12_26_9,
BB_Upper_20, BB_Middle_20, BB_Lower_20, BB_PctB_20,
ATR_14, ATR_14_dist_close,
Ichimoku_Tenkan_9, Ichimoku_Kijun_26,
ADX_14, Plus_DI_14, Minus_DI_14,
...
```

## 🧬 Synthetic Features - Best Practices

### Wann synthetische Features verwenden?

**Verwende synthetische Features wenn:**
- Du mit nicht-linearen ML-Modellen arbeitest (Random Forest, XGBoost, Neural Networks)
- Du Feature-Interaktionen automatisch entdecken möchtest
- Du zeitliche Abhängigkeiten erfassen willst (Lag Features)
- Du mehr Daten brauchst (Data Augmentation)

**Vermeide synthetische Features wenn:**
- Du mit linearen Modellen arbeitest (kann zu Multikollinearität führen)
- Dein Datensatz bereits sehr groß ist (>1000 Features)
- Du schnelle Inferenz benötigst (mehr Features = langsamere Prediction)

### Feature-Explosion vermeiden

🔥 **Achtung**: Synthetische Features können exponentiell wachsen!

**Beispiel**:
- 10 Base Features mit Polynomial Degree 2 → ~55 neue Features
- 10 Base Features mit Interactions → ~200 neue Features
- 10 Base Features mit Lag 5 → 50 neue Features

**Best Practice**:
1. Starte mit 5-10 wichtigen Base Features (z.B. SMA_20, RSI_14, MACD)
2. Teste einzelne synthetische Feature-Typen
3. Nutze Feature Importance nach Training um irrelevante Features zu entfernen
4. Kombiniere max. 2-3 synthetische Typen gleichzeitig

### Empfohlene Kombinationen

**Für Zeitreihen-Prediction:**
```
Base Features: Close, SMA_20, EMA_50, RSI_14, Volume
Synthetic: Lag Features (1,2,3,5) + Difference Features (1,2)
```

**Für Trend-Detection:**
```
Base Features: SMA_20, EMA_50, MACD, ADX_14
Synthetic: Feature Interactions + Ratio to Close
```

**Für Volatility-Trading:**
```
Base Features: ATR_14, BB_Width_20, Volatility_20
Synthetic: Rolling Statistics (3,5,10) + Mathematical Transforms
```

**Für automatisches Feature Discovery (Advanced):**
```
Base Features: Close, Volume, SMA_20, RSI_14, MACD_12_26, ATR_14
Synthetic: gplearn (10-20 components, 20 generations)
→ Lässt genetische Algorithmen optimale Kombinationen finden!
```

## 💡 Tips

1. **Start Small**: Wähle zunächst wenige Features für schnelle Tests
2. **Use Scaling**: Skalierung vor Feature-Berechnung kann sinnvoll sein
3. **Window Tuning**: Passe Window Lengths an deinen Timeframe an
4. **Check Statistics**: Überprüfe die Feature Statistics auf Outliers
5. **Volume Check**: Volume Features nur aktivieren, wenn Volume-Daten vorhanden sind
6. **Synthetic Features**: Starte mit 5-10 Base Features, teste Feature Importance
7. **Lag Features**: Besonders wichtig für LSTM/RNN Modelle
8. **Feature Selection**: Nutze Correlation Matrix um redundante Features zu entfernen
9. **gplearn**: Nutze es als letzten Schritt für automatisches Feature Discovery (ist langsam!)
10. **gplearn Settings**: Start mit 10 components, 20 generations. Bei Bedarf erhöhen.

## 🔧 Customization

Das Tool ist modular aufgebaut. Du kannst:
- Neue Features in der `FeatureEngineering` Klasse hinzufügen
- Window-Defaults ändern
- Scaler-Optionen erweitern
- UI-Layout anpassen

## 📊 Output Metrics

Das Tool zeigt dir:
- Total Features (Anzahl der Spalten)
- Total Rows (nach NaN-Removal)
- Missing Values Percentage
- Feature Summary gruppiert nach Kategorien

## 🎯 Use Cases

- **ML Training**: Erstelle Feature-Sets für Preis-Prediction
- **Feature Engineering Exploration**: Teste verschiedene Feature-Kombinationen
- **Backtesting**: Bereite Daten für Strategy-Backtests vor
- **Data Analysis**: Erweitere deine OHLC-Daten mit technischen Indikatoren

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

## ⚠️ Notes

- Rolling Calculations benötigen Warm-up Period (erste N Zeilen werden NaN)
- Je mehr Features, desto länger die Berechnung
- Sehr lange Windows (z.B. SMA_200) benötigen mindestens 200+ Datenpunkte
- Volume Features erfordern Volume-Spalte in den Daten
- **Synthetische Features**: Können zu 100+ zusätzlichen Features führen - Feature Selection empfohlen!
- **Lag Features**: Reduzieren die Anzahl verfügbarer Datenpunkte (Lag 10 = 10 Zeilen weniger)
- **Polynomial Features**: Degree 3 kann bei vielen Features sehr rechenintensiv werden

## 🤝 Contributing

Feel free to extend this tool with:
- Neue Technical Indicators
- Zusätzliche Scaling Methods
- Custom Feature Combinations
- Export Formats (Parquet, Excel, etc.)

Feel also free to contact me constantinwilharm@gmail.com!

## 📄 License

Open Source - Use freely for your ML projects!

---

**Happy Feature Engineering! 🚀📊🤖**
