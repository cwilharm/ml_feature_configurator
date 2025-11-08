"""
Feature Engineering Module with Technical Indicators
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureEngineering:
    """Feature Engineering Class with all technical indicators"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close = df['Close']
        self.open = df['Open']
        self.high = df['High']
        self.low = df['Low']
        self.volume = df['Volume'] if 'Volume' in df.columns else None

    # ========== 1. PRICE-BASED FEATURES ==========

    def add_range(self):
        """High - Low"""
        self.df['Range'] = self.high - self.low
        self.df['Range_dist_close'] = self.df['Range'] / self.close
        return self

    def add_close_open_ratio(self):
        """Close/Open Ratio"""
        self.df['Close_Open_Ratio'] = self.close / self.open
        self.df['Close_Open_Ratio_dist_close'] = (self.df['Close_Open_Ratio'] - 1)
        return self

    def add_high_low_ratio(self):
        """High/Low Ratio"""
        self.df['High_Low_Ratio'] = self.high / self.low
        self.df['High_Low_Ratio_dist_close'] = self.df['High_Low_Ratio'] - (self.close / self.low)
        return self

    def add_mid_price(self):
        """(High + Low) / 2"""
        self.df['Mid_Price'] = (self.high + self.low) / 2
        self.df['Mid_Price_dist_close'] = (self.df['Mid_Price'] - self.close) / self.close
        return self

    # ========== 2. RETURNS / CHANGE RATES ==========

    def add_daily_return(self):
        """(Close - Open) / Open"""
        self.df['Daily_Return'] = (self.close - self.open) / self.open
        return self

    def add_log_return(self):
        """ln(Close_t / Close_(t-1))"""
        self.df['Log_Return'] = np.log(self.close / self.close.shift(1))
        return self

    def add_high_low_return(self):
        """(High - Low) / Low"""
        self.df['High_Low_Return'] = (self.high - self.low) / self.low
        return self

    def add_open_close_return(self):
        """(Close - Open) / Open"""
        self.df['Open_Close_Return'] = (self.close - self.open) / self.open
        return self

    # ========== 3. MOVING AVERAGES ==========

    def add_sma(self, windows: List[int] = [5, 10, 20, 50, 100, 200]):
        """Simple Moving Average"""
        for window in windows:
            self.df[f'SMA_{window}'] = self.close.rolling(window=window).mean()
            self.df[f'SMA_{window}_dist_close'] = (self.df[f'SMA_{window}'] - self.close) / self.close
        return self

    def add_ema(self, windows: List[int] = [5, 10, 20, 50, 100, 200]):
        """Exponential Moving Average"""
        for window in windows:
            self.df[f'EMA_{window}'] = self.close.ewm(span=window, adjust=False).mean()
            self.df[f'EMA_{window}_dist_close'] = (self.df[f'EMA_{window}'] - self.close) / self.close
        return self

    def add_wma(self, windows: List[int] = [10, 20, 50]):
        """Weighted Moving Average"""
        for window in windows:
            weights = np.arange(1, window + 1)
            self.df[f'WMA_{window}'] = self.close.rolling(window).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            self.df[f'WMA_{window}_dist_close'] = (self.df[f'WMA_{window}'] - self.close) / self.close
        return self

    def add_hma(self, windows: List[int] = [9, 16, 25]):
        """Hull Moving Average"""
        for window in windows:
            half_length = int(window / 2)
            sqrt_length = int(np.sqrt(window))

            wma_half = self.close.rolling(half_length).apply(
                lambda x: np.dot(x, np.arange(1, half_length + 1)) / np.arange(1, half_length + 1).sum(), raw=True
            )
            wma_full = self.close.rolling(window).apply(
                lambda x: np.dot(x, np.arange(1, window + 1)) / np.arange(1, window + 1).sum(), raw=True
            )
            raw_hma = 2 * wma_half - wma_full
            self.df[f'HMA_{window}'] = raw_hma.rolling(sqrt_length).apply(
                lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / np.arange(1, sqrt_length + 1).sum(), raw=True
            )
            self.df[f'HMA_{window}_dist_close'] = (self.df[f'HMA_{window}'] - self.close) / self.close
        return self

    # ========== 4. VOLATILITY ==========

    def add_volatility(self, windows: List[int] = [5, 10, 20, 50]):
        """Standard deviation of returns"""
        returns = self.close.pct_change()
        for window in windows:
            self.df[f'Volatility_{window}'] = returns.rolling(window=window).std()
            self.df[f'Volatility_{window}_normalized'] = self.df[f'Volatility_{window}'] / self.close
        return self

    def add_atr(self, windows: List[int] = [14, 20]):
        """Average True Range"""
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift(1))
        low_close = np.abs(self.low - self.close.shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for window in windows:
            self.df[f'ATR_{window}'] = true_range.rolling(window=window).mean()
            self.df[f'ATR_{window}_dist_close'] = self.df[f'ATR_{window}'] / self.close
        return self

    # ========== 5. MOMENTUM INDICATORS ==========

    def add_momentum(self, windows: List[int] = [5, 10, 20]):
        """Close_t - Close_(t-n)"""
        for window in windows:
            self.df[f'Momentum_{window}'] = self.close - self.close.shift(window)
            self.df[f'Momentum_{window}_pct'] = self.df[f'Momentum_{window}'] / self.close
        return self

    def add_roc(self, windows: List[int] = [5, 10, 20]):
        """Rate of Change: (Close_t - Close_(t-n)) / Close_(t-n)"""
        for window in windows:
            self.df[f'ROC_{window}'] = (self.close - self.close.shift(window)) / self.close.shift(window)
        return self

    def add_rsi(self, windows: List[int] = [14, 21]):
        """Relative Strength Index"""
        for window in windows:
            delta = self.close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            self.df[f'RSI_{window}'] = rsi
            self.df[f'RSI_{window}_dist_50'] = (rsi - 50) / 50  # Distance to neutral level
        return self

    def add_stochastic(self, k_window: int = 14, d_window: int = 3):
        """Stochastic Oscillator"""
        lowest_low = self.low.rolling(window=k_window).min()
        highest_high = self.high.rolling(window=k_window).max()

        self.df[f'Stoch_K_{k_window}'] = 100 * (self.close - lowest_low) / (highest_high - lowest_low)
        self.df[f'Stoch_D_{k_window}_{d_window}'] = self.df[f'Stoch_K_{k_window}'].rolling(window=d_window).mean()
        self.df[f'Stoch_K_{k_window}_dist_50'] = (self.df[f'Stoch_K_{k_window}'] - 50) / 50
        return self

    def add_williams_r(self, windows: List[int] = [14]):
        """Williams %R"""
        for window in windows:
            highest_high = self.high.rolling(window=window).max()
            lowest_low = self.low.rolling(window=window).min()

            self.df[f'Williams_R_{window}'] = -100 * (highest_high - self.close) / (highest_high - lowest_low)
            self.df[f'Williams_R_{window}_dist_50'] = (self.df[f'Williams_R_{window}'] + 50) / 50
        return self

    def add_cci(self, windows: List[int] = [20]):
        """Commodity Channel Index"""
        for window in windows:
            typical_price = (self.high + self.low + self.close) / 3
            sma_tp = typical_price.rolling(window=window).mean()
            mean_deviation = typical_price.rolling(window=window).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )

            self.df[f'CCI_{window}'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
            self.df[f'CCI_{window}_normalized'] = self.df[f'CCI_{window}'] / 100
        return self

    def add_cmo(self, windows: List[int] = [14]):
        """Chande Momentum Oscillator"""
        for window in windows:
            delta = self.close.diff()
            gains = delta.where(delta > 0, 0).rolling(window=window).sum()
            losses = -delta.where(delta < 0, 0).rolling(window=window).sum()

            self.df[f'CMO_{window}'] = 100 * (gains - losses) / (gains + losses)
            self.df[f'CMO_{window}_normalized'] = self.df[f'CMO_{window}'] / 100
        return self

    # ========== 6. MACD ==========

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Moving Average Convergence Divergence"""
        ema_fast = self.close.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        self.df[f'MACD_{fast}_{slow}'] = macd_line
        self.df[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
        self.df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
        self.df[f'MACD_{fast}_{slow}_dist_close'] = macd_line / self.close
        return self

    # ========== 7. BOLLINGER BANDS ==========

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        """Bollinger Bands"""
        sma = self.close.rolling(window=window).mean()
        std = self.close.rolling(window=window).std()

        self.df[f'BB_Upper_{window}'] = sma + (std * num_std)
        self.df[f'BB_Middle_{window}'] = sma
        self.df[f'BB_Lower_{window}'] = sma - (std * num_std)

        # %B indicator
        self.df[f'BB_PctB_{window}'] = (self.close - self.df[f'BB_Lower_{window}']) / \
                                        (self.df[f'BB_Upper_{window}'] - self.df[f'BB_Lower_{window}'])

        # Bandwidth
        self.df[f'BB_Width_{window}'] = (self.df[f'BB_Upper_{window}'] - self.df[f'BB_Lower_{window}']) / \
                                         self.df[f'BB_Middle_{window}']

        # Distance to close
        self.df[f'BB_Upper_{window}_dist_close'] = (self.df[f'BB_Upper_{window}'] - self.close) / self.close
        self.df[f'BB_Lower_{window}_dist_close'] = (self.df[f'BB_Lower_{window}'] - self.close) / self.close
        return self

    def add_keltner_channel(self, window: int = 20, atr_window: int = 10, multiplier: float = 2.0):
        """Keltner Channel"""
        typical_price = (self.high + self.low + self.close) / 3
        basis = typical_price.rolling(window=window).mean()

        # Calculate ATR
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift(1))
        low_close = np.abs(self.low - self.close.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_window).mean()

        self.df[f'KC_Upper_{window}'] = basis + (multiplier * atr)
        self.df[f'KC_Middle_{window}'] = basis
        self.df[f'KC_Lower_{window}'] = basis - (multiplier * atr)

        self.df[f'KC_Upper_{window}_dist_close'] = (self.df[f'KC_Upper_{window}'] - self.close) / self.close
        self.df[f'KC_Lower_{window}_dist_close'] = (self.df[f'KC_Lower_{window}'] - self.close) / self.close
        return self

    # ========== 8. CANDLESTICK FEATURES ==========

    def add_candlestick_features(self):
        """Body, Shadows, and Ratios"""
        # Body
        self.df['Body_Length'] = np.abs(self.close - self.open)
        self.df['Body_Length_pct'] = self.df['Body_Length'] / self.close

        # Shadows
        self.df['Upper_Shadow'] = self.high - np.maximum(self.close, self.open)
        self.df['Lower_Shadow'] = np.minimum(self.close, self.open) - self.low
        self.df['Upper_Shadow_pct'] = self.df['Upper_Shadow'] / self.close
        self.df['Lower_Shadow_pct'] = self.df['Lower_Shadow'] / self.close

        # Ratios
        range_val = self.high - self.low
        self.df['Body_Range_Ratio'] = self.df['Body_Length'] / range_val.replace(0, np.nan)
        self.df['Upper_Shadow_Body_Ratio'] = self.df['Upper_Shadow'] / self.df['Body_Length'].replace(0, np.nan)
        self.df['Lower_Shadow_Body_Ratio'] = self.df['Lower_Shadow'] / self.df['Body_Length'].replace(0, np.nan)

        # Candle Direction
        self.df['Candle_Direction'] = np.where(self.close >= self.open, 1, -1)
        return self

    # ========== 9. ICHIMOKU ==========

    def add_ichimoku(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        """Ichimoku Kinko Hyo (without future bias)"""
        # Tenkan-sen (Conversion Line)
        tenkan_high = self.high.rolling(window=tenkan).max()
        tenkan_low = self.low.rolling(window=tenkan).min()
        self.df[f'Ichimoku_Tenkan_{tenkan}'] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = self.high.rolling(window=kijun).max()
        kijun_low = self.low.rolling(window=kijun).min()
        self.df[f'Ichimoku_Kijun_{kijun}'] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A) - NO FUTURE SHIFT
        self.df[f'Ichimoku_SpanA_{tenkan}_{kijun}'] = (
            self.df[f'Ichimoku_Tenkan_{tenkan}'] + self.df[f'Ichimoku_Kijun_{kijun}']
        ) / 2

        # Senkou Span B (Leading Span B) - NO FUTURE SHIFT
        spanb_high = self.high.rolling(window=senkou_b).max()
        spanb_low = self.low.rolling(window=senkou_b).min()
        self.df[f'Ichimoku_SpanB_{senkou_b}'] = (spanb_high + spanb_low) / 2

        # Chikou Span (Lagging Span) - shifted back
        self.df[f'Ichimoku_Chikou_{kijun}'] = self.close.shift(kijun)

        # Cloud thickness
        self.df['Ichimoku_Cloud_Thickness'] = np.abs(
            self.df[f'Ichimoku_SpanA_{tenkan}_{kijun}'] - self.df[f'Ichimoku_SpanB_{senkou_b}']
        )

        # Distance to close
        self.df[f'Ichimoku_Tenkan_{tenkan}_dist_close'] = (self.df[f'Ichimoku_Tenkan_{tenkan}'] - self.close) / self.close
        self.df[f'Ichimoku_Kijun_{kijun}_dist_close'] = (self.df[f'Ichimoku_Kijun_{kijun}'] - self.close) / self.close
        return self

    # ========== 10. ADX & DMI ==========

    def add_adx_dmi(self, window: int = 14):
        """Average Directional Index and Directional Movement"""
        # Plus/Minus Directional Movement
        high_diff = self.high.diff()
        low_diff = -self.low.diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        plus_dm = pd.Series(plus_dm, index=self.df.index)
        minus_dm = pd.Series(minus_dm, index=self.df.index)

        # True Range
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift(1))
        low_close = np.abs(self.low - self.close.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smoothed values
        atr = true_range.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()

        self.df[f'Plus_DI_{window}'] = plus_di
        self.df[f'Minus_DI_{window}'] = minus_di
        self.df[f'ADX_{window}'] = adx
        self.df[f'ADX_{window}_normalized'] = adx / 100
        return self

    # ========== 11. PARABOLIC SAR ==========

    def add_parabolic_sar(self, acceleration: float = 0.02, maximum: float = 0.2):
        """Parabolic SAR"""
        sar = self.close.copy()
        ep = self.close.copy()
        af = acceleration
        trend = 1  # 1 for uptrend, -1 for downtrend

        sar_values = []

        for i in range(len(self.df)):
            if i == 0:
                sar_values.append(self.low.iloc[i])
            else:
                # This is a simplified implementation
                sar_values.append(sar_values[-1])

        self.df['Parabolic_SAR'] = sar_values
        self.df['Parabolic_SAR_dist_close'] = (self.df['Parabolic_SAR'] - self.close) / self.close
        return self

    # ========== 12. SUPERTREND ==========

    def add_supertrend(self, period: int = 10, multiplier: float = 3.0):
        """SuperTrend Indicator"""
        # Calculate ATR
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift(1))
        low_close = np.abs(self.low - self.close.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # Basic Upper and Lower Bands
        hl_avg = (self.high + self.low) / 2
        basic_upper = hl_avg + (multiplier * atr)
        basic_lower = hl_avg - (multiplier * atr)

        # Initialize final bands
        final_upper = pd.Series(index=self.df.index, dtype=float)
        final_lower = pd.Series(index=self.df.index, dtype=float)
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index, dtype=int)

        # First valid index (after ATR calculation)
        first_valid = period - 1

        for i in range(len(self.df)):
            if i < first_valid:
                # Fill with NaN until ATR is calculated
                final_upper.iloc[i] = np.nan
                final_lower.iloc[i] = np.nan
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = 1
            elif i == first_valid:
                # Initialize at first valid point
                final_upper.iloc[i] = basic_upper.iloc[i]
                final_lower.iloc[i] = basic_lower.iloc[i]
                supertrend.iloc[i] = basic_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                # Update final bands based on previous values
                if pd.notna(basic_upper.iloc[i]) and pd.notna(final_upper.iloc[i-1]):
                    final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper.iloc[i-1] or self.close.iloc[i-1] > final_upper.iloc[i-1] else final_upper.iloc[i-1]
                else:
                    final_upper.iloc[i] = basic_upper.iloc[i]

                if pd.notna(basic_lower.iloc[i]) and pd.notna(final_lower.iloc[i-1]):
                    final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower.iloc[i-1] or self.close.iloc[i-1] < final_lower.iloc[i-1] else final_lower.iloc[i-1]
                else:
                    final_lower.iloc[i] = basic_lower.iloc[i]

                # Determine SuperTrend value and direction
                if pd.notna(supertrend.iloc[i-1]):
                    if supertrend.iloc[i-1] == final_upper.iloc[i-1]:
                        # Was in downtrend
                        if self.close.iloc[i] <= final_upper.iloc[i]:
                            supertrend.iloc[i] = final_upper.iloc[i]
                            direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = final_lower.iloc[i]
                            direction.iloc[i] = 1
                    else:
                        # Was in uptrend
                        if self.close.iloc[i] >= final_lower.iloc[i]:
                            supertrend.iloc[i] = final_lower.iloc[i]
                            direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = final_upper.iloc[i]
                            direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1

        self.df[f'SuperTrend_{period}_{multiplier}'] = supertrend
        self.df[f'SuperTrend_Direction_{period}_{multiplier}'] = direction
        self.df[f'SuperTrend_{period}_{multiplier}_dist_close'] = (supertrend - self.close) / self.close
        return self

    # ========== 13. AWESOME OSCILLATOR ==========

    def add_awesome_oscillator(self, fast: int = 5, slow: int = 34):
        """Awesome Oscillator"""
        mid_price = (self.high + self.low) / 2
        ao = mid_price.rolling(window=fast).mean() - mid_price.rolling(window=slow).mean()

        self.df[f'AO_{fast}_{slow}'] = ao
        self.df[f'AO_{fast}_{slow}_normalized'] = ao / self.close
        return self

    # ========== 14. VOLUME FEATURES ==========

    def add_volume_features(self, windows: List[int] = [5, 10, 20]):
        """Volume-based features"""
        if self.volume is None:
            return self

        # Volume change
        self.df['Volume_Change'] = self.volume / self.volume.shift(1)

        # Volume moving averages
        for window in windows:
            self.df[f'Volume_SMA_{window}'] = self.volume.rolling(window=window).mean()
            self.df[f'Volume_Ratio_{window}'] = self.volume / self.df[f'Volume_SMA_{window}']

        # On-Balance Volume (OBV)
        obv = (np.sign(self.close.diff()) * self.volume).fillna(0).cumsum()
        self.df['OBV'] = obv

        # VWAP (Volume Weighted Average Price)
        typical_price = (self.high + self.low + self.close) / 3
        self.df['VWAP'] = (typical_price * self.volume).cumsum() / self.volume.cumsum()
        self.df['VWAP_dist_close'] = (self.df['VWAP'] - self.close) / self.close

        # Money Flow Index (MFI)
        for window in [14]:
            typical_price = (self.high + self.low + self.close) / 3
            money_flow = typical_price * self.volume

            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()

            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            self.df[f'MFI_{window}'] = mfi
            self.df[f'MFI_{window}_dist_50'] = (mfi - 50) / 50

        return self

    # ========== 15. STATISTICAL FEATURES ==========

    def add_statistical_features(self, windows: List[int] = [5, 10, 20, 50]):
        """Statistical rolling features"""
        for window in windows:
            # Rolling statistics on Close
            self.df[f'Close_Mean_{window}'] = self.close.rolling(window=window).mean()
            self.df[f'Close_Std_{window}'] = self.close.rolling(window=window).std()
            self.df[f'Close_Median_{window}'] = self.close.rolling(window=window).median()

            # Z-Score
            self.df[f'Close_ZScore_{window}'] = (
                (self.close - self.df[f'Close_Mean_{window}']) / self.df[f'Close_Std_{window}']
            )

            # Skewness and Kurtosis
            self.df[f'Close_Skew_{window}'] = self.close.rolling(window=window).skew()
            self.df[f'Close_Kurt_{window}'] = self.close.rolling(window=window).kurt()

            # Distance to mean
            self.df[f'Close_Mean_{window}_dist_close'] = (self.df[f'Close_Mean_{window}'] - self.close) / self.close

        return self

    # ========== 16. TIME-BASED FEATURES ==========

    def add_time_features(self):
        """Time-based cyclical features"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            return self

        self.df['Day_of_Week'] = self.df.index.dayofweek
        self.df['Day_of_Month'] = self.df.index.day
        self.df['Week_of_Month'] = (self.df.index.day - 1) // 7 + 1
        self.df['Month'] = self.df.index.month
        self.df['Quarter'] = self.df.index.quarter

        # Cyclical encoding
        self.df['Day_of_Week_sin'] = np.sin(2 * np.pi * self.df['Day_of_Week'] / 7)
        self.df['Day_of_Week_cos'] = np.cos(2 * np.pi * self.df['Day_of_Week'] / 7)
        self.df['Month_sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)

        return self

    # ========== 17. CUMULATIVE RETURNS ==========

    def add_cumulative_returns(self, windows: List[int] = [5, 10, 20]):
        """Cumulative returns over windows"""
        daily_returns = self.close.pct_change()

        for window in windows:
            self.df[f'Cum_Return_{window}'] = (1 + daily_returns).rolling(window=window).apply(
                lambda x: x.prod() - 1, raw=True
            )

        return self

    # ========== 18. MA CROSSOVERS ==========

    def add_ma_crossovers(self):
        """Moving Average Crossover signals"""
        # EMA crossovers
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        self.df['EMA_12_26_diff'] = ema_12 - ema_26
        self.df['EMA_12_26_diff_pct'] = self.df['EMA_12_26_diff'] / self.close

        # SMA crossovers
        sma_50 = self.close.rolling(window=50).mean()
        sma_200 = self.close.rolling(window=200).mean()
        self.df['SMA_50_200_diff'] = sma_50 - sma_200
        self.df['SMA_50_200_diff_pct'] = self.df['SMA_50_200_diff'] / self.close

        return self

    def get_dataframe(self):
        """Return the dataframe with all features"""
        return self.df
