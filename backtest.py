import pandas as pd
import asyncio
import ccxt.async_support as ccxt
import logging
from main import StrategyEngine, load_config
import matplotlib.pyplot as plt

# Konfiguracja logowania dla backtestu
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Backtest")

class Backtester:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.symbol = self.config['exchange']['symbol']
        self.timeframe = self.config['exchange']['timeframe']
        
        self.strategy = StrategyEngine(
            rsi_period=self.config['strategy']['rsi_period'],
            ema_fast=self.config['strategy']['ema_fast'],
            ema_slow=self.config['strategy']['ema_slow']
        )
        
        self.initial_capital = self.config['risk_management']['total_capital']
        self.balance = self.initial_capital
        self.position = 0 # Ilość krypto
        self.trades = []
        self.equity_curve = []

    async def fetch_historical_data(self, days=30):
        """Pobiera dane historyczne z giełdy dla backtestu (ok. 1000 świeczek na miesiąc dla 1h)."""
        logger.info(f"Pobieranie danych historycznych dla {self.symbol} ({days} dni)...")
        
        exchange_class = getattr(ccxt, self.config['exchange']['id'])
        exchange = exchange_class()
        
        # Obliczanie limitu świeczek
        # 1 dzień * 24h = 24 świeczki (dla timeframe 1h)
        needed_candles = days * 24 
        limit = min(needed_candles, 1000) # CCXT zwykle ma limit per request
        
        # Uwaga: Dla prawdziwego backtestu na długim okresie trzeba by użyć pętli i since/pagination
        # Tutaj dla uproszczenia pobieramy ostatnie 1000 świeczek.
        ohlcv = await exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000)
        await exchange.close()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
             df[col] = df[col].astype(float)
             
        logger.info(f"Pobrano {len(df)} świeczek.")
        return df

    def run(self, df):
        logger.info("Rozpoczynanie symulacji...")
        
        # 1. Oblicz wskaźniki dla całego DF (szybko)
        df = self.strategy.calculate_indicators(df)
        
        # 2. Iteruj po świeczkach (pomijamy początek gdzie nie ma EMA)
        start_index = self.config['strategy']['ema_slow']
        
        for i in range(start_index, len(df)):
            # Symulacja widoczności danych "do teraz"
            # StrategyEngine.generate_signal używa .iloc[-1], więc musimy podać wycinek
            # Dla wydajności w Pythonie:
            # Zamiast kroić DF w pętli (bardzo wolne), zaimplementujemy logikę inline lub lekko zmienimy generate_signal
            # Ale trzymajmy się architektury:
            
            # Wersja "czysta architektonicznie" (wolniejsza):
            # current_slice = df.iloc[:i+1]
            # signal = self.strategy.generate_signal(current_slice)
            
            # Wersja "zoptymalizowana pod backtest":
            # Wyciągamy wartości bezpośrednio z wiersza
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            is_uptrend = row['EMA_fast'] > row['EMA_slow']
            rsi_oversold = row['RSI'] < 30
            rsi_overbought = row['RSI'] > 70
            
            # Cross above
            cross_above = (prev_row['EMA_fast'] <= prev_row['EMA_slow']) and (row['EMA_fast'] > row['EMA_slow'])
            
            signal = 'HOLD'
            if is_uptrend and rsi_oversold:
                signal = 'BUY'
            elif not is_uptrend and rsi_overbought:
                signal = 'SELL'
            if cross_above:
                signal = 'BUY'
            
            current_price = row['close']
            
            # Egzekucja (Symulacja)
            self.simulate_trade(signal, current_price, row['timestamp'])
            
            # Aktualizacja equity (Mark-to-Market)
            current_equity = self.balance + (self.position * current_price)
            self.equity_curve.append({'timestamp': row['timestamp'], 'equity': current_equity})

        self.print_results()

    def simulate_trade(self, signal, price, timestamp):
        # Prosta logika: Zawsze wchodzimy "All in" (dla uproszczenia backtestu) lub stała kwota
        # W main.py mamy RiskManager, tu uprościmy:
        
        # Prowizja (np. 0.1%)
        commission = 0.001
        
        if signal == 'BUY' and self.balance > 10: # Masz gotówkę, kupujesz
            amount_to_spend = self.balance
            amount_crypto = (amount_to_spend / price) * (1 - commission)
            
            self.position = amount_crypto
            self.balance = 0
            
            self.trades.append({'type': 'BUY', 'price': price, 'time': timestamp, 'amount': amount_crypto})
            # logger.info(f"BUY at {price:.2f}")
            
        elif signal == 'SELL' and self.position > 0: # Masz krypto, sprzedajesz
            amount_fiat = (self.position * price) * (1 - commission)
            
            self.balance = amount_fiat
            self.position = 0
            
            self.trades.append({'type': 'SELL', 'price': price, 'time': timestamp, 'amount': amount_fiat})
            # logger.info(f"SELL at {price:.2f}")

    def print_results(self):
        if not self.equity_curve:
            print("Brak transakcji lub za mało danych.")
            return
            
        final_equity = self.equity_curve[-1]['equity']
        returns = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        print("\n=== WYNIKI BACKTESTU ===")
        print(f"Kapitał początkowy: {self.initial_capital:.2f}")
        print(f"Kapitał końcowy: {final_equity:.2f}")
        print(f"Zwrot całkowity: {returns:.2f}%")
        print(f"Liczba transakcji: {len(self.trades)}")
        
        # Wykres
        try:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.set_index('timestamp', inplace=True)
            df_equity.plot(title=f'Equity Curve ({self.symbol})')
            plt.show()
        except Exception:
            print("Nie można wygenerować wykresu (brak środowiska graficznego?)")

async def main():
    backtester = Backtester()
    df = await backtester.fetch_historical_data(days=60)
    backtester.run(df)

if __name__ == "__main__":
    asyncio.run(main())
