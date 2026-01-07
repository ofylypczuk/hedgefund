import ccxt.async_support as ccxt  # Importujemy wersję asynchroniczną
import pandas as pd
import pandas_ta as ta
import time
import logging
import yaml
import asyncio
from datetime import datetime

import yaml
import asyncio
from datetime import datetime
from config_schema import BotConfig
from pydantic import ValidationError

# Wczytywanie konfiguracji z Walidacją Pydantic
def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as file:
            raw_config = yaml.safe_load(file)
            # Walidacja Pydantic
            validated_config = BotConfig(**raw_config)
            return validated_config.model_dump()
    except FileNotFoundError:
        print("BŁĄD KRYTYCZNY: Nie znaleziono pliku config.yaml!")
        exit(1)
    except ValidationError as e:
        print(f"BŁĄD WALIDACJI KONFIGURACJI:\n{e}")
        exit(1)
    except Exception as e:
        print(f"Nieoczekiwany błąd konfiguracji: {e}")
        exit(1)

config = load_config()

# Konfiguracja logowania
logging.basicConfig(
    level=getattr(logging, config['logging']['level'].upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CryptoHedgeFund")

class MarketDataHandler:
    """
    Odpowiada za komunikację z giełdą i pobieranie danych rynkowych (Async).
    """
    def __init__(self, exchange_id, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_id = exchange_id
        self.exchange = None

    async def initialize(self):
        """Asynchroniczna inicjalizacja giełdy"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Pobieranie kluczy z configu
            api_key = config['exchange'].get('api_key', '')
            secret = config['exchange'].get('secret', '')
            
            exchange_params = {
                'enableRateLimit': config['exchange'].get('enable_rate_limit', True),
            }
            
            if api_key and secret and api_key != 'YOUR_API_KEY':
                exchange_params['apiKey'] = api_key
                exchange_params['secret'] = secret
                logger.info("Załadowano klucze API.")
            
            self.exchange = exchange_class(exchange_params)
            logger.info(f"Połączono z giełdą (Async): {self.exchange_id}")
            
            # Załaduj rynki (niezbędne dla create_order)
            await self.exchange.load_markets()
            
        except AttributeError:
            logger.error(f"Nieznana giełda: {self.exchange_id}")
            raise ValueError(f"Exchange {self.exchange_id} not found in ccxt")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def fetch_ohlcv(self, limit=100, symbol=None):
        """
        Pobiera świeczki. Jeśli symbol nie podany, używa domyślnego.
        """
        target_symbol = symbol if symbol else self.symbol
        try:
            if not self.exchange:
                await self.initialize()

            ohlcv = await self.exchange.fetch_ohlcv(target_symbol, self.timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            logger.error(f"Błąd fetch_ohlcv dla {target_symbol}: {e}")
            return pd.DataFrame()

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        """
        Wysyła prawdziwe zlecenie na giełdę.
        """
        if not self.exchange:
            await self.initialize()
            
        try:
            logger.info(f"Wysyłanie zlecenia LIVE: {side} {amount} {symbol}")
            order = await self.exchange.create_order(symbol, type, side, amount, price, params)
            logger.info(f"Zlecenie wykonane: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"BŁĄD ZLECENIA LIVE: {e}")
            return None

class StrategyEngine:
    """
    Silnik strategii (pozostaje synchroniczny, bo operuje na CPU/DataFrame).
    """
    def __init__(self, rsi_period, ema_fast, ema_slow):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def calculate_indicators(self, df):
        if df.empty:
            return df
        df = df.copy()
        df['RSI'] = df.ta.rsi(length=self.rsi_period)
        df['EMA_fast'] = df.ta.ema(length=self.ema_fast)
        df['EMA_slow'] = df.ta.ema(length=self.ema_slow)
        return df

    def generate_signal(self, df):
        if df.empty or len(df) < self.ema_slow:
            logger.warning("Za mało danych.")
            return 'HOLD'

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        is_uptrend = last_row['EMA_fast'] > last_row['EMA_slow']
        rsi_oversold = last_row['RSI'] < 30
        rsi_overbought = last_row['RSI'] > 70
        cross_above = (prev_row['EMA_fast'] <= prev_row['EMA_slow']) and (last_row['EMA_fast'] > last_row['EMA_slow'])

        signal = 'HOLD'
        if is_uptrend and rsi_oversold:
            signal = 'BUY'
        elif not is_uptrend and rsi_overbought:
            signal = 'SELL'
        
        if cross_above:
            logger.info("Golden Cross!")
            signal = 'BUY'

        return signal

class RiskManager:
    """
    Zarządzanie ryzykiem (logika matematyczna, synchroniczna).
    """
    def __init__(self, total_capital, risk_per_trade_percent, leverage, stop_loss_pct_buy, stop_loss_pct_sell):
        self.total_capital = total_capital
        self.risk_per_trade_percent = risk_per_trade_percent
        self.leverage = leverage
        self.stop_loss_pct_buy = stop_loss_pct_buy
        self.stop_loss_pct_sell = stop_loss_pct_sell

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Oblicza wielkość pozycji. Obsługuje Kelly Criterion jeśli włączony w configu.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0

        # Sprawdź czy Kelly jest włączony (pobieramy z globalnego configa lub argumentu, tu: z globalnego dla uproszczenia sygnatury)
        use_kelly = config['risk_management'].get('use_kelly_criterion', False)
        
        if use_kelly:
            win_rate = config['risk_management'].get('kelly_win_rate', 0.5)
            ratio = config['risk_management'].get('kelly_win_loss_ratio', 1.0)
            fraction = config['risk_management'].get('kelly_fraction', 0.5)
            
            # Wzór Kelly'ego: f = (bp - q) / b = p - (1-p)/b
            # gdzie: p = win_rate, b = win_loss_ratio
            kelly_pct = win_rate - ((1 - win_rate) / ratio)
            
            # Zastosowanie ułamka Kelly'ego (np. Half Kelly)
            risk_percent = max(0, kelly_pct * fraction)
            
            # Zabezpieczenie: Nie ryzykuj więcej niż np. 5% portfela nawet jeśli Kelly mówi inaczej
            risk_percent = min(risk_percent, 0.05)
            
            risk_amount = self.total_capital * risk_percent
            logger.info(f"Kelly Criterion: Wyliczone ryzyko {risk_percent*100:.2f}% (Kwota: {risk_amount:.2f}$)")
        else:
            # Standardowe stałe ryzyko
            risk_amount = self.total_capital * self.risk_per_trade_percent

        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0.0

        position_size_units = risk_amount / price_diff
        position_value_usd = position_size_units * entry_price
        max_position_value = self.total_capital * self.leverage
        
        if position_value_usd > max_position_value:
            logger.warning(f"Zredukowano pozycję. Max: {max_position_value:.2f}$")
            position_size_units = max_position_value / entry_price

        return position_size_units

    def get_stop_loss_price(self, entry_price, signal):
        if signal == 'BUY':
            return entry_price * (1 - self.stop_loss_pct_buy)
        elif signal == 'SELL':
            return entry_price * (1 + self.stop_loss_pct_sell)
        return entry_price

    def calculate_trailing_stop(self, current_price, current_stop_loss, position_side):
        """
        Oblicza nowy poziom Trailing Stop.
        """
        if not config['risk_management'].get('use_trailing_stop', False):
            return current_stop_loss

        dist_pct = config['risk_management'].get('trailing_stop_distance_pct', 0.01)
        
        if position_side == 'BUY':
            # Dla BUY: SL przesuwa się w górę
            potential_new_sl = current_price * (1 - dist_pct)
            if potential_new_sl > current_stop_loss:
                return potential_new_sl
        
        elif position_side == 'SELL':
            # Dla SELL: SL przesuwa się w dół
            potential_new_sl = current_price * (1 + dist_pct)
            if potential_new_sl < current_stop_loss:
                return potential_new_sl

        return current_stop_loss

    def check_correlation(self, df_main, df_corr, threshold=0.7):
        """
        Sprawdza korelację między dwoma aktywami.
        Zwraca True jeśli korelacja jest ZBYT WYSOKA (ryzyko), False jeśli jest ok.
        """
        if df_main.empty or df_corr.empty:
            return False
            
        # Dopasowanie długości (bierzemy przecięcie indeksów czasowych lub po prostu ostatnie N)
        min_len = min(len(df_main), len(df_corr))
        series_a = df_main['close'].iloc[-min_len:].reset_index(drop=True)
        series_b = df_corr['close'].iloc[-min_len:].reset_index(drop=True)
        
        correlation = series_a.corr(series_b)
        logger.info(f"Korelacja z aktywem porównawczym: {correlation:.2f}")
        
        if abs(correlation) > threshold:
            logger.warning(f"Zbyt wysoka korelacja ({correlation:.2f} > {threshold}). Unikamy handlu.")
            return True
        return False

from database import DatabaseHandler
from notifier import Notifier

class ExecutionBot:
    """
    ExecutionBot działający w pętli asynchronicznej.
    """
    def __init__(self, data_handler, strategy_engine, risk_manager, database_handler, notifier):
        # Wstrzykiwanie zależności
        self.data_handler = data_handler
        self.strategy = strategy_engine
        self.risk_manager = risk_manager
        self.db = database_handler
        self.notifier = notifier # Nowe wstrzyknięcie
        
        # Pobieranie konfiguracji
        self.symbol = config['exchange']['symbol']
        self.paper_trading = config['execution']['paper_trading']
        self.is_running = False

    async def execute_order(self, signal, price, size, stop_loss):
        """
        Symuluje lub wykonuje zlecenie i zapisuje w DB.
        """
        message = f"🚨 SYGNAŁ: {signal} | {self.symbol} | Cena: {price} | SL: {stop_loss:.2f} | Ilość: {size:.6f}"
        if self.paper_trading:
            message = f"[PAPER] {message}"
        else:
            message = f"[LIVE] {message}"
            
        # Powiadomienie
        await self.notifier.notify(message)

        trade_type = 'PAPER' if self.paper_trading else 'LIVE'

        if self.paper_trading:
            logger.info(f"{message}")
            try:
                await self.db.log_trade(self.symbol, signal, price, size, stop_loss, trade_type)
            except Exception as e:
                logger.error(f"Nie udało się zapisać transakcji do DB: {e}")
        else:
            logger.warning(f"PRÓBA LIVE: {message}")
            
            # LIVE EXECUTION
            side = 'buy' if signal == 'BUY' else 'sell'
            order = await self.data_handler.create_order(self.symbol, 'market', side, size)
            
            if order:
                logger.info(f"Otrzymano potwierdzenie zlecenia: {order['id']}")
                await self.notifier.notify(f"✅ ZLECENIE WYKONANE: {order['id']}")
                # Log do DB
                await self.db.log_trade(self.symbol, signal, order.get('price', price), order['amount'], stop_loss, trade_type)

    async def run_once(self):
        """
        Pojedynczy przebieg pętli decyzyjnej.
        """
        logger.info("--- Async Analysis Cycle ---")
        
        # 1. Pobierz dane główne (Async)
        limit = max(config['strategy']['ema_slow'] + 100, 300) 
        df = await self.data_handler.fetch_ohlcv(limit=limit)
        
        if df.empty:
            logger.warning("Brak danych.")
            return

        # 1b. Pobierz dane do korelacji
        if config['risk_management'].get('check_correlation', False):
            corr_symbol = config['risk_management'].get('correlation_symbol', 'ETH/USDT')
            df_corr = await self.data_handler.fetch_ohlcv(limit=limit, symbol=corr_symbol)
            is_correlated = self.risk_manager.check_correlation(df, df_corr, config['risk_management'].get('correlation_threshold', 0.7))
        else:
            is_correlated = False

        # 2. Oblicz wskaźniki
        df = self.strategy.calculate_indicators(df)
        
        # 3. Sygnał
        signal = self.strategy.generate_signal(df)
        current_price = df.iloc[-1]['close']
        last_rsi = df.iloc[-1]['RSI']
        
        logger.info(f"Cena: {current_price}, RSI: {last_rsi:.2f}, Sygnał: {signal}")
        
        # Filtr korelacji
        if is_correlated and signal != 'HOLD':
            msg = f"⚠️ Ostrzeżenie o korelacji! Sygnał {signal} na {self.symbol} może być ryzykowny."
            logger.info(msg)
            await self.notifier.notify(msg)

        # Zapisz sygnał do DB
        await self.db.log_signal(self.symbol, signal, current_price, last_rsi)
        
        # 4. Egzekucja
        if signal != 'HOLD':
            stop_loss = self.risk_manager.get_stop_loss_price(current_price, signal)
            size = self.risk_manager.calculate_position_size(current_price, stop_loss)
            await self.execute_order(signal, current_price, size, stop_loss)

    async def start(self):
        """
        Uruchamia główną pętlę.
        """
        interval = config['execution']['loop_interval_seconds']
        self.is_running = True
        logger.info(f"Bot uruchomiony (Async). Interwał: {interval}s. LIVE: {not self.paper_trading}")
        
        # Inicjalizacja zasobów
        await self.data_handler.initialize()
        await self.db.initialize()
        await self.notifier.initialize() # Init Notifier
        
        try:
            while self.is_running:
                await self.run_once()
                await asyncio.sleep(interval) 
        except asyncio.CancelledError:
            logger.info("Zatrzymywanie zadania async...")
        except Exception as e:
            logger.error(f"Krytyczny błąd w pętli: {e}")
        finally:
            await self.data_handler.close()
            await self.db.close()
            await self.notifier.close() # Close Notifier

# ==========================================
# Uruchomienie Asynchroniczne
# ==========================================
async def main():
    print("Inicjalizacja Mini Hedge Fund Bot (Async)...")
    
    # 1. Inicjalizacja Modułów
    data_handler = MarketDataHandler(
        exchange_id=config['exchange']['id'],
        symbol=config['exchange']['symbol'],
        timeframe=config['exchange']['timeframe']
    )
    
    strategy_engine = StrategyEngine(
        rsi_period=config['strategy']['rsi_period'],
        ema_fast=config['strategy']['ema_fast'],
        ema_slow=config['strategy']['ema_slow']
    )
    
    risk_manager = RiskManager(
        total_capital=config['risk_management']['total_capital'],
        risk_per_trade_percent=config['risk_management']['risk_per_trade_percent'],
        leverage=config['risk_management']['leverage'],
        stop_loss_pct_buy=config['risk_management']['stop_loss_percent_buy'],
        stop_loss_pct_sell=config['risk_management']['stop_loss_percent_sell']
    )
    
    db_path = config.get('database', {}).get('db_path', 'hedge_fund.db')
    database_handler = DatabaseHandler(db_path=db_path)
    
    # Inicjalizacja Notifiera
    notif_cfg = config.get('notifications', {})
    notifier = Notifier(
        telegram_token=notif_cfg.get('telegram_token'),
        telegram_chat_id=notif_cfg.get('telegram_chat_id'),
        discord_webhook_url=notif_cfg.get('discord_webhook_url')
    )

    # 2. Wstrzyknięcie wszystkich zależności do ExecutionBot
    bot = ExecutionBot(data_handler, strategy_engine, risk_manager, database_handler, notifier)
    
    # Test
    print("Testowe uruchomienie pojedynczego cyklu (Async)...")
    await bot.db.initialize()
    await bot.notifier.initialize()
    
    await bot.run_once()
    
    await bot.data_handler.close()
    await bot.db.close()
    await bot.notifier.close()
    
    print("\nGotowe. Aby uruchomić w pętli, w produkcji użyj: asyncio.run(bot.start())")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Zatrzymano ręcznie.")
