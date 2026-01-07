import aiosqlite
import logging
import datetime

logger = logging.getLogger("CryptoHedgeFund")

class DatabaseHandler:
    """
    Asynchroniczna obsługa bazy danych SQLite dla zachowania trwałości danych.
    """
    def __init__(self, db_path='hedge_fund.db'):
        self.db_path = db_path
        self.connection = None

    async def initialize(self):
        """Inicjalizacja połączenia i tworzenie tabel."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self.create_tables()
            logger.info(f"Połączono z bazą danych: {self.db_path}")
        except Exception as e:
            logger.error(f"Błąd inicjalizacji bazy danych: {e}")
            raise

    async def create_tables(self):
        """Tworzy niezbędne tabele, jeśli nie istnieją."""
        create_trades_table = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL, -- BUY/SELL
            price REAL NOT NULL,
            amount REAL NOT NULL,
            stop_loss REAL,
            type TEXT NOT NULL -- PAPER/LIVE
        );
        """
        
        create_signals_table = """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL, -- BUY/SELL/HOLD
            price REAL,
            rsi REAL
        );
        """
        
        await self.connection.execute(create_trades_table)
        await self.connection.execute(create_signals_table)
        await self.connection.commit()

    async def log_trade(self, symbol, side, price, amount, stop_loss, trade_type='PAPER'):
        """Zapisuje transakcję w bazie danych."""
        try:
            timestamp = datetime.datetime.now().isoformat()
            query = """
            INSERT INTO trades (timestamp, symbol, side, price, amount, stop_loss, type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            await self.connection.execute(query, (timestamp, symbol, side, price, amount, stop_loss, trade_type))
            await self.connection.commit()
            logger.info(f"Zapisano transakcję w DB: {side} {symbol}")
        except Exception as e:
            logger.error(f"Błąd zapisu transakcji do DB: {e}")

    async def log_signal(self, symbol, action, price, rsi):
        """Zapisuje sygnał strategii."""
        try:
            timestamp = datetime.datetime.now().isoformat()
            query = """
            INSERT INTO signals (timestamp, symbol, action, price, rsi)
            VALUES (?, ?, ?, ?, ?)
            """
            await self.connection.execute(query, (timestamp, symbol, action, price, rsi))
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Błąd zapisu sygnału do DB: {e}")

    async def close(self):
        """Zamyka połączenie z bazą."""
        if self.connection:
            await self.connection.close()
            logger.info("Zamknięto połączenie z bazą danych.")
