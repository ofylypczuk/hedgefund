from pydantic import BaseModel, Field, ValidationError, validator
from typing import Optional

class ExchangeConfig(BaseModel):
    id: str
    symbol: str
    timeframe: str
    enable_rate_limit: bool = True
    api_key: Optional[str] = None
    secret: Optional[str] = None

class StrategyConfig(BaseModel):
    rsi_period: int = Field(..., gt=0, description="Okres RSI musi być dodatni")
    ema_fast: int = Field(..., gt=0)
    ema_slow: int = Field(..., gt=0)

    @validator('ema_slow')
    def validate_ema(cls, v, values):
        if 'ema_fast' in values and v <= values['ema_fast']:
            raise ValueError('EMA Slow musi być większa niż EMA Fast')
        return v

class RiskManagementConfig(BaseModel):
    total_capital: float = Field(..., gt=0)
    risk_per_trade_percent: float = Field(..., gt=0, le=1.0)
    leverage: int = Field(..., ge=1, le=125)
    stop_loss_percent_buy: float = Field(..., gt=0, le=1.0)
    stop_loss_percent_sell: float = Field(..., gt=0, le=1.0)
    
    # Advanced
    use_kelly_criterion: bool = False
    kelly_win_rate: float = 0.5
    kelly_win_loss_ratio: float = 1.0
    kelly_fraction: float = 0.5
    
    use_trailing_stop: bool = False
    trailing_stop_activation_pct: float = 0.01
    trailing_stop_distance_pct: float = 0.01
    
    # Correlation
    check_correlation: bool = False
    correlation_threshold: float = 0.7
    correlation_symbol: str = 'ETH/USDT'

class ExecutionConfig(BaseModel):
    paper_trading: bool = True
    loop_interval_seconds: int = Field(..., gt=0)

class DatabaseConfig(BaseModel):
    db_path: str = 'hedge_fund.db'

class LoggingConfig(BaseModel):
    level: str = 'INFO'
    
class NotificationConfig(BaseModel):
    enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None

class BotConfig(BaseModel):
    exchange: ExchangeConfig
    strategy: StrategyConfig
    risk_management: RiskManagementConfig
    execution: ExecutionConfig
    database: DatabaseConfig
    logging: LoggingConfig
    notifications: NotificationConfig = NotificationConfig() # Opcjonalne
