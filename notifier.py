import aiohttp
import logging
import asyncio

logger = logging.getLogger("Notifier")

class Notifier:
    """
    Klasa do wysyłania asynchronicznych powiadomień na Telegram i Discord.
    """
    def __init__(self, telegram_token=None, telegram_chat_id=None, discord_webhook_url=None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook_url = discord_webhook_url
        self.session = None

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def send_telegram(self, message):
        if not self.telegram_token or not self.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Błąd Telegram: {await response.text()}")
        except Exception as e:
            logger.error(f"Nie udało się wysłać na Telegram: {e}")

    async def send_discord(self, message):
        if not self.discord_webhook_url:
            return

        payload = {'content': message}
        try:
            async with self.session.post(self.discord_webhook_url, json=payload) as response:
                if response.status not in (200, 204):
                    logger.error(f"Błąd Discord: {await response.text()}")
        except Exception as e:
            logger.error(f"Nie udało się wysłać na Discord: {e}")

    async def notify(self, message):
        """Wysyła powiadomienie do wszystkich skonfigurowanych kanałów."""
        if not self.session:
            await self.initialize() # Auto-init jeśli zapomniano
            
        tasks = []
        if self.telegram_token and self.telegram_chat_id:
            tasks.append(self.send_telegram(message))
        
        if self.discord_webhook_url:
            tasks.append(self.send_discord(message))

        if tasks:
            await asyncio.gather(*tasks)
