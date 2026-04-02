"""
Telegram Bot API client implementation with retry logic.

This module provides an async HTTP client for Telegram Bot API with:
- Retry logic for transient errors
- Error classification for retryable vs non-retryable errors
- Rate limit handling with retry_after support
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from core.observability import get_logger, log_event
from telegram_bot.polling import TelegramClient, classify_telegram_error
from telegram_bot.observability import (
    METRICS,
    ClassifiedError,
    ErrorClass,
    log_telegram_event,
)


LOGGER = get_logger(__name__)


# ============================================================================
# Retry Configuration
# ============================================================================

@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 10.0
    multiplier: float = 2.0
    retryable_errors: tuple[str, ...] = ("timeout", "connection", "network")


# ============================================================================
# Telegram API Error
# ============================================================================

class TelegramAPIError(Exception):
    """
    Telegram API error with code and retry guidance.
    
    Attributes:
        message: Error description
        code: Telegram error code (None for network errors)
        classified: Error classification for retry decisions
    """
    
    def __init__(
        self,
        message: str,
        code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retry_after = retry_after
    
    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


# ============================================================================
# Telegram Bot Client
# ============================================================================

class TelegramBotClient(TelegramClient):
    """
    HTTP-based Telegram Bot API client.
    
    Features:
    - Automatic retry with exponential backoff for retryable errors
    - Error classification and proper error propagation
    - Rate limit handling (429 responses)
    - Structured logging with request tracking
    
    Retry strategy:
    - Network timeouts and connection errors: exponential backoff
    - 429 Rate Limited: retry after specified time
    - 5xx Server errors: exponential backoff
    - 4xx Client errors: fail immediately (except 429)
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    REQUEST_TIMEOUT = 30.0
    
    def __init__(
        self,
        bot_token: str,
        logger: logging.Logger | None = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize Telegram bot client.
        
        Args:
            bot_token: Bot token from @BotFather
            logger: Optional logger instance
            retry_config: Retry behavior configuration
        """
        self._token = bot_token
        self._logger = logger or LOGGER
        self._retry_config = retry_config or RetryConfig()
        self._client: httpx.AsyncClient | None = None
        
        # Request metrics
        self._request_count = 0
        self._error_count = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
                follow_redirects=True,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def _build_url(self, method: str) -> str:
        """Build URL for API method."""
        return self.BASE_URL.format(token=self._token, method=method)
    
    async def _request(
        self,
        method: str,
        retry_count: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make API request with retry logic.
        
        Args:
            method: API method name
            retry_count: Current retry attempt
            **kwargs: Arguments to pass to request
            
        Returns:
            API response data
            
        Raises:
            TelegramAPIError: On API errors
        """
        self._request_count += 1
        
        client = await self._get_client()
        url = self._build_url(method)
        
        log_telegram_event(
            self._logger,
            level=logging.DEBUG,
            event="telegram.api.request",
            message=f"Telegram API request: {method}",
            method=method,
            retry_count=retry_count,
            request_id=self._request_count,
        )
        
        try:
            response = await client.post(url, **kwargs)
            
            # Handle HTTP errors
            if response.status_code == 429:
                # Rate limited
                retry_after = self._extract_retry_after(response)
                self._error_count += 1
                
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.api.rate_limited",
                    message="Telegram API rate limited",
                    method=method,
                    retry_after=retry_after,
                )
                
                if retry_count < self._retry_config.max_attempts:
                    wait_time = retry_after or self._calculate_delay(retry_count)
                    await asyncio.sleep(wait_time)
                    return await self._request(method, retry_count + 1, **kwargs)
                
                raise TelegramAPIError(
                    "Rate limit exceeded",
                    code=429,
                    retry_after=retry_after,
                )
            
            # Server errors - retryable
            if response.status_code >= 500:
                self._error_count += 1
                
                log_telegram_event(
                    self._logger,
                    level=logging.WARNING,
                    event="telegram.api.server_error",
                    message=f"Telegram server error: {response.status_code}",
                    method=method,
                    status_code=response.status_code,
                    retry_count=retry_count,
                )
                
                if retry_count < self._retry_config.max_attempts:
                    delay = self._calculate_delay(retry_count)
                    await asyncio.sleep(delay)
                    return await self._request(method, retry_count + 1, **kwargs)
                
                response.raise_for_status()
            
            # Client errors - non-retryable
            if response.status_code >= 400:
                response.raise_for_status()
            
            # Success
            data = response.json()
            
            if not data.get("ok"):
                error_code = data.get("error_code")
                description = data.get("description", "Unknown error")
                
                self._error_count += 1
                
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.api.error",
                    message=f"Telegram API error: {description}",
                    method=method,
                    error_code=error_code,
                    description=description,
                    retry_count=retry_count,
                )
                
                # Classify error for retry decisions
                classified = classify_telegram_error(
                    TelegramAPIError(description, error_code)
                )
                
                if classified.is_retryable and retry_count < self._retry_config.max_attempts:
                    delay = self._calculate_delay(retry_count)
                    log_telegram_event(
                        self._logger,
                        level=logging.INFO,
                        event="telegram.api.retrying",
                        message=f"Retrying after error: {delay:.2f}s",
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                    return await self._request(method, retry_count + 1, **kwargs)
                
                raise TelegramAPIError(description, error_code)
            
            return data.get("result", {})
            
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            self._error_count += 1
            
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.api.timeout",
                message=f"Telegram API timeout: {exc}",
                method=method,
                retry_count=retry_count,
            )
            
            if retry_count < self._retry_config.max_attempts:
                delay = self._calculate_delay(retry_count)
                await asyncio.sleep(delay)
                return await self._request(method, retry_count + 1, **kwargs)
            
            raise TelegramAPIError(f"Request timeout after {retry_count + 1} attempts") from exc
            
        except httpx.ConnectError as exc:
            self._error_count += 1
            
            log_telegram_event(
                self._logger,
                level=logging.WARNING,
                event="telegram.api.connection_error",
                message=f"Telegram connection error: {exc}",
                method=method,
                retry_count=retry_count,
            )
            
            if retry_count < self._retry_config.max_attempts:
                delay = self._calculate_delay(retry_count)
                await asyncio.sleep(delay)
                return await self._request(method, retry_count + 1, **kwargs)
            
            raise TelegramAPIError(f"Connection failed after {retry_count + 1} attempts") from exc
            
        except httpx.HTTPStatusError as exc:
            self._error_count += 1
            
            # Already handled status codes above
            if exc.response.status_code not in (429, 500, 501, 502, 503, 504):
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.api.http_error",
                    message=f"HTTP error: {exc}",
                    method=method,
                    status_code=exc.response.status_code,
                )
            
            raise
    
    def _calculate_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self._retry_config.initial_delay * (
            self._retry_config.multiplier ** retry_count
        )
        return min(delay, self._retry_config.max_delay)
    
    def _extract_retry_after(self, response: httpx.Response) -> float | None:
        """Extract retry_after from rate limit response."""
        try:
            data = response.json()
            if "parameters" in data:
                return data["parameters"].get("retry_after")
        except Exception:
            pass
        
        # Fallback to Retry-After header
        return response.headers.get("retry-after")
    
    # ------------------------------------------------------------------------
    # API Methods
    # ------------------------------------------------------------------------
    
    async def get_me(self) -> dict[str, Any]:
        """Get bot information."""
        return await self._request("getMe")
    
    async def get_updates(
        self,
        offset: int = 0,
        timeout: int = 0,
        allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get updates from Telegram."""
        payload: dict[str, Any] = {
            "offset": offset,
            "timeout": timeout,
        }
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        
        result = await self._request("getUpdates", json=payload)
        return result if isinstance(result, list) else []
    
    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "Markdown",
    ) -> dict[str, Any]:
        """Send text message to chat."""
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        
        log_telegram_event(
            self._logger,
            level=logging.DEBUG,
            event="telegram.api.send_message",
            message="Sending Telegram message",
            chat_id=chat_id,
            text_length=len(text),
        )
        
        return await self._request("sendMessage", json=payload)

    async def get_file(self, file_id: str) -> dict[str, Any]:
        """Fetch Telegram file metadata for a media attachment."""
        payload = {"file_id": file_id}
        return await self._request("getFile", json=payload)

    async def download_file(self, file_id: str, destination: str | Path) -> Path:
        """Download a Telegram-hosted file to the given destination path."""
        file_info = await self.get_file(file_id)
        file_path = file_info.get("file_path")
        if not file_path:
            raise TelegramAPIError("Telegram did not return file_path for the requested media")

        client = await self._get_client()
        download_url = f"https://api.telegram.org/file/bot{self._token}/{file_path}"
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        log_telegram_event(
            self._logger,
            level=logging.DEBUG,
            event="telegram.api.download_file",
            message="Downloading Telegram media file",
            file_id=file_id,
            destination=str(destination_path),
        )

        try:
            response = await client.get(download_url)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TelegramAPIError(f"File download timeout: {exc}") from exc
        except httpx.ConnectError as exc:
            raise TelegramAPIError(f"File download connection error: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise TelegramAPIError(
                f"Telegram file download failed with status {exc.response.status_code}",
                code=exc.response.status_code,
            ) from exc

        destination_path.write_bytes(response.content)
        return destination_path
    
    async def send_voice(
        self,
        chat_id: int,
        audio: bytes,
        caption: str | None = None,
        duration: int | None = None,
    ) -> dict[str, Any]:
        """
        Send voice message to chat.
        
        Args:
            chat_id: Target chat ID
            audio: Audio bytes in OGG/OPUS format
            caption: Optional caption
            duration: Optional duration in seconds
            
        Returns:
            API response
        """
        # Prepare multipart form data
        files = {
            "voice": ("voice.ogg", io.BytesIO(audio), "audio/ogg"),
        }
        
        data: dict[str, Any] = {
            "chat_id": chat_id,
        }
        
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "Markdown"
        
        if duration:
            data["duration"] = duration
        
        client = await self._get_client()
        url = self._build_url("sendVoice")
        
        log_telegram_event(
            self._logger,
            level=logging.DEBUG,
            event="telegram.api.send_voice",
            message="Sending Telegram voice",
            chat_id=chat_id,
            audio_size=len(audio),
        )
        
        try:
            response = await client.post(
                url,
                data=data,
                files=files,
            )
            
            if response.status_code == 429:
                retry_after = self._extract_retry_after(response)
                raise TelegramAPIError(
                    "Rate limit exceeded for voice send",
                    code=429,
                    retry_after=retry_after,
                )
            
            response.raise_for_status()
            result = response.json()
            
            if not result.get("ok"):
                error_code = result.get("error_code")
                description = result.get("description", "Unknown error")
                
                log_telegram_event(
                    self._logger,
                    level=logging.ERROR,
                    event="telegram.api.voice_error",
                    message=f"Telegram voice send error: {description}",
                    error_code=error_code,
                    description=description,
                )
                raise TelegramAPIError(description, error_code)
            
            return result.get("result", {})
            
        except httpx.TimeoutException as exc:
            raise TelegramAPIError(f"Voice send timeout: {exc}") from exc
        except httpx.ConnectError as exc:
            raise TelegramAPIError(f"Voice send connection error: {exc}") from exc
