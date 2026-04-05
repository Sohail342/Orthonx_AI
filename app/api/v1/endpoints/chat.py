"""AI Chatbot WebSocket streaming for both authenticated and guest users."""

import json
import os
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from redis import Redis

from app.core.security import get_jwt_strategy
from app.models.users import User
from app.utils.logging_utils import get_logger
from app.utils.ollama_utils import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.utils.rate_limiter import check_rate_limit, get_redis_client

logger = get_logger(__name__)

router = APIRouter()

# Knowledge Base Path
KNOWLEDGE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "static", "CHATBOT_KNOWLEDGE.md"
)


# Cache for Knowledge Base Content
_KNOWLEDGE_CACHE = ""


def read_knowledge_base() -> str:
    """Read the chatbot knowledge from memory (cached) or file."""
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE:
        return _KNOWLEDGE_CACHE

    if os.path.exists(KNOWLEDGE_FILE_PATH):
        try:
            with open(KNOWLEDGE_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    _KNOWLEDGE_CACHE = content
                    logger.info(
                        f"Chatbot context cached in memory from {KNOWLEDGE_FILE_PATH}"
                    )
                    return content
        except Exception as e:
            logger.error(f"Failed to read knowledge file {KNOWLEDGE_FILE_PATH}: {e}")
    return ""


def refresh_knowledge_cache():
    """Manual refresh of the knowledge cache."""
    global _KNOWLEDGE_CACHE
    _KNOWLEDGE_CACHE = ""
    return read_knowledge_base()


async def get_user_from_token(token: Optional[str]) -> Optional[User]:
    """Manually decode JWT token to identify the user for WebSockets."""
    if not token:
        return None
    try:
        strategy = get_jwt_strategy()
        # Local import to avoid circular dependencies
        from app.users.manager import get_user_manager_context

        async with get_user_manager_context() as user_manager:
            user = await strategy.read_token(token, user_manager)
            return user
    except Exception as e:
        logger.warning(f"WebSocket auth failed: {e}")
        return None


async def stream_chat_with_llama(query: str, context: str) -> AsyncGenerator[str, None]:
    """Stream chunks from the local Ollama instance."""
    system_prompt = (
        "You are an AI assistant for the Orthonx platform. Your name is 'Orthonx Assistant'. "
        "Use the following knowledge base content to answer user questions about Orthonx. "
        "Keep your answers concise and professional.\n\n"
        "### KNOWLEDGE BASE CONTENT ###\n"
        f"{context}\n\n"
        "### USER QUERY ###\n"
        f"{query}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": True,  # Enable streaming
        "keep_alive": -1,  # Keep in memory indefinitely
        "options": {
            "temperature": 0.3,
            "num_predict": 500,
            "num_thread": 4,  # Use 4 cores for generation
            "num_ctx": 4096,  # Context window size
        },
    }

    try:
        # Increased timeout to 120s for slow model loads
        timeout = httpx.Timeout(120.0, connect=10.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama API returned status {response.status_code}")
                    yield "I'm having trouble connecting to my AI core right now. Please try again."
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        text = chunk.get("response", "")
                        if text:
                            yield text
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ReadTimeout:
        logger.error("Ollama stream timed out (ReadTimeout) after 120s")
        yield "The AI is taking too long to respond. Please try again."
    except httpx.TimeoutException as e:
        logger.error(f"Ollama stream timed out: {repr(e)}")
        yield "The request timed out. Please try a shorter question."
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to Ollama: {repr(e)}")
        yield "I'm offline for maintenance. Please check back shortly!"
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(
            f"Ollama stream error ({type(e).__name__}): {repr(e)}\n{error_details}"
        )
        yield "I've encountered an unexpected error. Please try again later."


@router.websocket("/ws")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    redis: Redis = Depends(get_redis_client),
):
    """WebSocket endpoint for real-time AI chat (streaming)."""
    await websocket.accept()

    # Identify user (Auth vs Guest)
    user = await get_user_from_token(token)
    ip_address = websocket.client.host if websocket.client else "unknown"

    logger.info(
        f"WebSocket connection: user={user.email if user else 'guest'} ip={ip_address}"
    )

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            try:
                msg_data = json.loads(data)
                user_query = msg_data.get("message", "")
            except json.JSONDecodeError:
                user_query = data  # Fallback to raw text

            if not user_query.strip():
                continue

            #  Rate Limiting (10/hr) - Only for Guests
            remaining = -1
            if not user:
                is_allowed, remaining = check_rate_limit(
                    redis, ip_address, limit_type="chat", limit=10
                )
                if not is_allowed:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": 429,
                            "message": "Rate limit exceeded. Please get started for free to continue chatting.",
                        }
                    )
                    continue

            # Get Context
            context = read_knowledge_base()

            # Stream back the response
            await websocket.send_json({"type": "start", "remaining": remaining})

            async for chunk in stream_chat_with_llama(user_query, context):
                await websocket.send_json({"type": "chunk", "text": chunk})

            # Inform client we are done
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {ip_address}")
    except Exception as e:
        logger.error(f"WebSocket error for {ip_address}: {e}")
        try:
            await websocket.close()
        except Exception as close_error:
            logger.warning(f"Error closing WebSocket for {ip_address}: {close_error}")
            pass
