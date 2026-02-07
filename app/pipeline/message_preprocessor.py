from typing import List

from app.core.logging import get_logger
from app.schemas.request import Message
from app.services.ocr_service import extract_text_from_image_url

logger = get_logger(__name__)


def normalize_messages_with_ocr(messages: List[Message]) -> List[Message]:
    processed: List[Message] = []
    for message in messages:
        if message.type != "URL":
            processed.append(message)
            continue

        try:
            extracted_text = extract_text_from_image_url(message.content)
        except Exception as exc:
            logger.warning("OCR failed for URL message (%s): %s", message.content, exc)
            processed.append(message)
            continue

        if not extracted_text:
            logger.warning("OCR returned empty text for URL message: %s", message.content)
            processed.append(message)
            continue

        processed.append(
            Message(
                type="TEXT",
                content=extracted_text,
                sender=message.sender,
                timestamp=message.timestamp,
            )
        )

    return processed
