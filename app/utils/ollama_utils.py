"""Utility for generating clinical summaries via the local Ollama (llama3.2:3b)."""

import httpx

from app.core.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL


def generate_clinical_summary(detections: list[dict]) -> str:
    """Call local Ollama llama3.2:3b to produce a clinical narrative for the detections.

    Args:
        detections: List of dicts with keys: class, confidence, box.

    Returns:
        A string containing the AI-generated clinical summary.
        Falls back to a structured text summary if Ollama is unreachable.
    """
    if not detections:
        return (
            "No fractures or abnormalities were detected in this X-ray image. "
            "The AI analysis indicates a normal radiological appearance. "
            "However, clinical correlation is always recommended."
        )

    # Build a concise finding list for the prompt
    findings_text = "\n".join(
        f"- Finding {i + 1}: {d['class']} (confidence: {d['confidence'] * 100:.0f}%)"
        for i, d in enumerate(detections)
    )

    prompt = f"""You are a medical radiology AI assistant generating a professional clinical summary for an X-ray diagnosis report.

The AI detection system has analyzed a bone X-ray and found the following:

{findings_text}

Total findings: {len(detections)}

Write a professional clinical summary paragraph (150-250 words) that:
1. Summarizes the detected findings clearly
2. Describes the potential clinical significance
3. Suggests appropriate next steps (e.g., orthopedic consultation)
4. Uses professional medical terminology while remaining understandable
5. Includes a note that this is AI-assisted and should be reviewed by a qualified physician

Write ONLY the summary paragraph, no headings or bullet points."""

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 400,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            summary = data.get("response", "").strip()

            if summary:
                logger.info("Ollama clinical summary generated successfully")
                return summary

    except httpx.ConnectError:
        logger.warning("Ollama is not reachable — using fallback summary")
    except httpx.TimeoutException:
        logger.warning("Ollama timed out — using fallback summary")
    except Exception as e:
        logger.error(f"Ollama error: {e}")

    # Fallback: structured summary without LLM
    classes = [d["class"] for d in detections]
    unique_classes = list(set(classes))
    avg_conf = sum(d["confidence"] for d in detections) / len(detections)

    return (
        f"AI-assisted radiological analysis has identified {len(detections)} finding(s) "
        f"in this X-ray image. The detected conditions include: "
        f"{', '.join(unique_classes)}. "
        f"The average detection confidence is {avg_conf * 100:.0f}%. "
        f"These findings may indicate bone pathology requiring further clinical evaluation. "
        f"It is recommended that the patient consult an orthopedic specialist for a "
        f"comprehensive assessment. Please note that this analysis is AI-generated and "
        f"should be confirmed by a qualified healthcare professional."
    )
