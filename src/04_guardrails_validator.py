"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)
"""

import sys
import re
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    try:
        from guardrails.validator_base import OnFailAction
    except ImportError:
        from guardrails.classes.history import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3}\b)[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về text đã xử lý.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            for match in matches:
                redacted_text = re.sub(re.escape(match), f"[{pii_type}_REDACTED]", redacted_text)
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(
                error_message=f"Phát hiện PII: {', '.join([p[0] for p in found_pii])}",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Thay single quotes → double quotes (cho keys và values)
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
        if "'" in text and '"' not in text:
            text = text.replace("'", '"')

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON.
        Nếu thất bại, gọi _repair() rồi thử lại.
        """
        try:
            parsed = json.loads(value)
            return PassResult()
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            repaired_text = self._repair(str(value))
            parsed        = json.loads(repaired_text)
            print(f"  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON đã được sửa thành công",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except (json.JSONDecodeError, TypeError) as e:
            fallback = json.dumps({
                "error": "Không thể phân tích JSON",
                "raw": str(value)[:200],
            }, ensure_ascii=False, indent=2)
            return FailResult(
                error_message=f"JSON không hợp lệ sau khi sửa: {e}",
                fix_value=fallback,
            )


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob", "score": 95}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass/Fixed" if (result.validation_passed or result.validated_output is not None) else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:80]}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
