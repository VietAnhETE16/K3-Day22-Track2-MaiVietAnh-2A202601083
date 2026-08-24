"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "mai-viet-anh-rag-v1"
PROMPT_V2_NAME = "mai-viet-anh-rag-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1: Phong cách ngắn gọn, trả lời 2-4 câu, thân thiện
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện và hữu ích. Dựa vào context được cung cấp dưới đây để trả lời câu hỏi. "
    "Hãy giữ câu trả lời ngắn gọn, trực tiếp và súc tích (khoảng 2-4 câu). "
    "Nếu không tìm thấy thông tin trong context, hãy trả lời 'Tôi không tìm thấy thông tin này trong tài liệu.'\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2: Phong cách có cấu trúc, chuyên gia phân tích, 3-5 câu
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích dữ liệu AI. Hãy đọc kỹ context dưới đây và trả lời câu hỏi một cách có cấu trúc rõ ràng: "
    "1) Tóm tắt ý chính; 2) Giải thích chi tiết và trích dẫn thông tin từ context (3-5 câu); 3) Khẳng định mức độ tin cậy dựa trên dữ liệu. "
    "Tuyệt đối không suy diễn ngoài context được cung cấp.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1: Phong cách ngắn gọn, thân thiện")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2: Phong cách chuyên gia, có cấu trúc")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}' (chi tiết: {e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}' (chi tiết: {e})")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    # Push cả 2 prompts lên Hub
    push_prompts_to_hub(client)

    # Pull cả 2 prompts từ Hub (dùng dict trả về)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    log_lines = []
    header = "=" * 60 + "\n  Bước 2: Prompt Hub & A/B Routing\n" + "=" * 60
    log_lines.append(header)
    log_lines.append(f"Prompt V1: {PROMPT_V1_NAME}")
    log_lines.append(f"Prompt V2: {PROMPT_V2_NAME}\n")

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        line1 = f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}..."
        line2 = f"     A: {str(result['answer'])[:80]}...\n"
        print(line1, flush=True)
        print(line2, flush=True)
        log_lines.append(line1)
        log_lines.append(line2)

    summary_line = f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}"
    print(summary_line, flush=True)
    log_lines.append(summary_line)
    log_lines.append("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")

    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "02_ab_routing_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"💾 Đã lưu log vào {evidence_dir / '02_ab_routing_log.txt'}", flush=True)


if __name__ == "__main__":
    main()
