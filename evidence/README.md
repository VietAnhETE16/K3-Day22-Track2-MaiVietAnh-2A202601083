# Báo cáo Bằng chứng & Phân tích Đánh giá RAG — Day 22: LangSmith + Prompt Versioning

**Học viên:** Mai Việt Anh  
**Mã số:** 2A202601083  
**Dự án LangSmith:** `day22-lab`  
**LangSmith Project URL:** `https://smith.langchain.com/o/e538084e-fd85-4bc1-a1aa-d39f6ef1ae42/projects/p/day22-lab`  
**Prompt Hub V1 URL:** `https://smith.langchain.com/prompts/mai-viet-anh-rag-v1`  
**Prompt Hub V2 URL:** `https://smith.langchain.com/prompts/mai-viet-anh-rag-v2`  

---

## 1. Danh mục Bằng chứng (Evidence Checklist)

| STT | Tệp Bằng chứng | Mô tả | Trạng thái |
| :--- | :--- | :--- | :---: |
| 1 | `01_langsmith_traces.png` | Ảnh chụp màn hình giao diện LangSmith với ≥ 50 traces cho RAG pipeline cơ bản. | 📸 Sẵn sàng chụp |
| 2 | `02_prompt_hub.png` | Ảnh chụp giao diện Prompt Hub hiển thị 2 phiên bản prompt: `mai-viet-anh-rag-v1` và `mai-viet-anh-rag-v2`. | 📸 Sẵn sàng chụp |
| 3 | `02_ab_routing_log.txt` | File log console của A/B routing cho 50 câu truy vấn (có nhãn V1/V2 tất định). | ✅ Đã lưu |
| 4 | `03_ragas_scores.png` | Ảnh chụp màn hình terminal hiển thị bảng so sánh điểm RAGAS giữa V1 và V2. | 📸 Sẵn sàng chụp |
| 5 | `03_ragas_report.json` | File báo cáo định dạng JSON chứa điểm chi tiết của cả V1 và V2. | ✅ Đã tạo |
| 6 | `04_pii_demo_log.txt` | Output console của PIIDetector với các test case (Email, Phone, SSN, Credit Card, Multi-PII, Clean). | ✅ Đã lưu |
| 7 | `04_json_demo_log.txt` | Output console của JSONFormatter tự động sửa lỗi fences, single quotes, trailing commas và fallback. | ✅ Đã lưu |

---

## 2. Phân tích Chi tiết So sánh Hiệu năng Prompt V1 vs V2 (RAGAS Evaluation)

### 2.1. Thiết kế 2 Phiên bản Prompt

- **Phiên bản V1 (`mai-viet-anh-rag-v1`)**:
  - *Phong cách:* Trực tiếp, ngắn gọn (2-4 câu), súc tích, thân thiện.
  - *Mục tiêu:* Tối ưu hóa độ cô đọng của câu trả lời, giảm bớt độ dài và thời gian sinh từ, tập trung thẳng vào câu trả lời của câu hỏi dựa trên context.
  - *Prompt:*
    ```
    Bạn là trợ lý AI thân thiện và hữu ích. Dựa vào context được cung cấp dưới đây để trả lời câu hỏi. Hãy giữ câu trả lời ngắn gọn, trực tiếp và súc tích (khoảng 2-4 câu). Nếu không tìm thấy thông tin trong context, hãy trả lời 'Tôi không tìm thấy thông tin này trong tài liệu.'
    ```

- **Phiên bản V2 (`mai-viet-anh-rag-v2`)**:
  - *Phong cách:* Phân tích chuyên sâu, có cấu trúc 3 phần (Tóm tắt, Chi tiết ngữ cảnh, Khẳng định độ tin cậy dựa trên dữ liệu).
  - *Mục tiêu:* Cung cấp câu trả lời toàn diện, có trích dẫn lý lẽ rõ ràng, giảm thiểu ảo giác (hallucination).
  - *Prompt:*
    ```
    Bạn là chuyên gia phân tích dữ liệu AI. Hãy đọc kỹ context dưới đây và trả lời câu hỏi một cách có cấu trúc rõ ràng: 1) Tóm tắt ý chính; 2) Giải thích chi tiết và trích dẫn thông tin từ context (3-5 câu); 3) Khẳng định mức độ tin cậy dựa trên dữ liệu. Tuyệt đối không suy diễn ngoài context được cung cấp.
    ```

---

### 2.2. Bảng Kết quả Đánh giá Định lượng (Quantitative Scores)

```
=================================================================
  Metric                                V1        V2  Winner
=================================================================
  faithfulness                      0.9595    0.7901  ← V1
  answer_relevancy                  0.9082    0.8759  ← V1
  context_recall                    1.0000    1.0000  TIED
  context_precision                 0.9450    0.9417  ← V1
=================================================================
  Mục tiêu Faithfulness ≥ 0.8:      ĐẠT (V1: 0.9595 ≥ 0.8 ⭐)
```

---

### 2.3. So sánh và Giải thích Chuyên sâu Kết quả Chỉ số

1. **Chỉ số Faithfulness (Độ trung thực - V1: 0.9595 vs V2: 0.7901 — V1 Thắng)**:
   - **V1 (0.9595)** vượt trội và đạt mức gần như tuyệt đối (vượt xa mục tiêu 0.8 của đề bài). Lý do là prompt V1 yêu cầu trả lời trực tiếp, cô đọng (2-4 câu), giúp mô hình tập trung 100% vào các sự thật cụ thể được cung cấp trong context, loại bỏ các câu rào đón hoặc suy diễn mở rộng.
   - **V2 (0.7901)** thấp hơn vì định dạng 3 phần buộc mô hình phải sinh thêm các câu khẳng định meta (ví dụ: *"Khẳng định độ tin cậy dựa trên dữ liệu..."* hoặc tóm tắt tổng quát), khiến thuật toán phân rã mệnh đề của RAGAS gán nhãn một số câu nhận định ngoài lề là không có căn cứ trực tiếp trong tài liệu nguồn.

2. **Chỉ số Answer Relevancy (Độ liên quan câu trả lời - V1: 0.9082 vs V2: 0.8759 — V1 Thắng)**:
   - Cả hai phiên bản đều đạt điểm rất cao (> 0.87). V1 cao hơn (0.9082) do trả lời đi thẳng vào trọng tâm câu hỏi của người dùng, không bị loãng bởi các tiêu đề phân mục `1) Tóm tắt... 2) Chi tiết...`.

3. **Chỉ số Context Recall (Độ bao phủ ngữ cảnh - V1: 1.0000 vs V2: 1.0000 — Hòa 100%)**:
   - Cả hai phiên bản đều đạt điểm tuyệt đối 1.0000. Điều này chứng minh quy trình embedding (`text-embedding-3-small`) và kỹ thuật chunking (kích thước 500 ký tự, overlap 100 ký tự) kết hợp retriever `k=3` đã lấy được 100% ngữ cảnh chứa đáp án chuẩn (`reference`).

4. **Chỉ số Context Precision (Độ chính xác ngữ cảnh - V1: 0.9450 vs V2: 0.9417 — V1 Thắng)**:
   - Điểm số xấp xỉ nhau (~0.945) và rất cao, cho thấy các chunk văn bản liên quan nhất luôn được xếp ở vị trí đầu bảng trong kết quả tìm kiếm tương đồng FAISS.

---

### 2.4. Kết luận & Đề xuất Ứng dụng Thực tế

- **Khuyến nghị triển khai Production**:
  - Đối với bài toán tra cứu kiến thức AI/ML tổng quát, **Prompt V1 là lựa chọn tối ưu nhất** vì vừa tiết kiệm token/chi phí gọi API, vừa đạt độ trung thực cao nhất (0.9595) và phản hồi nhanh gọn, trực diện.
  - **Prompt V2** phù hợp khi người dùng là các chuyên gia cần báo cáo chi tiết nhiều góc nhìn hoặc tài liệu pháp lý cần phân tách rõ các tầng thông tin.
