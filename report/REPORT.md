# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Vọng Triển
**Nhóm:** X2
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:*

High cosine similarity nghĩa là hai đoạn văn bản có hướng vectơ rất gần nhau trong không gian, cho thấy chúng có sự tương đồng lớn về mặt nội dung và ngữ nghĩa dù cách dùng từ có thể khác nhau

**Ví dụ HIGH similarity:**

- Sentence A: Dựa trên số liệu từ World Health Organization vào năm 2018, Việt Nam xếp hạng 16 trong nhóm 30 quốc gia có số lượng ca nhiễm Tuberculosis lớn nhất hành tinh, đồng thời giữ vị trí thứ 15 trong danh sách 30 nước chịu áp lực nặng nề nhất về tình trạng lao kháng đa thuốc trên toàn cầu.

- Sentence B: Theo báo cáo của của tổ chức Y tế Thế giới WHO công bố năm 2018, Việt Nam đứng thứ 16 trong 30 nước có số người bệnh lao cao nhất trên toàn cầu, đồng thời đứng thứ 15 trong số 30 nước có gánh nặng bệnh lao kháng đa thuốc cao nhất thế giới.

- Tại sao tương đồng: Hai câu này truyền tải cùng một thông tin về số liệu thống kê và thứ hạng của Việt Nam, chỉ khác biệt ở cách diễn đạt thuật ngữ Anh-Việt và cấu trúc câu.

**Ví dụ LOW similarity:**
- Sentence A: Sức khỏe là vốn quý nhất của con người nên chúng ta cần tập thể dục thường xuyên.

- Sentence B: Việt Nam đứng thứ 16 trong 30 nước có số người bệnh lao cao nhất trên toàn cầu theo báo cáo của WHO.

- Tại sao khác: Hai câu này hoàn toàn lệch nhau về chủ đề; một bên nói về lời khuyên rèn luyện sức khỏe cá nhân, một bên nói về số liệu y tế công cộng về bệnh lao.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*  

Cosine similarity được ưu tiên vì nó tập trung vào hướng của vectơ (nội dung) thay vì độ dài, giúp so sánh chính xác hai văn bản có độ dài ngắn khác nhau nhưng cùng chủ đề. Điều này giúp tránh việc hai bài viết giống hệt nhau bị coi là khác biệt chỉ vì một bài viết dài hơn bài kia.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*

 Sử dụng công thức tính số lượng chunks ($n$):$$n = \left\lceil \frac{L - O}{S - O} \right\rceil$$Trong đó:

 $L = 10,000$ (Độ dài văn bản)
 
 $S = 500$ (Kích thước chunk)
 
 $O = 50$ (Độ chồng lấp - Overlap)
 
 Tính toán:$$n = \frac{10,000 - 50}{500 - 50} = \frac{9,950}{450} \approx 22.11$$

> *Đáp án:*  23 chunks (Làm tròn lên để đảm bảo lấy hết dữ liệu ở đoạn cuối)

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*

Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên (khoảng 25 chunks) do bước nhảy của cửa sổ trượt bị thu hẹp lại. Việc tăng overlap giúp duy trì ngữ cảnh liền mạch giữa các đoạn văn bản, tránh việc các câu hoặc ý nghĩa quan trọng bị cắt đôi ở điểm giao nhau.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Bệnh

**Tại sao nhóm chọn domain này?**
> Vì nhóm quan tâm tới tìm hiểu bệnh và muốn ứng dụng RAG khi tra cứu question answering cho đề tài này

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | alzheimer.md| https://tamanhhospital.vn/alzheimer/| 27966 | source: "https://tamanhhospital.vn/alzheimer/", category: "bệnh thoái hóa thần kinh"|
| 2 | benh-san-day.md| https://tamanhhospital.vn/benh-san-day/ | 12700 | source: "https://tamanhhospital.vn/benh-san-day/", category: "bệnh ký sinh trùng" |
| 3 | benh-tri.md| https://tamanhhospital.vn/benh-tri/ | 12569 | source: "https://tamanhhospital.vn/benh-tri/", category: "bệnh lí hậu môn - trực tràng" |
| 4 | benh-dai.md| https://tamanhhospital.vn/benh-dai/ | 12700 | source: "https://tamanhhospital.vn/benh-dai/", category: "bệnh truyền nhiễm" |
| 5 | benh-lao-phoi.md| https://tamanhhospital.vn/benh-lao-phoi/ | 12704 | source: "https://tamanhhospital.vn/benh-lao-phoi/", category: "bệnh truyền nhiễm" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "https://tamanhhospital.vn/benh-tri/" | Dùng để trích dẫn nguồn (citation) trong câu trả lời của AI và giúp người dùng kiểm chứng thông tin gốc. |
| category | string | bệnh ký sinh trùng | Cho phép lọc (filter) nhanh các nhóm bệnh cụ thể, thu hẹp phạm vi tìm kiếm khi người dùng hỏi về một loại bệnh lý nhất định. |

---

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
| :--- | :--- | :---: | :---: | :--- |
| **benh-dai.md** | Fixed_sizeChunker (`fixed_size`) | 80 | 199.94 | No (Cắt ngang từ/câu) |
| **benh-dai.md** | By_sentencesChunker (`by_sentences`) | 29 | 414.00 | Yes (Giữ trọn câu) |
| **benh-dai.md** | RecursiveChunker (`recursive`) | 569 | 20.04 | Partial (Ưu tiên cấu trúc) |
| **benh-san-day.md** | Fixed_sizeChunker (`fixed_size`) | 100 | 198.55 | No (Cắt ngang từ/câu) |
| **benh-san-day.md** | By_sentencesChunker (`by_sentences`) | 42 | 353.45 | Yes (Giữ trọn câu) |
| **benh-san-day.md** | RecursiveChunker (`recursive`) | 658 | 21.44 | Partial (Ưu tiên cấu trúc) |
| **benh-tri.md** | Fixed_sizeChunker (`fixed_size`) | 78 | 198.69 | No (Cắt ngang từ/câu) |
| **benh-tri.md** | By_sentencesChunker (`by_sentences`) | 35 | 331.49 | Yes (Giữ trọn câu) |
| **benh-tri.md** | RecursiveChunker (`recursive`) | 639 | 17.08 | Partial (Ưu tiên cấu trúc) |

### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả cách hoạt động:**
Chiến lược này sử dụng biểu thức chính quy (Regex) để tách văn bản dựa trên các dấu hiệu kết thúc câu (., !, ?). Các câu sau đó được gom nhóm lại thành từng cụm dựa trên số lượng câu tối đa cấu hình sẵn thay vì dựa trên số lượng ký tự. Cách tiếp cận này đảm bảo mỗi mảnh văn bản (chunk) là một tập hợp các ý tưởng hoàn chỉnh, không bị ngắt quãng giữa chừng.

**Tại sao tôi chọn strategy này cho domain nhóm?**
*Viết 2-3 câu: Do tài liệu y khoa về các loại bệnh chứa nhiều thuật ngữ chuyên môn và quy trình điều trị phức tạp, việc giữ trọn vẹn câu văn là yếu tố cốt lõi để bảo toàn ngữ cảnh. Chiến lược này giúp AI truy xuất được chính xác các triệu chứng hoặc hướng dẫn sơ cứu mà không làm sai lệch ý nghĩa do bị cắt ngang dòng.


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| **benh-tri.md** | best baseline | 35 | 331.49 | Yes (Giữ trọn câu) |
| benh-tri.md| của tôi | 52     | 222.79  |Yes|

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 6/10 | Bảo toàn trọn vẹn ngữ nghĩa câu | Dễ mất ngữ cảnh liên kết giữa các câu|
| Yến | MarkdownHeadChunker | 8/10 | Từng vector là nội dung ngắn, thống nhất. Kèm với header 1,2 trong metadata giúp hệ thống retrieval có đủ thông tin từ đề mục | Phụ thuộc hoàn toàn vào định dạng gốc. Sẽ vô tác dụng nếu tài liệu là văn bản thô (plain text) không có sẵn các ký tự đánh dấu # hoặc các đoạn không có đánh dấu # như tiêu đề hoặc mở đầu|
| Minh | LateChunker | 8/10 |Giữ được tính mạch lạc và mối liên kết thông tin chặt chẽ nhờ việc duy trì ngữ cảnh lớn (long context) ở bước biểu diễn ban đầu (indexing). | Có thể gây ra hiện tượng trích xuất thừa thông tin không cần thiết nếu bước "late split" không được cấu hình độ dài cửa sổ (window size) phù hợp. |


**Strategy nào tốt nhất cho domain này? Tại sao?**
*Viết 2-3 câu:*  Chiến lược MarkdownHeadChunker tối ưu nhất. Bảo toàn ngữ cảnh (Context-Aware): Giữ trọn vẹn cấu trúc phân cấp tài liệu . Khi truy xuất, AI luôn biết chính xác đoạn text đang đọc thuộc bệnh lý nào.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach: Sử dụng regex để split text trên các dấu kết thúc câu như ". ", "! ", "? ", hoặc ".\n". Sau đó nhóm các câu thành chunks với số lượng tối đa là max_sentences_per_chunk, và loại bỏ khoảng trắng thừa ở mỗi chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
Mỗi document được embed bằng embedding_fn và lưu vào ChromaDB (nếu có) hoặc danh sách in-memory với id, content, embedding, và metadata. Search embed query và rank theo cosine similarity giữa embedding query và các embedding đã lưu.

**`search_with_filter` + `delete_document`** — approach:
earch_with_filter lọc metadata trước khi thực hiện similarity search. delete_document xóa tất cả records có id khớp với doc_id (đối với trường hợp không chunk, hoặc metadata['doc_id'] nếu có chunk).

### KnowledgeBaseAgent

**`answer`** — approach:
Retrieve top_k chunks liên quan nhất từ store. Xây dựng prompt với context là nội dung các chunks, và query là câu hỏi. Gọi LLM function để sinh ra câu trả lời dựa trên context.

### Test Results

Số tests pass: 42 / 42

collected 42 items                                                                                                                   

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                          [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                   [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                            [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                             [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                  [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                  [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                        [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                         [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                       [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                         [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                         [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                    [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                          [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                 [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                     [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                               [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                     [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                         [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                           [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                             [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                   [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                        [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                          [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                              [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                           [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                    [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                   [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                              [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                          [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                     [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                         [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                               [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                         [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                      [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                    [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                   [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                       [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                  [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                           [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                 [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                     [100%]

======================================================== 42 passed in 0.14s =========================================================

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|------------|------------|---------|--------------|-------|
| 1 | Thời gian ủ bệnh lao phổi khác nhau tùy vào sức đề kháng của từng người. | Giai đoạn ủ bệnh lao thường không có triệu chứng rõ ràng nên khó phát hiện. | high | 0.83 | Yes |
| 2 | Ho khan kéo dài là một dấu hiệu phổ biến của lao phổi. | Thị trường bất động sản đang có nhiều biến động trong năm nay. | low | 0.10 | Yes |
| 3 | Bệnh nhân lao phổi có thể ho ra đờm màu trắng hoặc lẫn máu. | Lao phổi có thể gây tổn thương nghiêm trọng đến hệ hô hấp. | high | 0.80 | Yes |
| 4 | Sốt nhẹ kéo dài hơn 3 tuần là dấu hiệu cần đi khám để kiểm tra lao phổi. | Việc tập thể dục thường xuyên giúp tăng cường sức khỏe tim mạch. | low | 0.18 | Yes |
| 5 | Khi nghi ngờ lao phổi, bác sĩ thường chỉ định chụp X-quang và xét nghiệm đờm. | Xét nghiệm đờm giúp phát hiện vi khuẩn lao trong cơ thể. | high | 0.87 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Cặp câu có cùng chủ đề (lao phổi) nhưng mô tả khác khía cạnh vẫn có similarity cao.
Ý nghĩa: Embeddings không chỉ dựa vào từ khóa giống nhau mà còn nắm được ngữ nghĩa tổng thể và ngữ cảnh, nên các câu liên quan về nội dung vẫn được xem là gần nhau dù diễn đạt khác.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)


| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không| Không |
| 2 | Ăn cá có bị sán không | Có. Có loại sán dây ở bên trong cá. Có khả năng lây bệnh cho người |
| 3 | Làm sao biết mình bị Alzheimer | Sa sút trí nhớ và khả năng nhận thức. Khó khăn diễn đạt bằng ngôn ngữ. Thay đổi hành vi, tâm trạng, tính cách. Nhầm lẫn thời gian hoặc địa điểm. Đặt đồ vật sai vị trí và không thể nhớ lại mình đã từng làm gì|
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ tình trạng lao tiềm ẩn sang bệnh lao phổi (lao bệnh)?| Người nhiễm HIV. Người sử dụng ma túy dạng chích. Người bị sụt cân (khoảng 10%). Bệnh nhân mắc bệnh bụi phổi silic, suy thận hoặc đang chạy thận, và bệnh đái tháo đường. Người từng thực hiện phẫu thuật cắt dạ dày hoặc ruột non. Người ghép tạng hoặc đang sử dụng thuốc corticoid kéo dài, thuốc ức chế miễn dịch. Bệnh nhân bị ung thư đầu cổ.|
| 5 | Trong trường hợp bị động vật cắn hoặc cào xước, quy trình sơ cứu tại chỗ và các biện pháp y tế cần thực hiện ngay lập tức là gì để ngăn chặn virus dại xâm nhập hệ thần kinh?|Dựa theo hướng dẫn từ tài liệu của Bệnh viện Đa khoa Tâm Anh, quy trình xử lý bao gồm các bước sau: Sơ cứu tại chỗ: Rửa ngay vết thương rộng bằng nước sạch và các dung dịch có khả năng tiêu diệt virus như xà phòng, chất tẩy rửa, povidone iodine,... trong ít nhất 15 phút. Sau khi rửa sạch, cần sát trùng vết thương bằng cồn 70% (ethanol) hoặc povidone-iodine. Băng bó đơn giản vết thương và nhanh chóng đưa nạn nhân đến cơ sở y tế. Can thiệp y tế: Nạn nhân cần được bác sĩ thăm khám để điều trị vết thương và chỉ định tiêm vắc xin phòng dại càng sớm càng tốt. Trong một số trường hợp cụ thể, bác sĩ có thể chỉ định tiêm thêm huyết thanh kháng dại để ngăn chặn sự khởi phát triệu chứng. Theo dõi động vật: Cần theo dõi con vật đã cắn; nếu vật nuôi có biểu hiện bất thường như cắn không lý do, ăn vật lạ, gầm gừ, tiết nước bọt quá mức hoặc chết sau vài ngày, khả năng mắc bệnh dại là rất cao. |

### Kết Quả Của Tôi

**Chunking Strategy:** SentenceChunker(max_sentences_per_chunk=4) + metadata-aware retrieval  
**Total Chunks:**  6/10 (tổng cộng từ 5 documents)

| # | Query | Top-1 Chunk (Global Search) | Score | Relevant? | Agent Answer |
|---|-------|-----------------------------|-------|-----------|--------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không | Tuổi thọ trung bình sau khi chẩn đoán là 8 đến 10 năm. Tuy nhiên, trong một số trường hợp, nó có thể ngắn hoặc kéo dài hơn. Mỗi người có một tiền sử sức khỏe riêng. Lịch sử sức khỏe này liên quan trực tiếp đến việc bệnh sẽ ảnh hưởng đến họ như thế nào.... | 0.4199 | No | [DEMO LLM] Generated answer from prompt preview: Context: Tuổi thọ trung bình sau khi chẩn đoán là 8 đến 10 năm. Tuy nhiên, trong một số trường hợp, n... |
| 2 | Ăn cá có bị sán không | Mặc dù vẫn chưa đưa ra bất kỳ câu trả lời rõ ràng nào, nhưng nhiều nghiên cứu cho thấy rằng các yếu tố sau có khả năng thúc đẩy bệnh phát triển: Bệnh tiểu đường; Stress, căng thẳng và phiền muộn kéo dài; Cholesterol cao; Hút thuốc; Ít giao tiếp xã hội. ## **Triệu chứng thường gặp của hội chứng Alzheimer**  Bộ não của chúng ta được tạo thành từ hơn 100 tỷ tế bào thần kinh – nhiều hơn cả các ngôi sa... | 0.3556 | No | [DEMO LLM] Generated answer from prompt preview: Context: Mặc dù vẫn chưa đưa ra bất kỳ câu trả lời rõ ràng nào, nhưng nhiều nghiên cứu cho thấy rằng ... |
| 3 | Làm sao biết mình bị Alzheimer | Các triệu chứng cũng có thể do trầm cảm hoặc các tình trạng thể chất khác như viêm màng não, đột quỵ hoặc chảy máu não. Sự thiếu hụt vitamin và khoáng chất, hoặc tác dụng phụ của thuốc đôi khi cũng có thể gây ra các triệu chứng như thế này. Một số phương pháp điều trị có thể giúp giảm nhẹ các triệu chứng, cải thiện dần và mất hẳn. Hiện tượng Alzheimer chỉ được chẩn đoán nếu các triệu chứng đã kéo ... | 0.4470 | Yes | [DEMO LLM] Generated answer from prompt preview: Context: Các triệu chứng cũng có thể do trầm cảm hoặc các tình trạng thể chất khác như viêm màng não,... |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ tình trạng lao tiềm ẩn sang bệnh lao phổi (lao bệnh) | Đối với các trường hợp bệnh khởi phát sớm khác, nghiên cứu đã chỉ ra rằng các thành phần di truyền khác có liên quan. Các nghiên cứu đang được tiến hành để xác định các biến thể nguy cơ di truyền bổ sung.... | 0.3288 | Yes | [DEMO LLM] Generated answer from prompt preview: Context: Đối với các trường hợp bệnh khởi phát sớm khác, nghiên cứu đã chỉ ra rằng các thành phần di ... |
| 5 | Quy trình sơ cứu tại chỗ khi bị động vật cắn để ngăn chặn virus dại là gì? | Việt Nam sử dụng vắc xin dại tế bào Verorab từ năm 1992. Với phác đồ tiêm bắp: Người bệnh được tiêm 0,5ml x 5 liều cho một đợt điều trị dự phòng vào các ngày 0, 3, 7, 14, 28. Với phác đồ tiêm trong da : Người bệnh được dùng liều đơn 0,1ml x 8 liều cho một đợt điều trị dự phòng vào các ngày 0, 3, 7. Lúc này, mỗi ngày tiêm 2 liều đơn vào 2 vị trí khác nhau của vùng cơ Delta.... | 0.3373 | Yes | [DEMO LLM] Generated answer from prompt preview: Context: Việt Nam sử dụng vắc xin dại tế bào Verorab từ năm 1992. Với phác đồ tiêm bắp: Người bệnh đư... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc tận dụng cấu trúc tài liệu (như Markdown headers) giúp cải thiện retrieval rõ rệt. Không chỉ nội dung, metadata cũng rất quan trọng để giữ ngữ cảnh. Điều này giúp kết quả tìm kiếm chính xác và dễ hiểu hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một số nhóm tối ưu chunk size và cách chia nhỏ dữ liệu để cân bằng giữa context và độ chính xác. Họ thử nghiệm nhiều strategy khác nhau thay vì chỉ dùng một cách cố định. Điều này giúp hệ thống linh hoạt hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ kết hợp nhiều strategy (hybrid), ví dụ vừa theo câu vừa theo cấu trúc heading. Đồng thời tối ưu chunk size để giữ đủ ngữ cảnh mà không quá dài. Ngoài ra, sẽ chú trọng metadata nhiều hơn để hỗ trợ retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 15/ 15 |
| My approach | Cá nhân | 8/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 8/ 10 |
| Core implementation (tests) | Cá nhân | 30/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | *86*/ 100** |

