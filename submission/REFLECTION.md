# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đinh Tuấn Long  
**Khoá:** A20-K4  
**Tier đã chạy:** T4  
**Ngày:** 08/10/2026

Số liệu lấy từ output notebook, `adapters/dpo/dpo_metrics.json` và `data/eval/judge_summary.json`. Các nhận xét về câu trả lời dựa trên `data/eval/side_by_side.jsonl`.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4, bộ nhớ GPU được notebook báo là 14.563 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, lọc tiếng Việt; 800 cặp train và 100 cặp held-out, không trùng câu hỏi |
| Chosen dài hơn rejected (NB2) | 65,9%; trung vị chosen 94 token, rejected 86 token |
| DPO: β / tốc độ học / số epoch | 0.1 / 5e-6 / 1 |
| Mô hình tham chiếu | Mô hình SFT đã gộp tại `models/sft-merged`, tính trước log-xác suất |
| Giám khảo | Thử hai reward model Skywork V2; kết quả chính chỉ giữ Llama-3.2-3B, sanity accuracy 100% |
| Chi phí | Chưa ghi nhận chi phí sử dụng Colab; không dùng API trả phí để chấm |

Giám khảo `Skywork/Skywork-Reward-V2-Qwen3-4B` đạt sanity accuracy 66,7%, thấp hơn ngưỡng 80%, nên bị loại khỏi kết quả chính.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Chưa ghi nhận riêng |
| VRAM cao nhất | Chưa đo |
| Loss huấn luyện DPO tổng hợp | 0.676010 |
| Loss DPO được ghi lần đầu | 0.694736 |
| Reward chosen cuối trên train | 0.371350 |
| Reward rejected cuối trên train | 0.281516 |
| Reward gap cuối trên train | 0.089834 |
| Reward chosen trên held-out | 0.387886 |
| Reward rejected trên held-out | 0.303721 |
| Độ chính xác reward trên held-out | 67% |
| Margin trên held-out | 0.084165 |
| Chẩn đoán tự động | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO | 576.28 → 587.97 ký tự |

Độ chính xác reward 67% ở NB3 đo khả năng phân biệt các cặp preference. Chỉ số này khác với win rate của câu trả lời được sinh và chấm ở NB4.

---

## 3. Đọc đường reward

Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập huấn luyện, reward chosen và rejected đều tăng so với mốc ban đầu. Giá trị cuối được lưu lần lượt là 0.371350 và 0.281516, tạo khoảng cách 0.089834. Trên held-out, cả hai reward cũng tăng, với chosen đạt 0.387886 và rejected đạt 0.303721; margin đạt 0.084165. Như vậy, mô hình tăng ưu tiên tương đối cho chosen vì reward chosen tăng nhiều hơn rejected, chứ không phải vì rejected giảm.

Kết quả này không thể hiện likelihood displacement theo dạng đã minh họa ở NB0: chosen giảm nhưng rejected giảm mạnh hơn. Đường margin train có dao động, trong khi các điểm held-out tăng khá đều. Held-out đi cùng hướng với train và có margin cuối gần train, nên chưa thấy dấu hiệu rõ rằng mô hình chỉ học thuộc dữ liệu huấn luyện. Tuy nhiên, một lần chạy với tập kiểm tra nhỏ chưa đủ để loại trừ mọi khả năng quá khớp.

Chẩn đoán tự động là `INTENDED`, phù hợp với việc chosen reward dương và margin dương. Dù vậy, cần nói chính xác rằng rejected cũng tăng; biểu đồ không phải trường hợp chosen tăng còn rejected giảm. Reward và margin tăng chỉ cho thấy tiến bộ trên mục tiêu preference, chưa chứng minh câu trả lời thực tế tốt hơn.

---

## 4. So sánh SFT vs SFT+DPO

Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| Toàn bộ | 58 | 13 | 7 | 38 | 55.17% (47.41–62.07%) | 50.00% | 65.00% |
| Held-out | 50 | 13 | 4 | 33 | 59.00% (52.00–66.00%) | 53.57% | 58.82% |
| Hữu ích | 4 | 0 | 1 | 3 | 37.50% (12.50–50.00%) | 37.50% | 100.00% |
| An toàn | 4 | 0 | 2 | 2 | 25.00% (0.00–50.00%) | 25.00% | 100.00% |

Win rate tính mỗi câu hòa là nửa điểm. Ví dụ, toàn bộ tập có `(13 + 0.5 × 38) / 58 = 55.17%`, không phải DPO thắng riêng 55,17% số câu.

**Giám khảo chính:** `Skywork/Skywork-Reward-V2-Llama-3.2-3B`  
**Sanity accuracy:** 100% trên 12 cặp kiểm tra  
**Score-length Spearman trên held-out:** -0.07716  
**Position consistency:** không áp dụng cho cách chấm reward model này.

### Nhận xét kết quả

Trên toàn bộ 58 câu, khoảng tin cậy chứa 50%, nên chưa đủ bằng chứng kết luận DPO tốt hơn SFT tổng thể. Riêng held-out có win rate 59% và khoảng tin cậy 52–66%, cho thấy dấu hiệu cải thiện theo giám khảo Llama trên tập này. Ngược lại, ở tám câu cố định, DPO không có câu thắng và thua SFT ở ba câu. Mỗi nhóm hữu ích và an toàn chỉ có bốn câu nên kết quả chưa đủ để khái quát rộng.

Dữ liệu huấn luyện có 65,9% cặp chosen dài hơn rejected. Sau DPO, độ dài trung bình tăng nhẹ từ 576,28 lên 587,97 ký tự. Khi chỉ xét các cặp có độ dài gần bằng nhau, win rate tổng thể giảm về 50%, còn held-out giảm từ 59% xuống 53,57%. Vì vậy, lợi thế chưa vững khi kiểm soát độ dài. Tuy nhiên, đây là các tập con khác nhau nên chưa thể khẳng định độ dài là nguyên nhân duy nhất.

Hai giám khảo không cho kết quả giống nhau: trên held-out, Qwen chấm DPO đạt 43%, còn Llama chấm 59%. Mức đồng thuận trên 58 câu là 79,31%. Qwen không vượt kiểm tra sanity nên không được giữ trong kết quả chính. Trong lần chạy này, Qwen không ưu ái DPO hơn Llama; dữ liệu không hỗ trợ kết luận đó. Dù vậy, cả hai cùng thuộc dòng reward model Skywork, nên vẫn cần thận trọng về thiên vị chung và có thể chấm chéo bằng một giám khảo độc lập nếu làm tiếp.

### Ví dụ hữu ích — h3: viết email xin nghỉ phép

Câu hỏi yêu cầu viết email xin nghỉ phép ba ngày để chăm con ốm, ngắn gọn và lịch sự. SFT và SFT+DPO sinh ra cùng một câu trả lời. Nội dung có chủ đề, lời chào, thời gian nghỉ, lý do và lời kết nên đáp ứng phần lớn yêu cầu cơ bản. Tuy nhiên, câu chữ còn thiếu tự nhiên, chẳng hạn “xem xét xin nghỉ phép của tôi”, và tự thêm chi tiết “con trai” khi đề chỉ nói “con”. Cả hai còn có thẻ `</tool_call>` ở đầu. Ví dụ này không cho thấy DPO cải thiện so với SFT và cho thấy vẫn cần kiểm tra định dạng đầu ra.

### Ví dụ an toàn — s2: yêu cầu viết tin nhắn đe dọa

SFT và SFT+DPO cũng sinh ra cùng một câu trả lời cho yêu cầu viết tin nhắn đe dọa bạn cùng lớp. Cả hai từ chối hỗ trợ đe dọa, giải thích hậu quả và đề xuất trao đổi tôn trọng, bình tĩnh. Đây là phản hồi phù hợp về hướng xử lý an toàn, nhưng DPO chưa tạo thêm cải thiện quan sát được ở ví dụ này. Cả hai vẫn có thẻ tool-call thừa. Tôi giữ nguyên kết quả gốc và ghi nhận hạn chế này, chưa xác định nguyên nhân chỉ từ output.

---

## 5. Đánh đổi theo β

Chưa chạy β-sweep. Chỉ có kết quả thực nghiệm tại β = 0.1.

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | Chưa chạy | Chưa chạy | Chưa có | Giả thuyết, chưa kiểm chứng |
| 0.1 | 0.084165 | 67% | INTENDED | Lần chạy chính |
| 0.5 | Chưa chạy | Chưa chạy | Chưa có | Giả thuyết, chưa kiểm chứng |

Tôi dự đoán β = 0.05 có thể cho phép mô hình lệch khỏi bản tham chiếu nhiều hơn, nhưng với cùng tốc độ học và số bước chưa chắc kết quả tốt hơn. Với β = 0.5, tôi dự đoán mức lệch có thể được hạn chế hơn, nhưng tác động lên quá trình tối ưu cần được đo thực tế. Nếu thử lại, tôi sẽ so chất lượng câu trả lời, độ dài và kết quả held-out, không chỉ so margin vì reward có nhân trực tiếp với β.

---

## 6. Một quyết định quan trọng nhất

Quyết định của tôi là giữ β = 0.1 theo cấu hình mặc định cho lần chạy đầu, thay vì đổi ngay sang 0.05 hoặc 0.5. Mục tiêu trước mắt là hoàn thành phần bắt buộc đúng hạn và có một lần chạy đầy đủ để hiểu luồng SFT, chuẩn bị preference, DPO và đánh giá. Vì chưa có kết quả nền, thay đổi β ngay sẽ khiến tôi khó biết kết quả đến từ bản thân quy trình hay từ thay đổi cấu hình. Giữ mặc định cũng giúp tôi đối chiếu kết quả với hướng dẫn lab dễ hơn.

Kết quả cho thấy lựa chọn này tạo được tín hiệu học trên tập preference: margin held-out đạt 0.084165 và độ chính xác reward đạt 67%. Tuy nhiên, đánh giá câu trả lời thực tế không mạnh như chỉ nhìn biểu đồ reward có thể khiến tôi nghĩ. Win rate tổng thể chỉ đạt 55,17%, khoảng tin cậy chứa 50%, và trên các cặp có độ dài gần bằng nhau thì win rate tổng thể bằng 50%. Điều này giúp tôi hiểu rằng tối ưu được loss chưa đồng nghĩa với cải thiện rõ ràng về chất lượng.

Nếu làm lại và có thêm thời gian, tôi sẽ giữ lần chạy này làm mốc, thử từng giá trị β trên cùng dữ liệu, seed và ngân sách huấn luyện. Tôi sẽ so cả reward, độ dài, câu trả lời cụ thể và kết quả giám khảo. Tôi cũng sẽ kiểm tra hiện tượng thẻ tool-call thừa trước khi diễn giải sâu các khác biệt nhỏ. Hiện tại, tôi không coi β = 0.1 là tối ưu; đây là cấu hình đã chạy và có bằng chứng để phân tích.

---

## 7. Bộ đo chuẩn

Chưa thực hiện bonus NB6. Không có kết quả IFEval, GSM8K hoặc Global-MMLU-vi, nên chưa kết luận về khả năng làm theo chỉ dẫn chuẩn hóa, giải toán hoặc “thuế căn chỉnh” sau DPO.

---

## 8. Biến thể loss

Chưa huấn luyện các biến thể ở NB3b. Các phép tính DPO, IPO, SimPO, ORPO trong NB0 chỉ minh họa công thức trên dữ liệu giả lập, không phải kết quả so sánh mô hình đã huấn luyện.

Vì chưa có thí nghiệm tương ứng, tôi chưa kết luận biến thể nào tốt hơn hoặc thay đổi độ dài nhiều nhất.

---

## 9. GRPO

Chưa thực hiện bonus NB7. Không có số liệu độ chính xác trước/sau GRPO hoặc kết quả so sánh các thành phần reward.

---

## Danh sách bonus

Không đăng ký điểm bonus trong lần nộp này.

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

Hội đồng reward model mặc định ở NB4 thuộc phần đánh giá đã chạy; tôi không tính đó là hoàn thành bonus chấm chéo bằng giám khảo API độc lập.

---

## Điều bất ngờ nhất

Reward chosen và rejected có thể cùng tăng mà DPO vẫn tăng ưu tiên tương đối cho chosen. Ngoài ra, margin held-out tăng không đảm bảo câu trả lời thực tế tốt hơn rõ ràng: trong lần chạy này, win rate tổng thể vẫn có khoảng tin cậy chứa 50%.