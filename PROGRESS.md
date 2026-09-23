# Tiến trình zeek-acd

Cập nhật: 2026-09-24. Dùng file này để tiếp tục ở phiên sau.

## Trạng thái hiện tại

Model dùng được: **`checkpoints/dqn_v4/agent_best.pt`** + `normalizer_best.json`
(chính là `agent_ep500.pt`). Kiểm chứng trên log Zeek **live** (3.604 flow có
nhãn, bắt trên card mạng chứ không phải `zeek -r`):

| lớp | n | action_rate | hành động |
|---|---|---|---|
| BENIGN | 1601 | **0.071** | `ALLOW` 0.929, `LOG_ALERT` 0.065 |
| C2 | 711 | **1.000** | `DECEIVE` 0.997 |
| OTHER_MALICIOUS | 1292 | **1.000** | `DECEIVE` 0.999 |

Trên tập giữ lại hoàn toàn (4 kịch bản `2026-*` chưa từng vào train):
`avg_reward=+0.205`, FP `0.00`, phát hiện `1.00` — so với baseline tốt nhất
(`always-LOG_ALERT`) là `+0.078`.

Chạy live (chỉ ghi log, không chặn gì):

```bash
cd /root/zeek-acd
PYTHONPATH=src .venv/bin/python -m zeek_acd.live.run_agent \
  --checkpoint checkpoints/dqn_v4/agent_best.pt \
  --normalizer checkpoints/dqn_v4/normalizer_best.json \
  --log-path /opt/zeek/logs/current/conn.log --from-start \
  --audit-log checkpoints/dqn_v4/live_audit.jsonl
```

Train lại đúng cấu hình đó:

```bash
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_dqn \
  --data 'data/iot23/*/conn.log.labeled' --repeat 1 \
  --data 'data/mta/2025-*/conn.log.labeled' --repeat 40 \
  --data 'data/live2025/conn.log.labeled' --repeat 20 \
  --payoff payoffs/low_fp.json \
  --max-records-per-file 15000 --episodes 600 --eps-decay-episodes 400 \
  --eval-every 100 --checkpoint-dir checkpoints/dqn_v4
```

### Kiểm tra model

```bash
# chấm trên một tập có nhãn (flat = mỗi flow một lần, không suppression)
PYTHONPATH=src .venv/bin/python -m zeek_acd.eval_checkpoint \
  --checkpoint checkpoints/dqn_v4/agent_best.pt \
  --normalizer checkpoints/dqn_v4/normalizer_best.json \
  --data 'data/mta/2026-*/conn.log.labeled' --baselines

# so mọi checkpoint của một lần train trên nhiều tập cùng lúc
PYTHONPATH=src .venv/bin/python -m zeek_acd.compare_checkpoints \
  --checkpoint-dir checkpoints/dqn_v4 \
  --data 'MTA-2026=data/mta/2026-*/conn.log.labeled' \
  --data 'live-now=data/live_now/conn.log.labeled' --select live-now

# đọc lại nhật ký một lần chạy live, đối chiếu với nhãn
PYTHONPATH=src .venv/bin/python -m zeek_acd.audit_report \
  --audit-log checkpoints/dqn_v4/live_audit.jsonl \
  --labels 'data/live_now/conn.log.labeled' --show 5 --only-wrong
```

`compare_checkpoints.py` tồn tại vì eval lúc train dao động rất mạnh — trong
cùng một lần train 600 episode, FP nhảy từ 1% đến 96% mà `avg_reward` gần như
không đổi. **Đừng mặc định lấy checkpoint cuối.**

### Các thư mục checkpoint

| thư mục | là gì | dùng được? |
|---|---|---|
| `checkpoints/iot23/` | minimax-DQN, phiên đầu | không (= baseline LOG_ALERT) |
| `checkpoints/dqn_iot23/` | DQN chỉ IoT-23, feature cũ 60 chiều | không (không tổng quát) |
| `checkpoints/dqn_mixed/`, `dqn_v2/`, `dqn_v3/` | các bước trung gian | không |
| **`checkpoints/dqn_v4/`** | **bản khuyến nghị (`agent_best.pt` = ep500)** | **có** |
| `checkpoints/dqn_v5/` | thêm benign mất gói | không — xem "Kết quả âm" |
| `checkpoints/selfplay_v4frozen/` | attacker học đấu v4 đóng băng | chỉ để chẩn đoán |
| `checkpoints/selfplay_joint/` | train chung | không — hỏng từ ep250 |

Lưu ý: checkpoint trước `dqn_v3` dùng feature cũ (60 hoặc 72 chiều, ngữ cảnh
tuyệt đối) nên **không nạp được** bằng code hiện tại (71 chiều).

## Bốn vấn đề đã tìm ra và cách sửa

### 1. Minimax-DQN chỉ học thuộc bảng payoff, không học được state → lớp

Checkpoint minimax (`checkpoints/iot23/`) cho `avg_reward=0.120`, **giống hệt**
baseline "luôn `LOG_ALERT`". Không phải do cách rút hành động: đổi LP sang
`argmax_a E_class[Q]` chỉ chuyển thành "luôn `BLOCK_SRC`".

Bằng chứng: trên 2.000 state eval, hành động tốt nhất theo *từng cột lớp* giống
nhau ở **mọi** state (BENIGN→ALLOW 1999/2000, RECON→DECEIVE 2000/2000,
DOS→BLOCK_SRC, C2→ISOLATE_HOST). Lý do có tính cấu trúc: ô `Q(s, a_d, c)` chỉ
được cập nhật ở đúng cột `c` thật sự xảy ra, nên target ≈ `payoff(a_d, c) +
γ·V(s')`; phần phụ thuộc state (`V`) cộng đều vào cả cột nên không đổi thứ hạng.
Mạng không bao giờ *cần* phân biệt lớp.

→ **Sửa**: `train_dqn.py`, DQN thường không có trục lớp. `Q(s, a_d)` học trực
tiếp từ reward thật, nên muốn điểm cao buộc phải suy ra lớp từ feature.

### 2. Model train trên IoT-23 không phát hiện được mã độc HTTP/TLS

Chấm model IoT-23 trên dữ liệu mã độc mới: `avg_reward=-0.548`, `ALLOW` gần như
toàn bộ — tệ hơn cả baseline `always-LOG_ALERT` (+0.090). Mã độc IoT-23 là
scan/DoS của botnet (`conn_state=S0/REJ`, không service); mã độc ở đây là phiên
TLS/HTTP hoàn chỉnh (`SF`, `ShADadFf`), nhìn y hệt traffic bình thường của
IoT-23.

→ **Sửa**: dựng dataset có nhãn từ `/root/pcaps` + `/root/indicators`
(`iocs.py` + `pcap_dataset.py`) rồi trộn vào tập train.

### 3. Đặc trưng ngữ cảnh tuyệt đối trôi theo lượng traffic

Thêm `context.py` (đặc trưng liên-flow) giúp phát hiện C2 rất tốt trên pcap
(MTA-2026 giữ lại: phát hiện 100%, FP 8%), nhưng **chạy live thì hỏng**: FP trên
BENIGN 77%, phát hiện tụt còn 70–73%.

Đã loại trừ nguyên nhân pipeline (so từng trường và từng đặc trưng trên cùng 400
uid giữa đường offline và đường live — khớp nhau) và loại trừ mất gói
(capture loss 0,56%). Nguyên nhân thật: trong pcap mỗi kịch bản là một capture
riêng nên `ctx_src_conns_long` trung vị = 10; trên dây thật mọi kịch bản chạy
chung một host nên = 316 và **còn tăng theo thời gian capture**. FP nhảy từ 8%
lên 78% chỉ vì log live dài thêm.

→ **Sửa**: toàn bộ ngữ cảnh chuyển sang **đại lượng tương đối** — tỉ lệ traffic
của nguồn trên toàn mạng, fan-out = số đích/số kết nối, tỉ trọng của đích trên
tổng traffic. Bỏ hẳn mọi bộ đếm tuyệt đối (scan vẫn lộ qua fan-out và tỉ lệ bắt
tay hỏng).

### 4. Bảng payoff mặc định gần như bàng quan với false positive

`LOG_ALERT` trên BENIGN chỉ tốn −0.05, trong khi đoán đúng tấn công được
+0.2…+0.4 → "cảnh báo tất cả" gần bằng "đúng chỗ". FP trôi từ 1% đến 96% giữa
các checkpoint mà `avg_reward` gần như không đổi.

→ **Sửa**: `--payoff` (có ở `train.py`, `train_dqn.py`, `eval_checkpoint.py`) và
bảng `payoffs/low_fp.json`: mọi hành động trên traffic sạch đắt gấp ~4 lần, phần
thưởng phát hiện giữ nguyên. FP ổn định 0.01–0.06 suốt ep200→500.

## Dữ liệu

| thư mục | nguồn | dùng làm gì |
|---|---|---|
| `data/iot23/` | 9 kịch bản IoT-23 (`conn.log.labeled`) | train |
| `data/mta/2025-*/` | 19 pcap `/root/pcaps` + IOC, qua `zeek -r` | train |
| `data/mta/2026-*/` | 4 pcap còn lại | **test, chưa từng train** |
| `data/live2025/` | log live, chỉ giữ flow khớp IOC `2025-*` | train |
| `data/live2026/` | log live, chỉ giữ flow khớp IOC `2026-*` | **test** |
| `data/live_now/` | toàn bộ log live hiện tại | kiểm chứng |

Dựng lại:

```bash
PYTHONPATH=src .venv/bin/python -m zeek_acd.pcap_dataset --out-dir data/mta
PYTHONPATH=src .venv/bin/python -m zeek_acd.pcap_dataset \
  --zeek-log-dir /opt/zeek/logs/current --ioc-glob '2025-*.txt' \
  --out-file data/live2025/conn.log.labeled --unmatched drop
```

Gán nhãn theo 3 bước: IP đích khớp indicator → SNI/HTTP Host theo `uid` → IP mà
DNS trong chính capture đó trả về cho domain indicator. Trên 23 pcap chỉ có
**1/403 flow ngoài mạng không khớp** indicator nào.

## Kết quả âm đã thử (đừng lặp lại)

### v5: thêm traffic benign của máy thật làm model **tệ đi**

Log Zeek đã xoay vòng có 222 kết nối của máy thật `192.168.6.135`, trong đó 323
flow ra cổng 443 — đúng loại benign HTTPS mà tập train đang thiếu. Đã thêm vào
train (`data/rot2025_*`, `--benign-src 192.168.6.135`).

Kết quả: FP trên tập giữ lại sạch **MTA-2026 tăng từ 0% lên 75%** (v4 ep500 so
với v5 ep600), chỉ đổi lấy 0.04 vs 0.12 trên `live-now`. Phát hiện vẫn 1.00.

Lý do: traffic đó có `conn_state` là `SHR`/`OTH`/`RSTO` — mạng `192.168.6.135`
bị mất gói nên Zeek chỉ thấy một phần mỗi kết nối, và `service` không xác định
được. Model dịch khái niệm "sạch" về phía kết nối đứt đoạn rồi quay ra nghi ngờ
các phiên hoàn chỉnh. **Dữ liệu mất gói không thay được dữ liệu benign sạch** —
muốn đóng lỗ hổng này phải có capture duyệt web bình thường mà Zeek thấy trọn
vẹn (`conn_state=SF`).

### Self-play: hữu ích để chẩn đoán, **phản tác dụng** khi dùng để train

Đã cho attacker khả năng né thật (`selfplay.py`: hành động = lớp × chế độ né,
5×5=25; env tự duy trì `ConnContext` trên dòng nó phát ra; episode kết thúc theo
cả số bước lẫn thời gian mô phỏng nên đi chậm phải trả giá). Đo trên 120 bước:
`none` cho beacon đều tăm tắp (`ctx_pair_iat_cv`=0.017), `jitter` phá sạch tính
đều (→0.532) mà vẫn lọt 115/120, `spread` hạ `ctx_pair_share` 0.614→0.014 mà
không mất gì, `slow` chỉ lọt 10/120.

**Đo độ dễ bị khai thác** (defender v4 đóng băng, attacker học 400 episode):
attacker chỉ đạt `+0.035`, bị chặn 80–97%, phát hiện C2/DOS vẫn 1.00. Nhưng
**toàn bộ lợi nhuận của nó đến từ false positive, không phải né tránh**: ở
ep150 nó chọn `BENIGN` 96% — không tấn công gì — mà vẫn ăn `+0.047` vì defender
ra tay với 73% traffic sạch. Đây là một đòn tấn công thật: tạo traffic vô hại
có hình dạng khiến IDS chặn nhầm, biến defender thành công cụ từ chối dịch vụ.

**Train chung** (`--defender vanilla --defender-init` v4): ep50–200 giữ nguyên
chất lượng v4 (self-play chưa kịp đổi gì), nhưng từ ep250 khi defender thật sự
học từ dòng tổng hợp thì **hỏng trên traffic thật** — FP trên `live-now` vọt
lên 0.87–0.96, có lúc phát hiện tụt còn 0.74.

Lý do: môi trường self-play phát traffic tổng hợp (một nguồn cố định beacon tới
một đích cố định) lệch quá xa phân phối thật; defender khớp vào môi trường giả.
Muốn dùng self-play để train thì phải **xen kẽ episode dữ liệu thật** với
episode self-play, hoặc cho env phát traffic theo đúng thành phần dòng thật.

## Việc tiếp theo

1. Tập BENIGN hiện gần như chỉ có DNS/NTP tới resolver nội bộ; **chưa có HTTPS
   bình thường ra Internet mà Zeek thấy trọn vẹn**. Cần capture duyệt web sạch
   (`conn_state=SF`, có `service=ssl`) rồi đo lại FP — đây là giới hạn lớn nhất
   còn lại của con số FP 7%. Xem phần "Kết quả âm" ở trên: lấy tạm traffic bị
   mất gói thì phản tác dụng.
2. Chọn checkpoint đang dựa trên chính các tập giữ lại (ep500 chọn theo
   `live_now`). Có thêm dữ liệu thì nên tách hẳn validation và test.
3. `dqn_v4` đang train bằng `payoffs/low_fp.json` nhưng chấm điểm bằng bảng mặc
   định (để so được giữa các bản). Nếu chốt dùng bảng low_fp thì nên chấm bằng
   chính nó.
4. Nếu quay lại self-play: xen kẽ episode dữ liệu thật với episode self-play
   (xem phần "Kết quả âm"). Riêng phép đo độ dễ bị khai thác
   (`--defender frozen`) thì dùng được ngay và nên chạy lại sau mỗi lần train.
5. Chưa bật enforcement thật (`--executor nftables --live-enforce`). Với FP 7%
   trên BENIGN thì **chưa nên bật**.

## Trạng thái git

Commit gần nhất là `887ed78` (do người dùng tự commit). Chưa commit:
`PROGRESS.md`, `src/zeek_acd/pcap_dataset.py`, `src/zeek_acd/selfplay.py`,
`src/zeek_acd/train_selfplay.py`, và file mới `src/zeek_acd/compare_checkpoints.py`.
`data/` và `checkpoints/` đã nằm trong `.gitignore`.

## Module trong `src/zeek_acd/` (những cái thêm trong các phiên này)

| file | việc |
|---|---|
| `context.py` | đặc trưng liên-flow, **toàn bộ là tỉ lệ/tỉ trọng**, dùng chung offline + live |
| `iocs.py` | đọc file IOC defang → tập chỉ báo khớp được |
| `pcap_dataset.py` | pcap hoặc thư mục log Zeek (TSV/JSON/`.gz`) → dữ liệu có nhãn |
| `dqn.py`, `train_dqn.py` | DQN thường — bản dùng được trên dữ liệu thật |
| `eval_checkpoint.py` | chấm một checkpoint trên một tập |
| `compare_checkpoints.py` | chấm mọi checkpoint của một lần train trên nhiều tập |
| `audit_report.py` | đọc lại nhật ký chạy live, đối chiếu nhãn |
| `selfplay.py`, `train_selfplay.py` | attacker có đòn né; `--defender frozen` để đo độ dễ bị khai thác |
