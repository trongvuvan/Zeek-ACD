# Tiến trình zeek-acd

Cập nhật: 2026-09-23. Dùng file này để tiếp tục ở phiên sau.

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

## Việc tiếp theo

1. Tập BENIGN hiện gần như chỉ có DNS/NTP tới resolver nội bộ; **chưa có HTTPS
   bình thường ra Internet**. Cần thêm capture duyệt web sạch rồi đo lại FP —
   đây là giới hạn lớn nhất còn lại của con số FP 7%.
2. Chọn checkpoint đang dựa trên chính các tập giữ lại (ep500 chọn theo
   `live_now`). Có thêm dữ liệu thì nên tách hẳn validation và test.
3. `dqn_v4` đang train bằng `payoffs/low_fp.json` nhưng chấm điểm bằng bảng mặc
   định (để so được giữa các bản). Nếu chốt dùng bảng low_fp thì nên chấm bằng
   chính nó.
4. Chưa thử `train_selfplay.py` với defender DQN đã tốt — đo xem attacker học
   được cách né tới đâu.
5. Chưa bật enforcement thật (`--executor nftables --live-enforce`). Với FP 7%
   trên BENIGN thì **chưa nên bật**.
