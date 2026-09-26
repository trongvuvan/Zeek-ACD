# Tiến trình zeek-acd

Cập nhật: 2026-09-26 (tối). Dùng file này để tiếp tục ở phiên sau.

## Cập nhật mới nhất (2026-09-26, khuya): **v8.1 = ensemble 6 seed** `dqn_g0_s{0..5}`

Thêm seed 4 (ep600) và 5 (ep350), cùng công thức v8. Ensemble 6 giống v8 trên
mọi tập benign (FP 0.00–0.01), nhỉnh hơn trên IoT-23 34-1 (RECON 0.99, C2
0.99). Chạy live: như lệnh v8 bên dưới nhưng `for s in 0 1 2 3 4 5`.

Attacker (self-play, defender đóng băng):
- Train chung có trộn dữ liệu thật (`--real-frac 0.5/0.75`, thư mục
  `joint_g0s2_rf{50,75}`): defender không hỏng trên dữ liệu thật (live_now FP
  0.00) nhưng HTTPS giữ lại FP 0.02→0.08, và attacker mới tấn công nó lại tìm
  ra đòn gây FP (+0.046). **Không dùng.**
- Attacker 1000 ep vs g0_s0 / g0_s1: chỉ +0.01–0.03. Lỗ hổng khác nhau theo
  seed, đều trên traffic tổng hợp: s0 OTHER_MAL+`pad` (0.67), s1 DOS+`spread`
  (0/10). Chưa đo được ensemble (`--defender frozen` chỉ nhận 1 checkpoint).
- Attacker 1000 ep vs g0_s4 / g0_s5: tối đa +0.039 / +0.044, đều chọn
  `spread` (C2 det 0.48–0.63 trên traffic tổng hợp). `spread` là đòn tốt nhất
  ở 3/4 seed → dữ liệu nên thêm tiếp: C2 thật dùng hạ tầng xoay vòng/fast-flux.
- Kiểm chứng thêm v8.1: log live mới 17:46–20:25 (`data/live0926b`, 13.4k
  flow) FP 0.00, det 1.00; traffic bảo trì máy chủ chưa từng thấy
  (`data/benign_ops`: dnf, git clone, pip, wget; 234 flow) FP 0.00 (v6 0.27,
  v4 0.40).
- v9 (`dqn_v9_s{0,1}`): v8.1 + log live sáng sớm 26/9 (nhãn IOC 2025) —
  không hơn v8.1 (RECON IoT 0.94, MTA-2026 OTHER_MAL 0.98). Replay lặp lại
  cùng kịch bản không thêm thông tin; cần kịch bản mới (người dùng đang tạo
  pcap). Vòng lặp tự động (`/loop`) đang chạy: kiểm tra pcap mới, train, chấm.
- Bảng đầy đủ mọi phiên bản defender/attacker: README, mục "Training versions".

## Trạng thái (2026-09-26, tối): v8 = ensemble 4 seed, `--gamma 0`

Model khuyến nghị: trung bình Q của 4 checkpoint `checkpoints/dqn_g0_s{0,1,2,3}/agent_best.pt`
(s0=ep350, s1/s2/s3=ep550; mỗi cái kèm `normalizer_best.json`). Cùng dữ liệu
với v6, khác duy nhất **`--gamma 0`**.

Chạy live (chỉ ghi log):

```bash
E=""; for s in 0 1 2 3; do E="$E --checkpoint checkpoints/dqn_g0_s$s/agent_best.pt \
  --normalizer checkpoints/dqn_g0_s$s/normalizer_best.json"; done
PYTHONPATH=src .venv/bin/python -m zeek_acd.live.run_agent $E \
  --log-path /opt/zeek/logs/current/conn.log --from-start \
  --audit-log checkpoints/v8/live_audit.jsonl
```

(`run_agent` và `eval_checkpoint` giờ nhận `--checkpoint/--normalizer` lặp
lại → ensemble, code ở `ensemble.py`.) Chạy thử đường live trên log live hôm
nay (`data/live0926`, 6.980 flow): **FP 0.000, phát hiện C2 / OTHER_MAL 1.000**.

Chấm toàn bộ tập test: `tools/test_battery.sh <dir> ... ` (`a+b+c` = ensemble,
`dir@ep550` = checkpoint cụ thể). Kết quả (action_rate):

| test | lớp | v4 | v6 | v6 seed1 | v6 seed2 | **v8** |
|---|---|---|---|---|---|---|
| benign_heldout ssl/http | BENIGN | 0.90 | 0.05 | 0.22 | 0.18 | **0.01** |
| live2026 | BENIGN | 0.16 | 0.04 | 0.37 | 0.77 | **0.00** |
| MTA-2026 | BENIGN | 0.00 | 0.00 | 0.02 | 0.10 | **0.00** |
| live0926 (live hôm nay) | BENIGN | 0.04 | 0.01 | 0.11 | 0.08 | **0.00** |
| IoT-23 3-1 | BENIGN | 0.04 | 0.48 | 0.54 | 0.66 | **0.00** |
| IoT-23 34-1 | BENIGN | 0.55 | 0.28 | 0.77 | 0.47 | **0.04** |
| live2026 / MTA-2026 / live0926 | C2, OTHER_MAL | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** (MTA C2 1.00) |
| IoT-23 34-1 | DOS / C2 / RECON | 1.00/1.00/0.96 | 1.00/1.00/0.91 | | | **1.00/0.97/0.96** |
| selfbroken_test (checksum hỏng) | BENIGN | 0.95 | 0.55 | 0.56 | 0.43 | 0.58 |

Từng seed gamma=0 riêng lẻ cũng đều FP 0.00–0.02 trên mọi tập benign sạch —
**không còn phụ thuộc may rủi seed**.

### Phát hiện 1: v6 "tốt" là do may seed

Cùng công thức v6 với seed 1 và 2: FP live2026 0.37 / 0.77 (seed 0: 0.04).
Phát hiện luôn 1.00; cái dao động là FP. Ensemble (3 seed, hoặc 4 checkpoint
cuối cùng một seed) chỉ giảm một phần (0.20–0.30).

### Phát hiện 2: `gamma=0.95` là nguồn nhiễu **và** làm lệch hành động

Chuỗi flow là ngoại sinh; hành động chỉ ảnh hưởng tương lai qua việc chặn
nguồn. Với gamma>0, target = r + γ·max Q(flow kế tiếp không liên quan) → nhiễu,
FP nhảy 1%↔86% giữa các checkpoint. Tệ hơn: chặn/cô lập kẻ tấn công **xoá các
flow sau của nó**, tức là mất phần thưởng "phát hiện" tương lai → model học
tránh chặn kẻ tấn công. Đó là lý do v4/v6 luôn chọn `DECEIVE`.

Với `--gamma 0`: `live_now` FP = 0.00 ở **mọi** checkpoint từ ep100, và hành
động là đúng cái bảng payoff đánh giá cao nhất — **`ISOLATE_HOST`** cho C2 và
OTHER_MAL. Đây là thay đổi hành vi: nếu bật enforcement, v8 sẽ cô lập host
chứ không deceive. Muốn phản ứng nhẹ hơn thì sửa bảng payoff
(`payoffs/low_fp.json`), không phải sửa model.

### Kết quả âm hôm nay

- **v7a** = v6 + traffic `192.168.6.135` bị hỏng checksum (`data/selfbroken_train`):
  sửa được traffic hỏng (0.55→0.01) nhưng FP MTA-2026 0.00→**0.52**, live2026
  0.04→0.17. Lặp lại đúng lỗi v5. **Sửa ở sensor, đừng train trên traffic hỏng.**
- **v7b** = v6 với benign_web `--repeat 8`: không sửa IoT 3-1, live0926 FP 0.36.
- Self-play `--defender frozen` với seed g0_s2: attacker ~0 phần lớn các lần
  eval, lần cuối +0.074 (v4 tối đa +0.120, v6 +0.034). Attacker **thôi dùng
  đòn gửi traffic BENIGN để gây FP** (đòn tìm ra ở phiên trước) vì không còn
  ăn. Log: `checkpoints/selfplay_g0s2frozen/train.log`.

Train lại một seed v8 (chạy 4 seed 0–3, mỗi cái ~15 phút, 2 cái song song):

```bash
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_dqn \
  --data 'data/iot23/*/conn.log.labeled' --repeat 1 \
  --data 'data/mta/2025-*/conn.log.labeled' --repeat 40 \
  --data 'data/live2025/conn.log.labeled' --repeat 20 \
  --data 'data/benign_web/train/conn.log.labeled' --repeat 15 \
  --payoff payoffs/low_fp.json --gamma 0 --seed 0 \
  --max-records-per-file 15000 --episodes 600 --eps-decay-episodes 400 \
  --eval-every 50 --checkpoint-dir checkpoints/dqn_g0_s0
# rồi chọn checkpoint bằng compare_checkpoints ... --select web-val (như v6)
```

## v6 (2026-09-26, chiều) — đã bị v8 thay

Model v6: **`checkpoints/dqn_v6/agent_best.pt`** + `normalizer_best.json`
(= `agent_ep500.pt`). Khác v4 duy nhất ở chỗ thêm dữ liệu **benign HTTPS/HTTP
sạch** (Zeek thấy trọn vẹn, `SF` + `service=ssl/http`) do `tools/gen_benign.sh`
sinh ra.

Checkpoint được **chọn trên tập validation** (`benign_web/val` + `live_now`),
rồi mới chấm trên các tập test chưa dùng để chọn (tách hẳn val/test, xử lý mục
#2 cũ). Chấm bằng `payoffs/low_fp.json` (mục #3 cũ).

Tỉ lệ ra tay (`action_rate`) — trên BENIGN là FP, trên tấn công là phát hiện:

| tập test | lớp | v4 | **v6** |
|---|---|---|---|
| `benign_heldout` (site **không trùng** tập train), chỉ ssl/http, n=375 | BENIGN | 0.90 | **0.05** |
| `benign_heldout` toàn bộ (kèm DNS), n=804 | BENIGN | 0.39 | **0.02** |
| `live2026` | BENIGN | 0.16 | **0.04** |
| `live2026` | C2 / OTHER_MAL | 1.00 / 1.00 | 1.00 / 1.00 |
| MTA-2026 | BENIGN | 0.00 | 0.00 |
| MTA-2026 | C2 / OTHER_MAL | 1.00 / 1.00 | 1.00 / 1.00 |
| IoT-23 34-1 (dữ liệu train) | DOS / C2 | 1.00 / 1.00 | 1.00 / 1.00 |

Validation (`compare_checkpoints`, low_fp): ep500 web-val FP 0.02, `live_now`
FP **0.01** phát hiện 1.00 (v4: FP 0.07).

**Con số FP 7% của v4 là quá lạc quan**: tập BENIGN cũ gần như chỉ có DNS/NTP.
Trên HTTPS bình thường ra Internet, v4 ra tay với 61–90% flow (chủ yếu
`DECEIVE`/`RATE_LIMIT`).

Điểm yếu còn lại của v6:
- IoT-23 capture 3-1: BENIGN `LOG_ALERT` 47% (v4: 3%) — chỉ là hành động nhẹ
  nhất, trên thiết bị honeypot IoT, nhưng là thoái lui thật.
- Self-play `--defender frozen` (400 ep, cùng cấu hình cho cả hai): attacker
  kiếm được ít hơn trước v6 (tối đa +0.034/bước, thường âm; v4 lên tới +0.120),
  FP trên benign tổng hợp thấp hơn (0.09–0.46 vs 0.35–0.81). Nhưng phát hiện
  **DOS tổng hợp** tụt còn 0.29–0.80 (v4 ~1.00), C2 dùng `spread` còn 0.65–1.00.
  Trên DOS thật (IoT-23) vẫn 1.00, nên đây là lệch phân phối của env tổng hợp;
  cứ theo dõi. Log: `checkpoints/selfplay_v{4,6}frozen_b/train.log`.

Train lại v6:

```bash
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_dqn \
  --data 'data/iot23/*/conn.log.labeled' --repeat 1 \
  --data 'data/mta/2025-*/conn.log.labeled' --repeat 40 \
  --data 'data/live2025/conn.log.labeled' --repeat 20 \
  --data 'data/benign_web/train/conn.log.labeled' --repeat 15 \
  --payoff payoffs/low_fp.json \
  --max-records-per-file 15000 --episodes 600 --eps-decay-episodes 400 \
  --eval-every 50 --checkpoint-dir checkpoints/dqn_v6
# chọn checkpoint CHỈ trên validation:
PYTHONPATH=src .venv/bin/python -m zeek_acd.compare_checkpoints \
  --checkpoint-dir checkpoints/dqn_v6 --payoff payoffs/low_fp.json \
  --data 'web-val=data/benign_web/val/conn.log.labeled' \
  --data 'live-now=data/live_now/conn.log.labeled' --select web-val
```

Dữ liệu benign mới (sinh từ chính máy này, `192.168.6.135`):

```bash
cd data/benign_web
tcpdump -i ens160 -s 0 -w benign_web.pcap \
  'host 192.168.6.135 and (tcp port 80 or tcp port 443 or udp port 53 or udp port 443)' &
../../tools/gen_benign.sh 1500          # tập test: thêm ../../tools/sites_heldout.txt
kill -INT %1
cd ../..
PYTHONPATH=src .venv/bin/python -m zeek_acd.pcap_dataset \
  --pcap data/benign_web/benign_web.pcap --ignore-checksums \
  --unmatched benign --benign-src 192.168.6.135 \
  --out-file data/benign_web/all.conn.log.labeled
# rồi tách theo thời gian: 75% đầu -> train/, 25% cuối -> val/
```

`benign_web`: 1.095 flow (train 821 / val 274), `benign_heldout`: 804 flow,
0 flow khớp IOC. Gồm: duyệt ~50 site phổ biến, tải file lớn, và **poll định kỳ
~30s** (kiểu kiểm tra kết nối/cập nhật) — benign nhưng có nhịp như beacon C2.

### Phát hiện: "mất gói" của `192.168.6.135` thật ra là **checksum offload**

Zeek live trên `ens160` thấy traffic của chính máy này là `OTH`/`SHR`/`RSTO`
không có service. Cùng traffic đó bắt bằng tcpdump rồi `zeek -C -r` thì ra
`SF`/`ssl` bình thường. Gói *đi ra* từ máy này bị bắt trước khi NIC điền
checksum, Zeek coi là checksum sai và **bỏ** (chỉ còn thấy chiều trả lời).
Đây là nguyên nhân thật của kết quả âm v5 bên dưới, không phải mạng mất gói.

Sửa ở sensor (chưa làm — là cấu hình của người dùng): thêm
`redef ignore_checksums = T;` vào `/usr/local/zeek/share/zeek/site/local.zeek`
rồi `zeekctl deploy`, hoặc `ethtool -K ens160 tx off rx off`. Sau khi sửa thì
log live của `192.168.6.135` dùng được làm benign thật.

## Trạng thái cũ (2026-09-24): v4

Model trước đó: **`checkpoints/dqn_v4/agent_best.pt`** + `normalizer_best.json`
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
| `checkpoints/dqn_v4/` | bản trước (`agent_best.pt` = ep500); FP cao trên HTTPS | thay bằng v6 |
| `checkpoints/dqn_v6/` | v6 (`agent_best.pt` = ep500), seed may mắn | thay bằng v8 |
| `checkpoints/dqn_v6_s{1,2}/` | v6 với seed khác — đo độ dao động | không |
| `checkpoints/dqn_v7{a,b}/` | thử nghiệm âm (xem trên) | không |
| **`checkpoints/dqn_g0_s{0,1,2,3}/`** | **v8: ensemble 4 seed, gamma=0** | **có** |
| `checkpoints/v8/` | audit chạy live thử của v8 | |
| `checkpoints/selfplay_v{4,6}frozen_b/` | đo độ dễ bị khai thác của v4/v6, cùng cấu hình | chỉ để chẩn đoán |
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
| `data/live_now/` | toàn bộ log live hiện tại | **validation** |
| `data/benign_web/{train,val}/` | benign HTTPS sạch, tách theo thời gian 75/25 | train / **validation** |
| `data/benign_heldout/` | benign HTTPS sạch, site khác hẳn (`tools/sites_heldout.txt`) | **test** |
| `data/benign_heldout_web/` | như trên, chỉ ssl/http | **test** |
| `data/live0926/` | log live 2026-09-26 16:33–17:46, nhãn theo IOC, bỏ external không khớp | **test** |
| `data/live0926b/` | log live 2026-09-26 17:46–20:25 | **test** |
| `data/benign_ops/` | traffic bảo trì máy chủ (dnf/git/pip/wget) từ `192.168.6.135` | **test** |
| `data/selfbroken_{train,test}/` | traffic web của `192.168.6.135` bị hỏng checksum, tách tại 16:57 | chỉ thử nghiệm |

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
được. **(2026-09-26: nguyên nhân thật là checksum offload, xem phần đầu —
không phải mất gói.)** Model dịch khái niệm "sạch" về phía kết nối đứt đoạn rồi quay ra nghi ngờ
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

1. Sửa checksum ở sensor live (`redef ignore_checksums = T;` hoặc
   `ethtool -K ens160 tx off`) — việc của người dùng. Sau đó chạy v8 live
   (lệnh ở đầu file) trên traffic thật của `192.168.6.135`.
2. Quyết định bảng payoff: v8 dùng `ISOLATE_HOST` cho mọi tấn công. Nếu muốn
   phản ứng nhẹ hơn (DECEIVE/RATE_LIMIT) khi bật enforcement thì sửa bảng.
3. Env có động lực sai khi gamma>0 (chặn kẻ tấn công = mất thưởng tương lai).
   Nếu muốn quay lại học nhiều bước (gamma>0), phải sửa reward: ví dụ thưởng
   cho mỗi flow độc hại bị chặn *trước khi xảy ra* bằng đúng phần thưởng phát
   hiện, để chặn sớm không bị phạt.
4. Benign vẫn chỉ từ curl trên một máy Linux. Thêm trình duyệt thật / Windows.
5. Chưa bật enforcement thật. v8 FP 0–1% trên benign sạch; nên chạy live ở
   chế độ chỉ ghi log vài ngày trước.

## Trạng thái git

2026-09-26: chưa commit — `PROGRESS.md`, `src/zeek_acd/pcap_dataset.py`
(thêm `--pcap`, `--ignore-checksums`), `src/zeek_acd/eval_checkpoint.py` và
`src/zeek_acd/live/run_agent.py` (ensemble), file mới `src/zeek_acd/ensemble.py`,
`tools/gen_benign.sh`, `tools/sites_heldout.txt`, `tools/test_battery.sh`.

Trước đó:

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
| `ensemble.py` | trung bình Q của nhiều checkpoint DQN (dùng bởi `eval_checkpoint`, `run_agent`) |
