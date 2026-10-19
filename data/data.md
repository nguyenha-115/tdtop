# Đặc tả sinh dữ liệu cho TD-TOP đa xe tải (MILP đa mục tiêu + các thuật toán base)

## 1. Các bộ dữ liệu cần sinh

| Bộ | Mục đích | Số nút thông tin `n` | Kiểu phân bố | Số instance lõi | T (phút) |
|---|---|---|---|---|---|
| **S** (small) | | 10, 20, 30 | R (hoặc giữ nguyên từ Morandi) | 3 × 5 = 15 | 200 
| **L** (large) | | 100, 200, …, 1000 | R, C, RC | 10 × 3 × 3 seed = 90 | 400 
| **SA-R** (nhạy cảm số UAV) | Ảnh hưởng của R | 200, 300, …, 1000 (9 instance) | xen kẽ R/C/RC | 9 | 400 
| **SA-θ** (nhạy cảm pin) | Ảnh hưởng của θ | ≈ 300 (9 instance: 3 kiểu × 3 seed) | R, C, RC | 9 | 400 

-  --count mặc định của từng bộ: S = 5, L = 5, SA-R = 1, SA-B = 3. 

**Đội xe**:
đặt một hằng số năng lực $C$ (số lượng nút tối đa mà 1 đội xe tải - UAV có thể phục vụ hiệu quả trong giới hạn thời gian $T$).

Công thức: $K = \lceil N / C \rceil$ (trong đó $N$ là tổng số nút của instance)

## 2. Tham số cố định - DEFAULT_CONFIG

**Đơn vị:** khoảng cách km; thời gian phút; công suất W; năng lượng Wh. Chuyển đổi: `t[phút] = d[km] / v[km/h] × 60`; `E[Wh] = P[W] × t[phút] / 60`.

#### Di chuyển

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| V_TRUCK | 50 km/h | tốc độ xe tải |
| V_UAV | 72 km/h (= 20 m/s) | tốc độ bay hành trình của UAV |
| Khoảng cách xe tải | Manhattan | dùng để tính thời gian xe chạy |
| Khoảng cách UAV | Euclid | dùng để tính thời gian và năng lượng bay |

#### Mô hình công suất UAV (cánh quạt, Zeng et al. 2019)
| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| P0 | 79,86 W | công suất cản của cánh quạt |
| Pi | 88,66 W | công suất cảm ứng khi bay lơ lửng |
| U_TIP | 120 m/s | tốc độ đầu cánh quạt |
| V0 | 4,03 m/s | vận tốc cảm ứng khi treo |
| D0 | 0,6 | hệ số cản của thân UAV |
| RHO| 1,225 kg/m³ | 	mật độ không khí |
| Q_SOLIDITY | 	0,05 | 	độ đặc của cánh quạt |
| A_DISK | 0,503 m² | diện tích đĩa quay |
| P_SENS | 5 W | công suất cảm biến |
| P_COMP | 8 W | công suất tính toán trên UAV |
| P_TX | 2 W | công suất truyền dữ liệu |

#### Pin

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| BATTERY_MAH_DEFAULT | 4276 mAh | dung lượng pin mặc định (cũng dùng để quyết định hậu tố tên file) |
| BATTERY_VOLT (hay U_pin) | 14,76 V | điện áp để đổi mAh sang Wh |
| BATTERY_RESERVE | 0,0 | hệ số dự trữ an toàn; θ được nhân với (1 − hệ số). Bằng 0 nghĩa là dùng toàn bộ pin |

- bài báo cho dung lượng theo mAh (DJI Air 3: 4276 mAh). Mô hình cần năng lượng nên quy đổi `θ[Wh] = mAh/1000 × BATTERY_VOLT`, với `BATTERY_VOLT = 14,76 V`:

| Dung lượng (mAh) | 1900 | 2850 | 4276 (mặc định) | 5100 |
|---|---|---|---|---|
| θ (Wh) | 28,044 | 42,066 | 63,1138 | 75,276 |
| Tầm bay thuần túy tối đa (km) | 	11,32 | 16,99 | 25,49 | 30,40 |

- Tầm bay thuần túy = θ / E_FLY_WH_PER_KM, chưa tính thời gian phục vụ và lơ lửng, nên tầm thực tế ngắn hơn.

#### Đội xe:

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| CAPACITY | 100 | năng lực C: số nút tối đa một đội xe tải - UAV phục vụ hiệu quả; số xe tải K = ceil(n / C) |
| DEFAULT_R | 3 | số UAV mỗi xe tải khi không chỉ định --R |
| SORTIES_PER_UAV (bộ S) | 2 | s_max: số sortie tối đa mỗi UAV |
| SORTIES_PER_UAV(L, SA-R, SA-B) | null | heuristic tự quyết số sortie |

#### Giá trị thông tin và thời gian phục vụ:

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| VALUE_RANGE | [1, 100] | khoảng của giá trị thông tin p (số nguyên) |
| VALUE_NORMAL | trung bình 50, độ lệch chuẩn 20 | tham số khi dùng `--value_dist normal` (chuẩn cắt cụt trong VALUE_RANGE) |
| SERVICE_RANGE_MIN | [0, 3] phút | khoảng của thời gian phục vụ s_min (phân bố đều liên tục, làm tròn 4 chữ số) |

#### Sinh tọa độ:

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| GRID | 1000 | lưới nguyên [0, 1000]²; km = lưới × (cạnh vùng / 1000) |
| CLUSTER_DECAY | 40 | hằng số suy giảm của xác suất chấp nhận quanh hạt giống cụm (đơn vị lưới; ≈ 1 km ở vùng 25 km, ≈ 0,4 km ở vùng 10 km) |
| CLUSTER_RANGE | [3, 8] | số cụm S chọn ngẫu nhiên đều trong khoảng này |

#### Nút đường bộ:

| Tham số | Giá trị | Ý nghĩa | 
|---|---|---|
| MIN_ROAD_DIST | 0,1 km | khoảng cách tối thiểu giữa hai nút đường bộ, và giữa nút đường bộ với kho |
| ROAD_NOISE_RETRIES | 100 | số lần lấy lại nhiễu tối đa cho mỗi nút khi vi phạm khoảng cách tối thiểu (hết lần thì nhận lần cuối) |
| ROAD_SIGMA (S) | 0,05 km | độ lệch chuẩn nhiễu Gauss áp lên vị trí nút đường bộ |
| ROAD_SIGMA (L, SA) | 0,2 km | độ lệch chuẩn nhiễu Gauss áp lên vị trí nút đường bộ |
| ROAD_NODES_PER_TRUCK (S) | 4 | số nút đường bộ mỗi xe, cố định |
| ROAD_NODES_DIV, ROAD_NODES_RANGE (L, SA) | 25; [5, 15] | số nút đường bộ mỗi xe m = clip(ceil(n / 25), 5, 15) |

#### Thông số theo bộ (PROFILES):

| Thông số | S | L, SA-R, SA-B | 
|---|---|---|
| MAX_COORD (cạnh vùng) | 10 km | 25 km |
| T_MAX_MIN (thời hạn nhiệm vụ) | 200 phút | 400 phút |
| Kho | tâm vùng (5, 5) | tâm vùng (12,5; 12,5) |

#### Đại lượng dẫn xuất (tính từ tham số, ghi vào mục Energy):

| Đại lượng | Công thức | Giá trị | Ý nghĩa |
|---|---|---|---|
| P_AVIO_W (P_AVIO) | P_SENS + P_COMP + P_TX | 15,00 W | công suất thiết bị thu thập
| P_FLY_W | công thức công suất bay tại 20 m/s | 178,3063 W | công suất bay hành trình
| P_HOVER_W | Pi + P0 | 168,52 W | công suất lơ lửng (dùng cho năng lượng chờ)
| P_SRV_W | P_AVIO + P_HOVER | 183,52 W | công suất khi phục vụ một nút
| E_FLY_WH_PER_KM | P_FLY / V_UAV | 2,4765 Wh/km | năng lượng bay mỗi km

Năng lượng phục vụ một nút = `P_SRV_W × s_min / 60`: trung bình (1,5 phút) ≈ 4,59 Wh, tối đa (3 phút) ≈ 9,18 Wh. Thời gian lơ lửng tối đa với pin 4276 mAh ≈ 22,5 phút (63,1138 Wh / 168,52 W).

---

## 3. Cách sinh từng thành phần

#### Nút thông tin

| Kiểu | Cách sinh |
|---|---|
| R | n điểm nguyên độc lập, đều trên lưới [0, 1000]². |
| C | Chọn `S ∈ {3,…,8}` hạt giống cụm (chính là S nút đầu). Các nút còn lại sinh bằng rejection sampling: lấy điểm ngẫu nhiên, chấp nhận với xác suất `min(1, Σ exp(−d / 40))` (d = khoảng cách tới từng hạt giống, đơn vị lưới). |
| RC | Nửa đầu (n // 2) kiểu C, phần còn lại kiểu R. |

Điều kiện: các nút có tọa độ **khác nhau đôi một** (nếu trùng, lấy lại điểm đó).

#### Giá trị thông tin `p_i` và thời gian phục vụ `s_min`

- `p`: mặc định phân bố đều rời rạc trên {1, …, 100}, kỳ vọng ≈ 50,5 nên TOTAL_VALUE ≈ 50,5 × n. Tùy chọn normal: chuẩn (50, 20), bỏ mẫu ngoài [1, 100].
- `s_min`: đều liên tục trên [0, 3] phút, kỳ vọng 1,5 phút.

#### Kho

- Tọa độ kho: **tâm vùng** `(max_coord/2, max_coord/2)`.
- Mô hình đa xe cần hai bản sao: `0` (xuất phát) và `0'` (kết thúc), **cùng tọa độ**. Tất cả K xe xuất phát từ `0` và về `0'`.
- Số nút đường bộ: `|N_1| = K × m`, với `m` là số nút đường bộ trên mỗi xe.

#### Số xe và số nút đường bộ theo n (mặc định C = 100)

| n | K | m (mỗi xe) | Tổng nút đường bộ |
|---|---|---|---|
| 10, 20, 30 (bộ S) | 1 | 4 | 4
| 100 | 1 | 5 | 5
| 200 | 2 | 8 | 16
| 300 | 3 | 12 | 36
| 400 | 4 | 15 | 60
| 500 | 5 | 15 | 75
| 600 | 6 | 15 | 90
| 700 | 7 | 15 | 105
| 800 | 8 | 15 | 120
| 900 | 9 | 15 | 135
| 1000 | 10 | 15 | 150

#### Seed và luồng ngẫu nhiên

Instance thứ `k` dùng `seed_i = seed_gốc + k` (mặc định seed gốc 42). Mỗi thành phần có luồng riêng:

| Thành phần | Seed | Phụ thuộc | 
|---|---|---|
| Tọa độ nút thông tin (R, hoặc nửa C của RC) | seed_i | n, kiểu
| Nửa R của RC | seed_i + 500 | n
| Giá trị thông tin p | seed_i + 1000 | n, kiểu phân bố giá trị |
| Thời gian phục vụ s_min | seed_i + 2000 | n |
| Nhiễu nút đường bộ | seed_i + 3000 | lõi instance, số nút đường bộ 
| Số cụm và hạt giống cụm | seed_i + 4000 | n, kiểu

Hệ quả: thay R, pin, C không làm đổi tọa độ, p, s_min; tăng K chỉ thêm nút đường bộ ở cuối danh sách.

## 5. Cấu trúc file đầu ra

Mỗi instance là một file JSON: `data/<bộ>/N<n>/S<seed>_N<n>_<kiểu>_R<R>.json`, ví dụ data/L/N300/S042_N300_RC_R3.json. Với mức pin khác 4276 mAh thêm hậu tố _B<mAh>, ví dụ S042_N300_RC_R3_B1900.json.

#### Config.General

| Trường | Ý nghĩa |
|---|---|
| `SUITE` | `bộ dữ liệu: S, L, SA-R, SA-B` |
| `NUM_INFO_NODES` | số nút thông tin n |
| `NUM_ROAD_NODES` | số nút đường bộ (không tính kho) |
| `MAX_COORD` | cạnh vùng (10 hoặc 25) |
| `T_MAX_MIN` | thời hạn nhiệm vụ `T` (phút) |
| `COORD_DISTRIBUTION` | kiểu phân bố `R, C hoặc RC` |
| `NUM_CLUSTERS` | 	số cụm |
| `SOURCE` | luôn là `synthetic` |
| `SEED`, `SEED_BASE` | seed của instance và seed gốc |
| `VALUE_DIST` | phân bố giá trị thông tin `uniform` hoặc `normal` |
| `CAPACITY` | năng lực C đã dùng |
| `ROAD_NODES_PER_TRUCK` | số nút đường bộ mỗi xe `m` |

#### Config.Vehicles

| Trường | Ý nghĩa |
|---|---|
| `NUM_TRUCKS` | số xe tải `K` |
| `UAVS_PER_TRUCK` | danh sách K phần tử, mỗi phần tử là `R` |
| `SORTIES_PER_TRUCK` | `|Q_k|` (dùng cho MILP) |
| `V_TRUCK`, `V_UAV` | tốc độ xe tải, UAV |
| `TRUCK_DISTANCE`, `UAV_DISTANCE` | `manhattan`, `euclid` |
| `BATTERY_MAH`, `BATTERY_VOLT`, `THETA_WH` | pin (mAh), điện áp, và dung lượng quy đổi θ (Wh) |

#### Config.Energy

| Trường | Ý nghĩa |
|---|---|
| `P0`, `Pi`, `U_TIP`, `V0`, `D0`, `RHO`, `Q_SOLIDITY`, `A_DISK` | tham số công thức (1) |
| `P_SENS`, `P_COMP`, `P_TX`, `P_AVIO` | công suất thu thập |
| `P_FLY_W` | `P^fly(v^U)` đã tính |
| `P_HOVER_W` | `P_i + P_0` |
| `P_SRV_W` | `P_avio + P^hover` |
| `E_FLY_WH_PER_KM` | năng lượng bay mỗi km |

#### Node

| Khối | Trường mỗi phần tử |
|---|---|
| `Depot` | `id_start = 0`, `id_end = 0'`, `coord` |
| `RoadNodes` | `id`, `coord`: node đường bộ |
| `InfoNodes` | `id`, `coord`, `p`, `s_min`: node thông tin |

**Quy ước chỉ số:** `0` = kho xuất phát; `1…|N_1|` = nút đường bộ; `|N_1|+1` = kho kết thúc `0'`; `|N_1|+2 … |N_1|+n+1` = nút thông tin.

#### Stats và manifest

- `Stats.TOTAL_VALUE`: tổng giá trị thông tin `Σ p`, cũng là cận trên của mục tiêu `f₁`.

- `manifest.csv`: mỗi dòng một file, gồm file, suite, n, dist, K, R, mAh, theta_wh, T_min, seed, source, TOTAL_VALUE, sha256 (16 ký tự đầu của mã băm nội dung). Chạy lại sẽ cập nhật các dòng đã có và giữ các dòng cũ.

## 9. Tổ chức file, đặt tên, dòng lệnh

**Tham số dòng lệnh đề xuất**:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--suite` | `L` | `S`, `L`, `SA-R`, `SA-B` |
| `--n` | theo bộ | một hoặc nhiều giá trị số nút thông tin |
| `--dist` | 	theo bộ (S: R; còn lại: R, C, RC) | một hoặc nhiều kiểu phân bố |
| `--seed` | 42 | seed gốc |
| `--count` | 5 | số instance mỗi cấu hình |
| `--R` | 3 (SA-R: 2–6) | số UAV mỗi xe, một hoặc nhiều giá trị |
| `--battery_mah` | 4276 (SA-B: 1900, 2850, 4276, 5100) | dung lượng pin, một hoặc nhiều giá trị |
| `--C` | 100 | năng lực C, ảnh hưởng K = ceil(n / C) |
| `--sorties_per_uav` | S: 2, còn lại: không đặt | s_max cho |
| `--value_dist` | uniform | uniform hoặc normal |
| `--config` | config.json | file cấu hình ghi đè; nếu không có thì dùng mặc định trong code |
| `--out` | data | thư mục gốc đầu ra |
| `--write_config` | tắt | ghi config.json mặc định rồi thoát (không ghi đè nếu file đã tồn tại) |

Tất cả hằng số cố định của Mục 2 nằm trong `config.json`
---
