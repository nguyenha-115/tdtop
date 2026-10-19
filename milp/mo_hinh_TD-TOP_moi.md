# Mô hình toán học mới: TD-TOP đa xe tải (MILP đa mục tiêu)

## 1. Tập hợp (Sets)

| Tập | Ý nghĩa |
|---|---|
| $K = \{1, \dots, K\}$ | tập xe tải |
| $N_1$ | các nút đường bộ trung gian |
| $V_1 = \{0\} \cup N_1 \cup \{0'\}$ | tập nút đường bộ (0: kho xuất phát, $0'$: kho kết thúc) |
| $V_2$ | tập nút thông tin cần thu thập |
| $V = V_1 \cup V_2$ | tập tất cả các nút |
| $Q_k$ | tập sortie của xe $k$; $Q = \bigcup_k Q_k$ |
| $A^T$ | tập cung của xe tải |
| $A^U$ | tập cung bay của UAV |
| $p_i$ | giá trị thông tin cố định của mỗi node $i \in V_2$ |
| $s_i$ | thời gian dịch vụ yêu cầu của mỗi node $i \in V_2$ |
| $t_{ij}^G = d_{ij}^{\text{Man}} / v^G$ | thời gian di chuyển của xe trên cạnh $(i,j) \in E_1$ |
| $t_{ij}^U = d_{ij}^{\text{Euclid}} / v^U$ | thời gian bay của UAV trên cạnh $(i,j) \in E_2$ |
| $\theta$ | dung lượng pin |
| $T$ | thời hạn nhiệm vụ |

## 2. Biến quyết định

| Biến | Ý nghĩa |
|---|---|
| $y_{ij}^k \in \{0,1\}$ | xe $k$ đi qua cung $(i,j) \in A^T$ |
| $w_i^k \in \{0,1\}$ | xe $k$ ghé nút đường bộ $i$ |
| $z_{iq} \in \{0,1\}$ | nút $i \in V_2$ được phục vụ trong sortie $q$ |
| $x_{ij,q} \in \{0,1\}$ | sortie $q$ bay qua cung $(i,j) \in A^U$ |
| $u_q \in \{0,1\}$ | sortie $q$ được sử dụng |
| $r_i^k \in \{0, \dots, R_k\}$ | số UAV trên xe $k$ khi rời nút $i$ |
| $\tau_{ik}^A, \tau_{ik}^D \in [0, T]$ | thời điểm xe $k$ đến và rời nút $i$ |
| $t_q^L, t_q^A, t_q^W \ge 0$ | thời điểm cất cánh, thời điểm tới điểm thu hồi, thời gian lơ lửng chờ của sortie $q$ |
| $E_q \ge 0$ | năng lượng tiêu thụ của sortie $q$ |
| $\nu_i^k, \mu_{iq} \ge 0$ | thứ tự thăm (MTZ) của xe và của sortie |

với $q \in Q_k$.

**Chỉ thị nút cất cánh / thu hồi** của sortie $q$ được suy ra từ tổng cung bay:

$$
h_{iq}^L = \sum_{j \in V_2} x_{ij,q} \quad (i \in V_1 \setminus \{0'\}), \qquad
h_{jq}^R = \sum_{i \in V_2} x_{ij,q} \quad (j \in V_1 \setminus \{0\})
\tag{34}
$$

**Số UAV được cất cánh/thu hồi** của xe $k$ tại một nút:

$$
\text{Lau}_i^k = \sum_{q \in Q_k} h_{iq}^L, \qquad
\text{Rec}_j^k = \sum_{q \in Q_k} h_{jq}^R, \qquad
w_{0'}^k := w_0^k
\tag{35}
$$

## 3. Hàm mục tiêu (đa mục tiêu)

**Mục tiêu 1** — tối đa hóa tổng giá trị thông tin thu thập được:

$$
\max f_1 = \sum_{q \in Q} \sum_{i \in V_2} p_i \, z_{iq}
\tag{36}
$$

**Mục tiêu 2** — tối thiểu hóa tổng năng lượng tiêu thụ của đội drone:

$$
\min f_2 = \sum_{q \in Q} E_q
\tag{37}
$$

## 5. Ràng buộc

### 5.1 Lộ trình xe tải

Mỗi xe tải xuất phát từ kho $0$ và quay về kho $0'$ đúng một lần nếu được sử dụng:

$$
w_0^k = \sum_j y_{0j}^k = \sum_i y_{i0'}^k \le 1, \quad \forall k \in K
\tag{38}
$$

Bảo toàn luồng tại các nút đường bộ trung gian (xe $k$ vào nút $i$ thì phải rời nút $i$, và chỉ khi nó ghé nút đó):

$$
\sum_j y_{ij}^k = \sum_j y_{ji}^k = w_i^k, \quad \forall i \in N_1, \ \forall k \in K
\tag{39}
$$

Mỗi nút đường bộ trung gian được tối đa một xe ghé thăm:

$$
\sum_{k \in K} w_i^k \le 1, \quad \forall i \in N_1
\tag{40}
$$

Ràng buộc MTZ loại bỏ chu trình con trong lộ trình của từng xe tải:

$$
1 \le \nu_i^k \le |N_1|, \qquad
\nu_j^k \ge \nu_i^k + 1 - |N_1| (1 - y_{ij}^k), \quad \forall i, j \in N_1, i \ne j, \ \forall k \in K
\tag{41}
$$

### 5.2 Lộ trình UAV / sortie

Mỗi nút thông tin được phục vụ tối đa một lần bởi một sortie:

$$
\sum_{q \in Q} z_{iq} \le 1, \quad \forall i \in V_2
\tag{42}
$$

Cân bằng luồng tại nút thông tin (nút được phục vụ trong sortie $q$ thì có đúng một cung vào và một cung ra):

$$
\sum_j x_{ij,q} = \sum_j x_{ji,q} = z_{iq}, \quad \forall i \in V_2, \ \forall q \in Q
\tag{43}
$$

Mỗi sortie được dùng có đúng một điểm cất cánh và một điểm thu hồi:

$$
\sum_{i \in V_1 \setminus \{0'\}} h_{iq}^L = u_q, \qquad
\sum_{j \in V_1 \setminus \{0\}} h_{jq}^R = u_q, \quad \forall q \in Q
\tag{44}
$$

Ràng buộc MTZ loại bỏ chu trình con trong đường bay của từng sortie:

$$
1 \le \mu_{iq} \le |V_2|, \qquad
\mu_{jq} \ge \mu_{iq} + 1 - |V_2| (1 - x_{ij,q}), \quad \forall i, j \in V_2, i \ne j, \ \forall q \in Q
\tag{45}
$$

Phá đối xứng giữa các sortie của cùng một xe (sortie có chỉ số nhỏ được dùng trước):

$$
u_q \ge u_{q+1}, \quad \forall q, q+1 \in Q_k, \ \forall k \in K
\tag{46}
$$

UAV chỉ cất cánh hoặc được thu hồi tại các nút mà xe $k$ đi qua:

$$
h_{iq}^L \le w_i^k \ (i \in V_1 \setminus \{0'\}), \qquad
h_{jq}^R \le w_j^k \ (j \in V_1 \setminus \{0\}), \quad \forall q \in Q_k, \ \forall k \in K
\tag{47}
$$

UAV được thu hồi tại điểm dừng kế tiếp của xe (nếu cất cánh tại $i$ và thu hồi tại $j$ thì xe phải đi trực tiếp từ $i$ đến $j$):

$$
h_{iq}^L + h_{jq}^R \le 1 + y_{ij}^k, \quad \forall (i,j) \in A^T, \ \forall q \in Q_k, \ \forall k \in K
\tag{48}
$$

Điểm cất cánh và điểm thu hồi của một sortie phải khác nhau:

$$
h_{iq}^L + h_{iq}^R \le 1, \quad \forall i \in N_1, \ \forall q \in Q
\tag{49}
$$

### 5.3 Bảo toàn số lượng UAV

Tại kho, số UAV còn lại sau khi cất cánh bằng $R_k$ trừ số UAV đã phóng; khi về kho, toàn bộ $R_k$ UAV phải trở lại xe:

$$
r_0^k = R_k - \text{Lau}_0^k, \qquad
R_k \, w_0^k \le r_{0'}^k \le R_k, \quad \forall k \in K
\tag{50}
$$

Cập nhật số UAV dọc theo cung xe tải (số UAV khi rời nút $j$ bằng số UAV khi rời nút $i$ cộng số UAV thu hồi tại $j$ trừ số UAV phóng tại $j$, với $\text{Lau}_{0'}^k = 0$); dạng giá trị tuyệt đối được tách thành hai bất đẳng thức tuyến tính:

$$
\left| r_j^k - \left( r_i^k + \text{Rec}_j^k - \text{Lau}_j^k \right) \right| \le 2 R_k (1 - y_{ij}^k), \quad \forall (i,j) \in A^T, \ \forall k \in K
\tag{51}
$$

### 5.4 Thời gian và đồng bộ hóa

Lan truyền thời gian dọc theo lộ trình xe tải:

$$
\left| \tau_{jk}^A - \tau_{ik}^D - t_{ij}^G \right| \le M_T (1 - y_{ij}^k), \quad \forall (i,j) \in A^T, \ \forall k \in K
\tag{52}
$$

Thời điểm rời nút không sớm hơn thời điểm đến:

$$
\tau_{ik}^D \ge \tau_{ik}^A, \quad \forall i \in V_1 \setminus \{0\}; \qquad
\tau_{0k}^D \ge 0, \quad \forall k \in K
\tag{53}
$$

UAV cất cánh đúng lúc xe rời điểm phóng:

$$
\left| t_q^L - \tau_{ik}^D \right| \le M_T (1 - h_{iq}^L), \quad \forall i \in V_1 \setminus \{0'\}, \ \forall q \in Q_k, \ \forall k \in K
\tag{54}
$$

Nhất quán thời gian của sortie (thời điểm tới điểm thu hồi bằng thời điểm cất cánh cộng tổng thời gian bay và thời gian phục vụ):

$$
\left| t_q^A - t_q^L - \sum_{(i,j) \in A^U} t_{ij}^U x_{ij,q} - \sum_{i \in V_2} s_i z_{iq} \right| \le M_T (1 - u_q), \quad \forall q \in Q
\tag{55}
$$

Xe chỉ được rời điểm thu hồi sau khi UAV đã được thu hồi:

$$
\tau_{jk}^D \ge t_q^A - M_T (1 - h_{jq}^R), \quad \forall j \in V_1 \setminus \{0\}, \ \forall q \in Q_k, \ \forall k \in K
\tag{56}
$$

Thời gian UAV lơ lửng chờ xe tải (chỉ xuất hiện trong năng lượng nên cận dưới là đủ):

$$
t_q^W \ge \tau_{jk}^A - t_q^A - M_T (1 - h_{jq}^R), \qquad t_q^W \ge 0, \quad \forall j \in V_1 \setminus \{0\}, \ \forall q \in Q_k, \ \forall k \in K
\tag{57}
$$

Mọi xe tải và mọi UAV phải trở về kho trước thời hạn $T$:

$$
\tau_{0'k}^D \le T, \qquad t_q^A \le T, \quad \forall k \in K, \ \forall q \in Q
\tag{58}
$$

### 5.5 Năng lượng

#### 5.5.1 UAV endurance capability

Năng lượng drone gồm ba thành phần: năng lượng bay, năng lượng chờ treo và năng lượng phục vụ thu thập thông tin.

**Công suất bay** (mô hình cánh quạt quay, theo Zeng et al. 2019):

$$
P^{fly}(V) = P_0 \left( 1 + \frac{3V^2}{U_{tip}^2} \right) + P_i \left( \sqrt{1 + \frac{V^4}{4 v_0^4}} - \frac{V^2}{2 v_0^2} \right)^{1/2} + \frac{1}{2} d_0 \rho q A V^3
\tag{1}
$$

trong đó:

- $P_0, P_i$ là công suất cản khí động của cánh quạt và công suất cảm ứng khi bay lơ lửng;
- $U_{tip}$ là tốc độ đầu cánh quạt;
- $v_0$ là vận tốc cảm ứng khi treo;
- $d_0, \rho, q, A$ lần lượt là hệ số cản thân UAV, mật độ không khí, độ đặc cánh quạt và diện tích đĩa quay.

**Năng lượng bay trên cung $(i,j)$:**

$$
e_{ij}^{fly} = P^{fly}(v^U) \, t_{ij}^U
\tag{2}
$$

**Năng lượng chờ lơ lửng** trong sortie $q$ (nếu drone đến trước xe):

$$
E_q^{wait} = P^{hover} \, t_q^{W} = (P_i + P_0) \, t_q^{W}
\tag{3}
$$

**Năng lượng phục vụ** tại node $i$ (cảm biến, tính toán, truyền dữ liệu):

$$
E_i^{srv} = P^{srv} s_i = (P_{avio} + P_{hover}) \, s_i, \qquad P_{avio} = P_{sens} + P_{comp} + P_{tx}
\tag{4}
$$

**Tổng năng lượng của sortie $q$** (dạng tổng quát, trước khi tuyến tính hóa theo biến quyết định ở (59)):

$$
E_q = \sum e_{ij}^{fly} + \sum E_i^{srv} + E_q^{wait}
\tag{5}
$$

#### 5.5.2 Ràng buộc năng lượng theo biến quyết định

Năng lượng của mỗi sortie gồm năng lượng bay, năng lượng phục vụ và năng lượng lơ lửng chờ, viết theo các biến nhị phân $x_{ij,q}$, $z_{iq}$ và biến $t_q^W$:

$$
E_q = \sum_{(i,j) \in A^U} e_{ij}^{fly} x_{ij,q} + \sum_{i \in V_2} P^{srv} s_i z_{iq} + P^{hover} t_q^W, \quad \forall q \in Q
\tag{59}
$$

Ràng buộc dung lượng pin cho từng sortie:

$$
E_q \le \theta \, u_q, \quad \forall q \in Q
\tag{60}
$$

với $\theta$ là dung lượng pin tối đa (xem Mục 2 - Tập hợp), đóng vai trò giới hạn trên cho $E_q$ tính theo (1)-(5).

NSGA2 + RL nhưng kết hợp chứng minh lý thuyết KL

### Cách đọc file json
Sortie	`T1-S3` - Sortie thứ 3 của xe 1, đánh số theo thời điểm cất cánh

UAV	`T1-U2`	- UAV số 2 của xe 1

`truck`: xe mang UAV.

`uav`: UAV thực hiện sortie, ví dụ "T1-U2".

`uav_trip_no`: đây là chuyến thứ mấy của UAV đó.