"""
Ví dụ:
    python td_top.py --filename S042_N10_R_R3.json --subfolder S/N10
    python td_top.py --subfolder S/N10
"""
from ortools.linear_solver import pywraplp
import numpy as np
import json
import argparse
import sys
import os
import glob
import time

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str,
                     help="Tên file JSON instance, vd S042_N10_R_R3.json")
parser.add_argument("--subfolder", type=str,
                     help="Thư mục con dưới DATA_ROOT, vd S/N10 (khớp cấu trúc data.py)")
parser.add_argument("--s_max", type=int, default=2,
                     help="Số sortie tối đa / UAV nếu data không khai báo SORTIES_PER_TRUCK")
parser.add_argument("--timelimit", type=float, default=36000.0,
                     help="Giới hạn thời gian giải (giây); mặc định 10 giờ = 36000s")
args = parser.parse_args()

DATA_ROOT = "../data/data"
RESULT_ROOT = "./result/"

filename = args.filename
subfolder = args.subfolder

files = []  # list[(filename, subfolder)]
if filename and subfolder:
    files.append((filename, subfolder))
elif filename:
    matches = glob.glob(os.path.join(DATA_ROOT, "**", filename), recursive=True)
    if not matches:
        print(f"Không tìm thấy {filename} trong {DATA_ROOT}")
        sys.exit(1)
    for m in matches:
        files.append((os.path.basename(m), os.path.relpath(os.path.dirname(m), DATA_ROOT)))
elif subfolder:
    subfolder_path = os.path.join(DATA_ROOT, subfolder)
    for f in sorted(os.listdir(subfolder_path)):
        if os.path.isfile(os.path.join(subfolder_path, f)):
            files.append((f, subfolder))
else:
    print("Cần truyền --filename và/hoặc --subfolder")
    sys.exit(1)


def run(filename, subfolder):
    data_path = os.path.join(DATA_ROOT, subfolder, filename)
    output_dir = os.path.join(RESULT_ROOT, "f1", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfgG = data["Config"]["General"]
    cfgV = data["Config"]["Vehicles"]
    cfgE = data["Config"]["Energy"]

    depot = data["Depot"]
    id_start = depot["id_start"]
    id_end = depot["id_end"]
    depot_coord = np.array(depot["coord"], dtype=float)

    road_nodes = data["RoadNodes"]
    info_nodes = data["InfoNodes"]

    # ---------------- Mục 1: Tập hợp ----------------
    N1 = [nd["id"] for nd in road_nodes]
    V2 = [nd["id"] for nd in info_nodes]
    V1 = [id_start] + N1 + [id_end]

    coord = {id_start: depot_coord, id_end: depot_coord}
    for nd in road_nodes:
        coord[nd["id"]] = np.array(nd["coord"], dtype=float)
    for nd in info_nodes:
        coord[nd["id"]] = np.array(nd["coord"], dtype=float)

    p = {nd["id"]: int(nd["p"]) for nd in info_nodes}
    s_hours = {nd["id"]: nd["s_min"] / 60.0 for nd in info_nodes}

    num_trucks = int(cfgV["NUM_TRUCKS"])
    K = list(range(1, num_trucks + 1))
    R_list = cfgV["UAVS_PER_TRUCK"]
    R = {K[idx]: int(R_list[idx]) for idx in range(num_trucks)}

    V_TRUCK = float(cfgV["V_TRUCK"])
    V_UAV = float(cfgV["V_UAV"])
    THETA = float(cfgV["THETA_WH"])
    T = float(cfgG["T_MAX_MIN"]) / 60.0  # giờ

    sorties_per_truck_cfg = cfgV.get("SORTIES_PER_TRUCK")
    s_max = {}
    for k in K:
        if sorties_per_truck_cfg and R[k] > 0:
            s_max[k] = max(1, round(sorties_per_truck_cfg / R[k]))
        else:
            s_max[k] = args.s_max

    P_SRV = float(cfgE["P_SRV_W"])
    P_HOVER = float(cfgE["P_HOVER_W"])
    E_FLY_PER_KM = float(cfgE["E_FLY_WH_PER_KM"])

    def manhattan(i, j):
        return float(np.sum(np.abs(coord[i] - coord[j])))

    def euclid(i, j):
        return float(np.linalg.norm(coord[i] - coord[j]))

    # Tập cung xe tải: trong V1, không rời kho kết thúc, không tới kho xuất phát
    A_T = [(i, j) for i in V1 for j in V1 if i != j and i != id_end and j != id_start]
    A_T_set = set(A_T)
    t_G = {(i, j): manhattan(i, j) / V_TRUCK for (i, j) in A_T}
    out_T = {i: [j for j in V1 if (i, j) in A_T_set] for i in V1}
    in_T = {i: [j for j in V1 if (j, i) in A_T_set] for i in V1}

    # Tập cung bay UAV 
    LN = [i for i in V1 if i != id_end]   # có thể cất cánh
    LD = [j for j in V1 if j != id_start]  # có thể thu hồi
    A_U = [(i, j) for i in LN for j in V2] + \
          [(i, j) for i in V2 for j in V2 if i != j] + \
          [(i, j) for i in V2 for j in LD]
    t_U = {(i, j): euclid(i, j) / V_UAV for (i, j) in A_U}
    e_fly = {(i, j): E_FLY_PER_KM * euclid(i, j) for (i, j) in A_U}
    V2_out = {i: [j for j in V2 if j != i] + LD for i in V2}
    V2_in = {i: LN + [j for j in V2 if j != i] for i in V2}

    # Tập sortie Q_k (đánh số phẳng theo từng xe)
    Qk = {}
    for k in K:
        Qk[k] = [(k, idx) for idx in range(1, R[k] * s_max[k] + 1)]
    Q = [q for k in K for q in Qk[k]]

    M_T = T if T > 0 else 1.0

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Không tạo được solver SCIP")
        return

    # ---------------- Mục 2: Biến quyết định ----------------
    y = {(k, i, j): solver.BoolVar(f"y_{k}_{i}_{j}") for k in K for (i, j) in A_T}
    w0 = {k: solver.BoolVar(f"w0_{k}") for k in K}          # w_0^k = w_{0'}^k (35)
    w = {(k, i): solver.BoolVar(f"w_{k}_{i}") for k in K for i in N1}

    def w_of(k, i):
        return w0[k] if (i == id_start or i == id_end) else w[k, i]

    z = {(i, q): solver.BoolVar(f"z_{i}_{q[0]}_{q[1]}") for i in V2 for q in Q}
    x = {(i, j, q): solver.BoolVar(f"x_{i}_{j}_{q[0]}_{q[1]}") for (i, j) in A_U for q in Q}
    u = {q: solver.BoolVar(f"u_{q[0]}_{q[1]}") for q in Q}

    r = {(k, i): solver.IntVar(0, R[k], f"r_{k}_{i}")
         for k in K for i in ([id_start] + N1 + [id_end])}

    tauA = {(k, i): solver.NumVar(0, T, f"tauA_{k}_{i}") for k in K for i in V1}
    tauD = {(k, i): solver.NumVar(0, T, f"tauD_{k}_{i}") for k in K for i in V1}

    tL = {q: solver.NumVar(0, T, f"tL_{q[0]}_{q[1]}") for q in Q}
    tA = {q: solver.NumVar(0, T, f"tA_{q[0]}_{q[1]}") for q in Q}
    tW = {q: solver.NumVar(0, T, f"tW_{q[0]}_{q[1]}") for q in Q}
    E = {q: solver.NumVar(0, THETA, f"E_{q[0]}_{q[1]}") for q in Q}

    nu = {(k, i): solver.NumVar(1, max(len(N1), 1), f"nu_{k}_{i}") for k in K for i in N1}
    mu = {(i, q): solver.NumVar(1, max(len(V2), 1), f"mu_{i}_{q[0]}_{q[1]}")
          for i in V2 for q in Q}

    # Chỉ thị cất cánh / thu hồi (34), suy từ tổng cung bay
    def hL(i, q):
        if i not in LN:
            return 0
        return solver.Sum([x[i, j, q] for j in V2])

    def hR(j, q):
        if j not in LD:
            return 0
        return solver.Sum([x[i, j, q] for i in V2])

    # Hàm mục tiêu
    f1_expr = solver.Sum([p[i] * z[i, q] for i in V2 for q in Q])                 # (36)
    # f2_expr = solver.Sum([E[q] for q in Q])                                      # (37)
    solver.Maximize(f1_expr)
    # solver.Minimize(f2_expr)
    # Lộ trình xe tải
    for k in K:  # (38)
        solver.Add(w0[k] == solver.Sum([y[k, id_start, j] for j in out_T[id_start]]))
        solver.Add(w0[k] == solver.Sum([y[k, i, id_end] for i in in_T[id_end]]))

    for k in K:  # (39)
        for i in N1:
            solver.Add(solver.Sum([y[k, i, j] for j in out_T[i]]) == w[k, i])
            solver.Add(solver.Sum([y[k, j, i] for j in in_T[i]]) == w[k, i])

    for i in N1:  # (40)
        solver.Add(solver.Sum([w[k, i] for k in K]) == 1)

    for k in K:  # (41) MTZ xe tải
        for i in N1:
            for j in N1:
                if i != j and (i, j) in A_T_set:
                    solver.Add(nu[k, j] >= nu[k, i] + 1 - len(N1) * (1 - y[k, i, j]))

    # Mục 5.2: Lộ trình UAV / sortie 
    for i in V2: 
        solver.Add(solver.Sum([z[i, q] for q in Q]) <= 1)

    for i in V2:  # (43)
        for q in Q:
            solver.Add(solver.Sum([x[i, j, q] for j in V2_out[i]]) == z[i, q])
            solver.Add(solver.Sum([x[j, i, q] for j in V2_in[i]]) == z[i, q])

    for q in Q:  # (44)
        solver.Add(solver.Sum([hL(i, q) for i in LN]) == u[q])
        solver.Add(solver.Sum([hR(j, q) for j in LD]) == u[q])

    for q in Q:  # (45) MTZ sortie
        for i in V2:
            for j in V2:
                if i != j:
                    solver.Add(mu[j, q] >= mu[i, q] + 1 - len(V2) * (1 - x[i, j, q]))

    for k in K:  # (46) phá đối xứng sortie cùng xe
        qs = Qk[k]
        for idx in range(len(qs) - 1):
            solver.Add(u[qs[idx]] >= u[qs[idx + 1]])

    for k in K:  # (47)
        for q in Qk[k]:
            for i in LN:
                solver.Add(hL(i, q) <= w_of(k, i))
            for j in LD:
                solver.Add(hR(j, q) <= w_of(k, j))

    for k in K:  # (48)
        for q in Qk[k]:
            for (i, j) in A_T:
                solver.Add(hL(i, q) + hR(j, q) <= 1 + y[k, i, j])

    for q in Q:  # (49)
        for i in N1:
            solver.Add(hL(i, q) + hR(i, q) <= 1)

    # ---------------- Mục 5.3: Bảo toàn số lượng UAV ----------------
    for k in K:  # (50)
        Lau0 = solver.Sum([hL(id_start, q) for q in Qk[k]])
        solver.Add(r[k, id_start] == R[k] - Lau0)
        solver.Add(r[k, id_end] >= R[k] * w0[k])
        solver.Add(r[k, id_end] <= R[k])

    for k in K:  # (51)
        for (i, j) in A_T:
            Rec_j = solver.Sum([hR(j, q) for q in Qk[k]])
            Lau_j = solver.Sum([hL(j, q) for q in Qk[k]]) if j in LN else 0
            solver.Add(r[k, j] - (r[k, i] + Rec_j - Lau_j) <= 2 * R[k] * (1 - y[k, i, j]))
            solver.Add(r[k, j] - (r[k, i] + Rec_j - Lau_j) >= -2 * R[k] * (1 - y[k, i, j]))

    # ---------------- Mục 5.4: Thời gian và đồng bộ hóa ----------------
    for k in K:  # (52)
        for (i, j) in A_T:
            solver.Add(tauA[k, j] - tauD[k, i] - t_G[i, j] <= M_T * (1 - y[k, i, j]))
            solver.Add(tauA[k, j] - tauD[k, i] - t_G[i, j] >= -M_T * (1 - y[k, i, j]))

    for k in K:  # (53)
        for i in V1:
            if i != id_start:
                solver.Add(tauD[k, i] >= tauA[k, i])
        solver.Add(tauD[k, id_start] >= 0)

    for k in K:  # (54)
        for q in Qk[k]:
            for i in LN:
                solver.Add(tL[q] - tauD[k, i] <= M_T * (1 - hL(i, q)))
                solver.Add(tL[q] - tauD[k, i] >= -M_T * (1 - hL(i, q)))

    for q in Q:  # (55)
        flight = solver.Sum([t_U[i, j] * x[i, j, q] for (i, j) in A_U])
        service = solver.Sum([s_hours[i] * z[i, q] for i in V2])
        solver.Add(tA[q] - tL[q] - flight - service <= M_T * (1 - u[q]))
        solver.Add(tA[q] - tL[q] - flight - service >= -M_T * (1 - u[q]))

    for k in K:  # (56)
        for q in Qk[k]:
            for j in LD:
                solver.Add(tauD[k, j] >= tA[q] - M_T * (1 - hR(j, q)))

    for k in K:  # (57)
        for q in Qk[k]:
            for j in LD:
                solver.Add(tW[q] >= tauA[k, j] - tA[q] - M_T * (1 - hR(j, q)))

    for k in K:  # (58)
        solver.Add(tauD[k, id_end] <= T)
    for q in Q:
        solver.Add(tA[q] <= T)

    # ---------------- Mục 5.5: Năng lượng ----------------
    for q in Q:  # (59)
        flight_e = solver.Sum([e_fly[i, j] * x[i, j, q] for (i, j) in A_U])
        srv_e = solver.Sum([P_SRV * s_hours[i] * z[i, q] for i in V2])
        solver.Add(E[q] == flight_e + srv_e + P_HOVER * tW[q])

    for q in Q:  # (60)
        solver.Add(E[q] <= THETA * u[q])

    print(f"[{filename}] Model: {solver.NumVariables()} biến, {solver.NumConstraints()} ràng buộc")
    print("Đang giải...")

    solver.SetTimeLimit(int(args.timelimit * 1000))
    start_time = time.time()
    status = solver.Solve()
    running_time = time.time() - start_time

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        status_str = "Optimal" if status == pywraplp.Solver.OPTIMAL else "Feasible"

        # ---------- Quy ước đặt tên ----------
        #   Xe tải     : T{k}              vd T1, T2
        #   UAV        : T{k}-U{m}         vd T1-U1  (UAV số 1 của xe 1)
        #   Sortie     : T{k}-S{n}         vd T1-S3  (sortie thứ 3 của xe 1, đánh số theo thời điểm cất cánh)
        #   Mỗi sortie có thêm: "uav" (UAV thực hiện) và "uav_trip_no" (đây là chuyến thứ mấy của UAV đó)
        EPS = 1e-6
        routes = []
        total_sorties = 0
        used_uavs = set()

        for k in K:
            if w0[k].solution_value() < 0.5:
                continue  # xe k không được dùng

            route = [id_start]
            current = id_start
            visited = {id_start}
            for _ in range(len(V1) + 2):
                nxt = None
                for j in out_T[current]:
                    if j not in visited and y[k, current, j].solution_value() > 0.5:
                        nxt = j
                        break
                if nxt is None:
                    break
                route.append(nxt)
                visited.add(nxt)
                current = nxt
                if current == id_end:
                    break

            # --- Thu thập các sortie của xe k ---
            raw_sorties = []
            for q in Qk[k]:
                if u[q].solution_value() <= 0.5:
                    continue
                launch = None
                for i in LN:
                    h = hL(i, q)
                    if not isinstance(h, int) and h.solution_value() > 0.5:
                        launch = i
                        break
                land = None
                for j in LD:
                    h = hR(j, q)
                    if not isinstance(h, int) and h.solution_value() > 0.5:
                        land = j
                        break
                if launch is None:
                    continue

                path = [launch]
                current_n = launch
                visited_n = {launch}
                for _ in range(len(V2) + 2):
                    nxt = None
                    candidates = V2 if current_n in LN else V2_out.get(current_n, [])
                    for j in candidates:
                        if j in visited_n:
                            continue
                        if (current_n, j) not in t_U:
                            continue
                        if x[current_n, j, q].solution_value() > 0.5:
                            nxt = j
                            break
                    if nxt is None:
                        break
                    path.append(nxt)
                    visited_n.add(nxt)
                    current_n = nxt
                    if current_n == land:
                        break

                raw_sorties.append({
                    "launch": launch,
                    "land": land,
                    "path": path,
                    "t_launch": tL[q].solution_value(),
                    "t_arrival": tA[q].solution_value(),
                    "t_wait": tW[q].solution_value(),
                    "energy_wh": E[q].solution_value(),
                    "served_nodes": [i for i in V2 if z[i, q].solution_value() > 0.5],
                })

            # --- Đánh số sortie theo thời điểm cất cánh ---
            raw_sorties.sort(key=lambda s: s["t_launch"])

            # --- Gán UAV cho từng sortie (first-fit: UAV rảnh có số thứ tự nhỏ nhất) ---
            uav_free_at = [0.0] * R[k]      # thời điểm UAV m được thu hồi xong / sẵn sàng
            uav_trip_count = [0] * R[k]
            uav_sortie_ids = [[] for _ in range(R[k])]
            sorties_out = []

            for n, s in enumerate(raw_sorties, start=1):
                free = [m for m in range(R[k]) if uav_free_at[m] <= s["t_launch"] + EPS]
                m = free[0] if free else min(range(R[k]), key=lambda mm: uav_free_at[mm])
                uav_free_at[m] = s["t_arrival"]
                uav_trip_count[m] += 1

                sortie_id = f"T{k}-S{n}"
                uav_id = f"T{k}-U{m + 1}"
                uav_sortie_ids[m].append(sortie_id)
                used_uavs.add(uav_id)

                sorties_out.append({
                    "id": sortie_id,
                    "truck": k,
                    "uav": uav_id,
                    "uav_trip_no": uav_trip_count[m],
                    **s,
                })

            total_sorties += len(sorties_out)

            uavs_out = [
                {"id": f"T{k}-U{m + 1}",
                 "num_sorties": len(uav_sortie_ids[m]),
                 "sorties": uav_sortie_ids[m]}
                for m in range(R[k]) if uav_sortie_ids[m]
            ]

            routes.append({
                "id": k,
                "route": route,
                "arrival": [tauA[k, i].solution_value() for i in route],
                "departure": [tauD[k, i].solution_value() for i in route],
                "uavs": uavs_out,
                "sorties": sorties_out,
            })

        # --- Tóm tắt ---
        served_set = {i for i in V2 if sum(z[i, q].solution_value() for q in Q) > 0.5}
        summary = {
            "num_trucks_available": len(K),
            "num_trucks_used": len(routes),
            "num_info_nodes": len(V2),
            "num_info_nodes_served": len(served_set),
            "unserved_info_nodes": [i for i in V2 if i not in served_set],
            "total_sorties": total_sorties,
            "total_drones_used": len(used_uavs),
        }

        result_data = {
            "status": status_str,
            "objective": solver.Objective().Value(),
            "f1_total_value": f1_expr.solution_value(),
            # "f2_total_energy_wh": f2_expr.solution_value(),
            "time_sec": running_time,
            "summary": summary,
            "routes": routes,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"f1 (giá trị thu được) = {result_data['f1_total_value']:.2f}")
        # print(f"f2 (năng lượng, Wh)   = {result_data['f2_total_energy_wh']:.4f}")
        print(f"Trạng thái: {status_str}, thời gian giải: {running_time:.1f}s")
        print(f"✔ Kết quả đã lưu: {output_path}")
    else:
        print(f"[{filename}] Không tìm được lời giải khả thi trong giới hạn thời gian "
              f"(status={status}, {running_time:.1f}s)")


for fname, sfolder in files:
    run(fname, sfolder)