"""Render docs/roadmap.png: go-live Gantt chart with dependency arrows (relative time, no dates)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "Microsoft JhengHei", "sans-serif"]

SECTIONS = [  # (name, color)
    ("1 開戶與資金", "#2a78d6"),
    ("2 架設環境", "#eb6834"),
    ("3 紙上交易", "#1baf7a"),
    ("4 API 與演練", "#eda100"),
    ("5 實盤上線", "#6250d6"),
    ("6 混合策略", "#e87ba4"),
]

# id, section, label, display length, dependencies, duration note
# display length is schematic: long phases are compressed so short steps stay readable
TASKS = [
    ("a1", 0, "台灣交易所開戶、實名", 2, [], "約 2 週"),
    ("a2", 0, "幣安開戶、實名、兩步驟驗證", 2, [], "約 2 週"),
    ("a3", 0, "小額測試轉帳", 1, ["a1", "a2"], ""),
    ("a4", 0, "開啟 BNB 抵扣", 1, ["a3"], ""),
    ("b1", 1, "準備執行機器", 2, [], "約 2 週"),
    ("b2", 1, "安裝環境、跑測試", 1, ["b1"], ""),
    ("b3", 1, "設定每日排程 08:05", 1, ["b2"], ""),
    ("c1", 2, "paper 模式每日執行", 5, ["b3"], "2～3 個月"),
    ("m1", 2, "◆ 紙上交易驗收", 0, ["c1"], ""),
    ("d1", 3, "建立 API key", 1, ["a4"], ""),
    ("d2", 3, "dry 模式演練", 1, ["d1", "m1"], ""),
    ("e1", 4, "小額實盤 5%～10%", 6, ["d2"], "約半年"),
    ("m2", 4, "◆ 小額實盤驗收", 0, ["e1"], ""),
    ("e3", 4, "加碼到 20%", 6, ["m2"], "約半年"),
    ("m3", 4, "◆ 加碼到 30%（目標配置）", 0, ["e3"], ""),
    ("f1", 5, "追蹤相關性、檢查停止條件", None, ["d2"], "持續"),
    ("m4", 5, "◆ 每年再平衡回 70 / 30", 0, ["m3"], ""),
]

start, end = {}, {}
for tid, _, _, length, deps, _ in TASKS:
    start[tid] = max([end[d] for d in deps], default=0)
    end[tid] = start[tid] + (length or 0)
horizon = max(end.values()) + 2
for tid, _, _, length, _, _ in TASKS:
    if length is None:
        end[tid] = horizon

row = {t[0]: i for i, t in enumerate(TASKS)}
n = len(TASKS)
fig, ax = plt.subplots(figsize=(14, 8.5), dpi=150)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# stage bands across the top, in order
STAGES = [("準備", 0, 4), ("紙上交易", 4, end["c1"]), ("演練", end["c1"], end["d2"]),
          ("小額實盤", end["d2"], end["e1"]), ("加碼 20%", end["e1"], end["e3"]),
          ("目標配置 30%", end["e3"], horizon)]
for i, (name, x0, x1) in enumerate(STAGES):
    ax.axvspan(x0, x1, color="#f3f2ee" if i % 2 == 0 else "#ffffff", zorder=0)
    ax.axvline(x0, color="#c3c2b7", lw=1, ls="--", zorder=1)
    ax.text((x0 + x1) / 2, -1.1, f"{i + 1}. {name}", ha="center", va="center",
            fontsize=11, color="#3d3d3a", fontweight="bold")

H = 0.56
for tid, sec, label, length, deps, note in TASKS:
    y, color = row[tid], SECTIONS[sec][1]
    x0, x1 = start[tid], end[tid]
    if length == 0:
        ax.scatter([x0], [y], marker="D", s=110, color=color, edgecolor="#ffffff", lw=2, zorder=5)
        ax.text(x0 + 0.3, y, label.lstrip("◆ "), va="center", fontsize=10, color="#1a1a19", zorder=6)
        continue
    bar = FancyBboxPatch((x0 + 0.04, y - H / 2), x1 - x0 - 0.08, H,
                         boxstyle="round,pad=0,rounding_size=0.12", fc=color, ec="none", zorder=4)
    ax.add_patch(bar)
    if length is None:  # continuous: arrowhead at the right edge
        ax.annotate("", xy=(horizon + 0.5, y), xytext=(horizon - 0.1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=4)
    inside = (x1 - x0) >= 4
    ax.text(x0 + 0.2 if inside else x1 + 0.2, y, label + (f"（{note}）" if note else ""),
            va="center", fontsize=10, color="#ffffff" if inside else "#1a1a19", zorder=6)

# dependency arrows: from end of predecessor to start of successor
for tid, _, _, _, deps, _ in TASKS:
    for d in deps:
        x0, y0, x1, y1 = end[d], row[d], start[tid], row[tid]
        ax.annotate("", xy=(x1 + 0.04, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#6b6a64", lw=1.1,
                                    connectionstyle="angle,angleA=-90,angleB=180,rad=4" if x1 > x0 + 0.3
                                    else "angle,angleA=-90,angleB=180,rad=0", shrinkA=0, shrinkB=0),
                    zorder=3)

ax.set_yticks(range(n))
ax.set_yticklabels([SECTIONS[t[1]][0] if i == 0 or TASKS[i - 1][1] != t[1] else "" for i, t in enumerate(TASKS)],
                   fontsize=11)
for i, t in enumerate(TASKS):
    if i and TASKS[i - 1][1] != t[1]:
        ax.axhline(i - 0.5, color="#e1e0d9", lw=1, zorder=1)
ax.tick_params(axis="y", length=0)
ax.set_xticks([])
ax.set_xlim(-0.3, horizon + 0.8)
ax.set_ylim(n - 0.4, -1.7)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("上線路線圖：由左到右依序進行，箭頭表示「完成後才進入下一步」", fontsize=14, loc="left", pad=12)
fig.text(0.01, 0.01, "長度只示意相對時間（長階段已壓縮，括號內為實際約略時間）；每一步以完成條件為準。",
         fontsize=9, color="#6b6a64")
fig.tight_layout()
fig.savefig("docs/roadmap.png", facecolor="#ffffff")
print("saved docs/roadmap.png")
