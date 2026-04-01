#!/usr/bin/env python3
"""Generate experiment result visualizations."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Experiment 2 (HFP V2 Distillation) — parsed from training log
# ============================================================
hfp_data = {
    "step":  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
              210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
              310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
              410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
              510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
              610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
              710, 720, 730, 740, 750, 760, 770, 780, 790, 800,
              810, 820, 830, 840, 850, 860, 870, 880, 890, 900,
              910, 920, 930, 940, 950, 960, 970, 980, 990, 1000,
              1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100,
              1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200,
              1210, 1220, 1230, 1240, 1250],
    "loss":  [17.4899, 17.3933, 17.3677, 17.3520, 17.3007, 17.2335, 17.1375, 17.0309, 16.9113, 16.7688,
              16.6024, 16.4362, 16.2428, 16.0327, 15.8240, 15.6169, 15.4130, 15.2205, 15.0356, 14.8521,
              14.6759, 14.5088, 14.3443, 14.1906, 14.0471, 13.9013, 13.7602, 13.6340, 13.5074, 13.3837,
              13.2683, 13.1582, 13.0470, 12.9365, 12.8349, 12.7389, 12.6431, 12.5478, 12.4575, 12.3732,
              12.2873, 12.2040, 12.1229, 12.0479, 11.9725, 11.8986, 11.8230, 11.7525, 11.6852, 11.6202,
              11.5610, 11.4970, 11.4364, 11.3801, 11.3263, 11.2738, 11.2230, 11.1726, 11.1219, 11.0744,
              11.0274, 10.9821, 10.9385, 10.8940, 10.8514, 10.8079, 10.7666, 10.7263, 10.6881, 10.6492,
              10.6142, 10.5782, 10.5443, 10.5117, 10.4796, 10.4479, 10.4149, 10.3833, 10.3527, 10.3240,
              10.2955, 10.2679, 10.2410, 10.2132, 10.1879, 10.1622, 10.1374, 10.1126, 10.0880, 10.0650,
              10.0416, 10.0178, 9.9949, 9.9721, 9.9514, 9.9316, 9.9117, 9.8915, 9.8716, 9.8541,
              9.8352, 9.8164, 9.7990, 9.7812, 9.7636, 9.7467, 9.7309, 9.7143, 9.6987, 9.6831,
              9.6667, 9.6514, 9.6373, 9.6230, 9.6092, 9.5960, 9.5830, 9.5696, 9.5567, 9.5426,
              9.5285, 9.5173, 9.5053, 9.4931, 9.4802],
    "kl":    [15.6432, 16.9077, 16.3653, 15.3004, 15.7770, 15.5787, 15.6836, 15.3112, 15.1287, 14.2795,
              14.1527, 13.3852, 12.6425, 12.4275, 12.1899, 11.4964, 11.3061, 11.2774, 11.2135, 11.0019,
              9.9302, 9.9848, 10.2603, 10.4644, 10.3295, 9.7120, 9.7276, 9.7759, 9.4147, 9.4936,
              9.2165, 9.0920, 9.3462, 8.9451, 8.6210, 8.7627, 8.8899, 8.8947, 8.1964, 8.7805,
              8.1103, 8.0546, 8.1362, 8.6852, 8.4727, 8.0544, 7.7835, 7.8471, 8.0579, 7.9736,
              8.4240, 8.8903, 7.4614, 8.3322, 7.4158, 7.8856, 8.1430, 8.1340, 7.6771, 7.5431,
              7.2596, 7.6350, 8.2937, 7.6336, 7.5952, 7.2724, 7.2919, 7.4159, 7.1732, 7.5407,
              7.2591, 8.0465, 7.6478, 7.5167, 7.6858, 6.9824, 7.5049, 7.6896, 7.6138, 7.3974,
              7.4835, 7.4546, 7.9795, 7.2343, 7.3754, 7.8101, 7.9632, 7.2794, 7.3795, 7.3118,
              7.3885, 7.3338, 7.5998, 7.0586, 7.5513, 7.5885, 7.5027, 7.5661, 7.0712, 7.6533,
              8.2393, 7.2758, 7.7414, 7.5867, 7.3278, 8.2873, 7.5883, 7.4901, 7.5182, 7.8175,
              7.0408, 7.6560, 7.1846, 7.7725, 7.4592, 6.8388, 7.5390, 7.0724, 6.9814, 7.2459,
              7.0690, 8.0344, 8.0866, 7.3213, 7.4332],
    "ce":    [18.7748, 18.9254, 18.4054, 18.5085, 18.3401, 18.0308, 17.4705, 17.4123, 16.7335, 16.0572,
              15.4230, 15.5288, 14.6655, 13.9586, 13.3269, 12.9192, 12.0170, 11.7409, 12.8363, 11.6621,
              11.4828, 11.2495, 11.4149, 11.3269, 11.5739, 10.4937, 10.5650, 10.7133, 10.8160, 10.3548,
              9.8166, 10.0972, 10.1622, 9.7717, 10.1625, 9.6952, 9.3840, 10.2437, 8.8887, 9.3390,
              9.3096, 9.5877, 9.2938, 9.4698, 9.4736, 8.8606, 9.3422, 8.4690, 9.4018, 8.8175,
              8.8798, 8.1604, 8.8457, 8.8083, 8.7719, 8.3680, 9.4759, 8.7124, 8.6233, 8.9348,
              8.1482, 8.4013, 8.8883, 8.9725, 8.7717, 7.4879, 8.5243, 8.8243, 8.1239, 8.3775,
              8.3120, 8.6060, 8.5035, 8.7183, 8.6445, 8.1986, 8.8781, 8.2392, 8.9658, 8.7133,
              8.8554, 8.3197, 8.6480, 8.3697, 8.3934, 8.7276, 8.6566, 8.3880, 8.7021, 7.9630,
              8.2242, 8.4814, 8.6004, 8.2299, 8.7113, 8.8056, 8.8544, 8.4175, 8.2822, 8.2357,
              8.5635, 8.3128, 8.6771, 8.5334, 8.5567, 9.5629, 7.9628, 8.3806, 8.9988, 9.1021,
              8.2367, 8.5039, 8.4899, 8.9818, 8.1377, 8.6480, 8.3939, 8.5183, 8.3245, 8.6338,
              8.0134, 8.6269, 8.6594, 8.8283, 7.9769],
}

os.makedirs("figures", exist_ok=True)

# --- Style ---
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'figure.dpi': 150,
})

steps = np.array(hfp_data["step"])
loss = np.array(hfp_data["loss"])
kl = np.array(hfp_data["kl"])
ce = np.array(hfp_data["ce"])

# ============================================================
# Figure 1: Total Loss Curve
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, loss, color='#2563eb', linewidth=2, label='Total Loss')
ax.fill_between(steps, loss, alpha=0.1, color='#2563eb')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('HFP V2 Distillation — Total Loss Curve\n(Teacher: Qwen2.5-0.5B, Student: HFP Architecture)')
ax.legend()
ax.annotate(f'Start: {loss[0]:.2f}', xy=(steps[0], loss[0]),
            xytext=(150, loss[0]+0.3), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'End: {loss[-1]:.2f}', xy=(steps[-1], loss[-1]),
            xytext=(steps[-1]-300, loss[-1]+1.5), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray'))
plt.tight_layout()
plt.savefig('figures/01_total_loss.png')
plt.close()
print("Saved figures/01_total_loss.png")

# ============================================================
# Figure 2: KL Divergence + CE Loss (dual axis)
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 5))
color_kl = '#dc2626'
color_ce = '#16a34a'

ax1.plot(steps, kl, color=color_kl, linewidth=1.5, alpha=0.4, label='KL (raw)')
# Smoothed
window = 15
kl_smooth = np.convolve(kl, np.ones(window)/window, mode='valid')
ce_smooth = np.convolve(ce, np.ones(window)/window, mode='valid')
steps_smooth = steps[window-1:]

ax1.plot(steps_smooth, kl_smooth, color=color_kl, linewidth=2.5, label='KL Divergence (smoothed)')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('KL Divergence', color=color_kl)
ax1.tick_params(axis='y', labelcolor=color_kl)

ax2 = ax1.twinx()
ax2.plot(steps, ce, color=color_ce, linewidth=1.5, alpha=0.4, label='CE (raw)')
ax2.plot(steps_smooth, ce_smooth, color=color_ce, linewidth=2.5, label='Cross-Entropy (smoothed)')
ax2.set_ylabel('Cross-Entropy Loss', color=color_ce)
ax2.tick_params(axis='y', labelcolor=color_ce)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax1.set_title('HFP V2 Distillation — KL Divergence & Cross-Entropy Loss')
plt.tight_layout()
plt.savefig('figures/02_kl_ce_loss.png')
plt.close()
print("Saved figures/02_kl_ce_loss.png")

# ============================================================
# Figure 3: Loss Reduction Summary (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['Total Loss', 'KL Divergence', 'Cross-Entropy']
start_vals = [loss[0], kl[0], ce[0]]
end_vals = [loss[-1], kl[-1], ce[-1]]
reductions = [(s - e) / s * 100 for s, e in zip(start_vals, end_vals)]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, start_vals, width, label='Initial (Step 10)', color='#f87171', alpha=0.8)
bars2 = ax.bar(x + width/2, end_vals, width, label='Final (Step 1250)', color='#4ade80', alpha=0.8)

for i, (bar, red) in enumerate(zip(bars2, reductions)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'-{red:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Loss Value')
ax.set_title('HFP V2 Distillation — Loss Reduction Summary')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.savefig('figures/03_loss_reduction.png')
plt.close()
print("Saved figures/03_loss_reduction.png")

# ============================================================
# Figure 4: Training Phases Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Divide into 3 phases
phase1 = steps <= 200   # warmup + rapid descent
phase2 = (steps > 200) & (steps <= 700)  # steady descent
phase3 = steps > 700    # convergence

ax.plot(steps[phase1], loss[phase1], color='#ef4444', linewidth=2.5, label='Phase 1: Rapid Learning (1-200)')
ax.plot(steps[phase2], loss[phase2], color='#f59e0b', linewidth=2.5, label='Phase 2: Steady Descent (200-700)')
ax.plot(steps[phase3], loss[phase3], color='#22c55e', linewidth=2.5, label='Phase 3: Convergence (700-1250)')

# Phase boundaries
ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=700, color='gray', linestyle='--', alpha=0.5)

# Rate annotations
rate1 = (loss[phase1][0] - loss[phase1][-1]) / (steps[phase1][-1] - steps[phase1][0]) * 10
rate2 = (loss[phase2][0] - loss[phase2][-1]) / (steps[phase2][-1] - steps[phase2][0]) * 10
rate3 = (loss[phase3][0] - loss[phase3][-1]) / (steps[phase3][-1] - steps[phase3][0]) * 10

ax.text(100, 16.5, f'Rate: {rate1:.3f}/10 steps', fontsize=9, color='#ef4444', ha='center')
ax.text(450, 12.5, f'Rate: {rate2:.3f}/10 steps', fontsize=9, color='#f59e0b', ha='center')
ax.text(975, 10.8, f'Rate: {rate3:.3f}/10 steps', fontsize=9, color='#22c55e', ha='center')

ax.set_xlabel('Training Step')
ax.set_ylabel('Total Loss')
ax.set_title('HFP V2 Distillation — Training Phase Analysis')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures/04_training_phases.png')
plt.close()
print("Saved figures/04_training_phases.png")

# ============================================================
# Figure 5: Architecture Comparison Table (as figure)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['Component', 'Standard Transformer', 'HFP V2'],
    ['Attention', 'O(N² · D)', 'O(k² · D)  (k=N/8 → 64x reduction)'],
    ['High-freq Processing', '—', 'O((M-k) · D · κ)  (linear)'],
    ['FFT / IFFT', '—', 'O(N · D · log N)'],
    ['Total', 'O(N² · D)', 'O(k²D + ND·log N)'],
    ['Student Params', '494.0M', '552.1M  (58.1M HFP-specific)'],
    ['Training Speed', '—', '3.0 samples/s  (V100-SXM2)'],
    ['Final Loss (1 epoch)', 'TBD', '9.48'],
]

colors = [['#e0e7ff'] * 3] + [['white'] * 3] * (len(table_data) - 1)
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.22, 0.35, 0.43])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#3b82f6')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[i, j].set_facecolor('#f0f4ff')

ax.set_title('Architecture Comparison: Standard Transformer vs HFP V2', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/05_architecture_comparison.png')
plt.close()
print("Saved figures/05_architecture_comparison.png")

print("\nAll figures generated!")
