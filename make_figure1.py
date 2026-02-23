import matplotlib.pyplot as plt

steps = [
    "Dataset\n(CSV)",
    "Categorical\nEncoding",
    "Feature Selection\n(RFE)",
    "Train–Test\nSplit",
    "Scaling\n(Min–Max)",
    "SMOTE\n(Balancing)",
    "Soft Voting\nModel",
    "Evaluation\n(CM + ROC)",
    "Streamlit\nTool",
]

plt.figure(figsize=(14, 3))
plt.axis("off")

x_start = 0.02
box_w = 0.1
box_h = 0.35
gap = 0.01
y = 0.5

for i, step in enumerate(steps):
    x = x_start + i * (box_w + gap)
    plt.gca().add_patch(
        plt.Rectangle((x, y - box_h / 2), box_w, box_h, fill=False)
    )
    plt.text(x + box_w / 2, y, step, ha="center", va="center", fontsize=9)

    if i < len(steps) - 1:
        plt.annotate(
            "",
            xy=(x + box_w + gap, y),
            xytext=(x + box_w, y),
            arrowprops=dict(arrowstyle="->")
        )

plt.title("Figure 1: End-to-end ML pipeline for ASD screening (prototype)", fontsize=12)
plt.tight_layout()
plt.savefig("Figure_1_ASD_Pipeline.png", dpi=300)
plt.close()

print("✅ Figure 1 saved as Figure_1_ASD_Pipeline.png")