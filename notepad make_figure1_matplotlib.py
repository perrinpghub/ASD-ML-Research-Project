import matplotlib.pyplot as plt

steps = [
    "Dataset (CSV)",
    "Categorical Encoding\n(LabelEncoder)",
    "Feature Selection\n(RFE Top 12)",
    "Train–Test Split\n(80/20 Stratified)",
    "Scaling\n(Min–Max)",
    "SMOTE\n(Train only)",
    "Soft Voting Model\n(SVM + NB)",
    "Evaluation\n(CM + ROC-AUC)",
    "Streamlit Deployment",
]

plt.figure(figsize=(10, 5))
plt.axis("off")

x0, y0 = 0.05, 0.5
box_w, box_h = 0.18, 0.18
gap = 0.02

for i, text in enumerate(steps):
    x = x0 + i * (box_w + gap)
    plt.gca().add_patch(plt.Rectangle((x, y0 - box_h/2), box_w, box_h, fill=False))
    plt.text(x + box_w/2, y0, text, ha="center", va="center", fontsize=9)

    if i < len(steps) - 1:
        x_next = x0 + (i + 1) * (box_w + gap)
        plt.annotate(
            "",
            xy=(x_next, y0),
            xytext=(x + box_w, y0),
            arrowprops=dict(arrowstyle="->")
        )

plt.title("Figure 1: End-to-end ML pipeline for ASD screening (prototype)")
plt.tight_layout()
plt.savefig("Figure_1_ASD_Pipeline.png", dpi=300)
plt.close()

print("✅ Saved: Figure_1_ASD_Pipeline.png")