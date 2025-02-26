import matplotlib.pyplot as plt
import pandas as pd

# Prepare data
data = {
    "Aspect": [
        "Handling Long Context",
        "Parallelization",
        "Pretraining Benefits",
        "Complexity",
        "Computational Efficiency",
        "Handling Out-of-Vocab Words",
        "Attention Mechanism",
        "Fine-tuning Flexibility",
        "Performance on Long Sequences",
        "Adaptability to Domain-Specific Tasks",
        "Evaluation Metrics (ROUGE, BLEU)",
        "Community and Library Support"
    ],
    "Transformers": [
        "Captures long-range dependencies effectively, ideal for long texts.",
        "Highly parallelizable; faster execution.",
        "Uses massive pretrained datasets for state-of-the-art results.",
        "Slightly complex but well-supported by libraries like Hugging Face.",
        "Requires more computational resources but performs better.",
        "Uses subword tokenization to handle rare or unseen words.",
        "Built-in self-attention assigns importance to parts of input text.",
        "Task-agnostic: Pretrained and fine-tuned for multiple tasks.",
        "Excels with self-attention, handling lengthy documents effectively.",
        "Easily fine-tuned on domain-specific datasets with pretrained weights.",
        "Achieves higher scores on metrics like ROUGE and BLEU.",
        "Strong community support and ready-to-use libraries."
    ],
    "Deep Learning Models (LSTM/GRU)": [
        "Struggles with long dependencies due to vanishing gradient problem.",
        "Sequential computation; slower and less scalable.",
        "Requires task-specific training from scratch, data-intensive.",
        "Simpler architecture, easier for smaller projects.",
        "Requires fewer resources but performs poorly on complex tasks.",
        "Fixed vocabulary; struggles with rare or unseen words.",
        "Attention mechanism increases complexity and overhead.",
        "Requires separate architectures and training for tasks.",
        "Limited memory retention; unsuitable for long documents.",
        "Extensive domain-specific training required from scratch.",
        "Performs worse on evaluation metrics compared to Transformers.",
        "Basic library support; lacks pretrained models for summarization."
    ]
}

df = pd.DataFrame(data)

# Create a table plot
fig, ax = plt.subplots(figsize=(22, 12))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#f7f7f7'] * 3
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2])

# Save the table as an image
plt.savefig("transformers_vs_lstm_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
