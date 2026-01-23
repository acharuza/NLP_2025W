import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
INPUT_FILE = 'gpt5-wikipedia_dataset_evaluation.json'
OUT_IMG_1 = 'gpt5_wiki_accuracy_by_category.png'
OUT_IMG_2 = 'gpt5_wiki_length_vs_score.png'
PLOT_STYLE = 'whitegrid'
FIG_SIZE = (10, 5)
# =================================================

with open(INPUT_FILE) as f:
    df = pd.json_normalize(json.load(f)['results'])

# Extract category from sample_id (e.g., 'neutral' from 'wiki-neutral-1')
df['category'] = df['sample_id'].str.split('-').str[1]
df.loc[df['category'].str.isdigit(), 'category'] = 'factual'
print(df['category'])
df['len'] = df['model_response'].str.len()

# Handle potential None scores before converting to int
df.dropna(subset=['score'], inplace=True)
df['score'] = df['score'].astype(int)

# Count failures by category
failures_df = df[df['score'] == 0]
failure_counts = failures_df['category'].value_counts()

print("\n--- Failure Count by Category ---")
print(failure_counts)
print("---------------------------------\n")


sns.set_theme(style=PLOT_STYLE)

plt.figure(figsize=FIG_SIZE)
sns.barplot(df, x='category', y='score', palette='viridis', hue='category', legend=False).set(title='Accuracy by Category for the hallucination test', ylabel='Accuracy')
plt.savefig(OUT_IMG_1, bbox_inches='tight')

plt.figure(figsize=FIG_SIZE)
sns.boxplot(df, x='score', y='len', palette='coolwarm', hue='score', legend=False).set(title='Response Length Distribution for the hallucination test')
plt.savefig(OUT_IMG_2, bbox_inches='tight')