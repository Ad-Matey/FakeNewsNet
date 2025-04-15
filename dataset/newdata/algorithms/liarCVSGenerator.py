import pandas as pd
import os

# Guide from discrete labels to binary labels
label_map = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0
}


def convert_liar_file(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t', header=None)
    df = df[[1, 2]]  
    df.columns = ['label_text', 'text']
    df['label'] = df['label_text'].map(label_map)
    df = df.dropna(subset=['label'])  
    df = df[['label', 'text']]
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


convert_liar_file("train.tsv", "liar_train_labeled.csv")
convert_liar_file("test.tsv", "liar_test_labeled.csv")
convert_liar_file("valid.tsv", "liar_valid_labeled.csv")
