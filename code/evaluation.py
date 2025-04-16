import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def preprocess_input(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def evaluate_model(model, tokenizer, test_df):
    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(test_df), 16)):
        batch = test_df.iloc[i:i+16]
        inputs = preprocess_input(batch['title'].tolist(), tokenizer)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).tolist()
        labels = batch['label'].tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    return acc, precision, recall, f1

if __name__ == "__main__":
    model_choice = input("Evaluate which model? (p=politifact, g=gossipcop, b=both): ").strip().lower()

    if model_choice == 'p':
        model_path = "politifactModel/"
        test_csv = "data/processed/test.csv"  
    elif model_choice == 'g':
        model_path = "gossipcopModel/"
        test_csv = "data/processed/test.csv"
    elif model_choice == 'b':
        model_path = "bothModel/"
        test_csv = "data/processed/test.csv"
    else:
        raise ValueError("Invalid model choice")

    print(f"\nLoading model from {model_path} and test data from {test_csv}...\n")
    model, tokenizer = load_model_and_tokenizer(model_path)

    test_df = pd.read_csv(test_csv)

    print(f"Evaluating {len(test_df)} samples...")
    acc, precision, recall, f1 = evaluate_model(model, tokenizer, test_df)

    print("\n Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
# GossipCop Model
# Eval Loss: 0.3814
# Accuracy: 0.2959
# Precision: 0.2548
# Recall: 0.9591
# F1 Score: 0.4026
# Interpretation: Very high recall but very low precision and accuracy.

# PolitiFact Model
# Eval Loss: 0.3992
# Accuracy: 0.8425
# Precision: 0.6999
# Recall: 0.6359
# F1 Score: 0.6664
# Interpretation: Well balanced performance with decent precision and recall.

# Combined Model
# Eval Loss: 0.3887
# Accuracy: 0.8752
# Precision: 0.8335
# Recall: 0.6193
# F1 Score: 0.7106
# Interpretation: Best overall performance with high precision and accuracy; recall is slightly lower.
