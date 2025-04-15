import pandas as pd

politifact_real = pd.read_csv("politifact_real.csv")
gossipcop_real = pd.read_csv("gossipcop_real.csv")
politifact_fake = pd.read_csv("politifact_fake.csv")
gossipcop_fake = pd.read_csv("gossipcop_fake.csv")

def label_and_save(df, label, filename):
    labeled = df[['title']].copy()
    labeled['label'] = label
    labeled.rename(columns={'title': 'text'}, inplace=True)
    labeled = labeled[['label', 'text']]
    labeled.to_csv(filename, index=False)
    print(f"Saved: {filename}")

label_and_save(politifact_real, 1, "politifact_real_labeled.csv")
label_and_save(gossipcop_real, 1, "gossipcop_real_labeled.csv")
label_and_save(politifact_fake, 0, "politifact_fake_labeled.csv")
label_and_save(gossipcop_fake, 0, "gossipcop_fake_labeled.csv")
