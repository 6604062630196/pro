"""
Dataset Generation Script
- Dataset 1: Bank Customer Churn (Structured) — based on common churn dataset structure
- Dataset 2: Customer Support NLP (Unstructured Text) — sentiment on bank service
Both datasets are intentionally made incomplete for data preparation demonstration.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)


def generate_churn_dataset(n=1000):
    data = {
        'customer_id':         np.arange(1, n + 1),
        'credit_score':        np.random.normal(650, 100, n),
        'age':                 np.random.randint(18, 75, n).astype(float),
        'tenure':              np.random.randint(0, 11, n).astype(float),
        'balance':             np.random.exponential(50000, n),
        'num_products':        np.random.choice([1, 2, 3, 4], n, p=[0.35, 0.45, 0.15, 0.05]).astype(float),
        'has_credit_card':     np.random.choice([0, 1], n, p=[0.3, 0.7]).astype(float),
        'is_active_member':    np.random.choice([0, 1], n, p=[0.4, 0.6]).astype(float),
        'estimated_salary':    np.random.normal(100000, 45000, n),
        'geography':           np.random.choice(['France', 'Germany', 'Spain'], n, p=[0.5, 0.25, 0.25]),
        'gender':              np.random.choice(['Male', 'Female'], n),
    }

    df = pd.DataFrame(data)
    df['credit_score']     = df['credit_score'].clip(300, 850)
    df['balance']          = df['balance'].clip(0, 300000)
    df['estimated_salary'] = df['estimated_salary'].clip(10000, 250000)

    # Churn logic
    churn_score = (
        (df['credit_score'] < 550).astype(int) * 2 +
        (df['age'] > 55).astype(int) * 2 +
        (df['balance'] == 0).astype(int) +
        (df['num_products'] >= 3).astype(int) * 2 +
        (df['is_active_member'] == 0).astype(int) * 2 +
        (df['geography'] == 'Germany').astype(int) +
        np.random.randint(0, 3, n)
    )
    df['exited'] = (churn_score >= 5).astype(int)

    # Introduce missing values (~7-10% per column)
    for col in ['credit_score', 'age', 'balance', 'estimated_salary', 'tenure']:
        idx = np.random.choice(n, size=int(n * 0.08), replace=False)
        df.loc[idx, col] = np.nan

    # Introduce invalid zeros (credit_score = 0 is invalid)
    zero_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
    df.loc[zero_idx, 'credit_score'] = 0

    # Add duplicates (15 rows)
    dup_idx = np.random.choice(n, size=15, replace=False)
    df = pd.concat([df, df.iloc[dup_idx].assign(customer_id=range(n + 1, n + 16))], ignore_index=True)

    # Add outlier salary
    df.loc[np.random.choice(len(df), 8), 'estimated_salary'] = np.random.uniform(900000, 1500000, 8)

    return df


def generate_nlp_dataset(n=600):
    positive_texts = [
        "The customer service was excellent and very professional.",
        "I am very satisfied with the loan process, it was smooth and fast.",
        "Great experience! The staff resolved my issue within minutes.",
        "Online banking app works perfectly, love the new features.",
        "Quick response from the support team, highly recommend this bank.",
        "The interest rates are competitive and the account setup was easy.",
        "Friendly staff, efficient service. Will definitely stay with this bank.",
        "My mortgage was approved quickly. Very happy with the process.",
        "Transferred money internationally without any issues. Great service!",
        "Best bank I have ever used. Transparent fees and excellent support.",
        "The mobile app is intuitive and I can manage everything easily.",
        "Fast and reliable service. My card issue was resolved same day.",
        "Excellent communication throughout my loan application. Thank you!",
        "Really pleased with how the team handled my fraud complaint.",
    ]

    negative_texts = [
        "Horrible experience, waited 45 minutes just to speak to someone.",
        "The app keeps crashing and I cannot access my account.",
        "Hidden fees that were never disclosed at account opening. Disgusting.",
        "My card was blocked without any notification. Very frustrated.",
        "Customer service is rude and unhelpful. Will be switching banks.",
        "Loan rejection with no explanation given. Terrible service.",
        "Waited 3 weeks for a simple document. Completely unacceptable.",
        "The ATM ate my card and nobody could help me. Disaster.",
        "Overcharged three times this month. No resolution after two calls.",
        "Interest rate changed without warning. Feels like fraud.",
        "Staff at the branch were dismissive and made me feel unwelcome.",
        "Cannot reach anyone on the phone. Automated system is useless.",
        "Transfer failed but money was deducted. Still waiting for refund.",
        "Worst bank ever. Every interaction is a nightmare.",
    ]

    neutral_texts = [
        "Average bank, nothing special but gets the job done.",
        "Service is okay, not great but acceptable.",
        "Standard experience, similar to other banks.",
        "The branch is fine but the wait times could be shorter.",
        "Decent mobile app but missing some features I need.",
    ]

    texts, labels, categories, dates = [], [], [], []

    cats_pos = ['service', 'loan', 'mobile_app', 'fees', 'support']
    cats_neg = ['service', 'fees', 'app_issues', 'card_issues', 'loan']

    for i in range(n):
        r = np.random.random()
        if r < 0.48:
            txt = np.random.choice(positive_texts)
            suffix = np.random.choice(["", " Highly recommend!", " Five stars!", " Perfect!"])
            txt += suffix
            lbl = 1
            cat = np.random.choice(cats_pos)
        elif r < 0.88:
            txt = np.random.choice(negative_texts)
            suffix = np.random.choice(["", " Zero stars.", " Avoid this bank!", ""])
            txt += suffix
            lbl = 0
            cat = np.random.choice(cats_neg)
        else:
            txt = np.random.choice(neutral_texts)
            lbl = np.random.choice([0, 1])
            cat = np.random.choice(cats_pos + cats_neg)
        texts.append(txt)
        labels.append(lbl)
        categories.append(cat)
        dates.append(pd.Timestamp('2023-01-01') + pd.Timedelta(days=int(np.random.randint(0, 730))))

    df = pd.DataFrame({
        'review_id':   range(1, n + 1),
        'review_text': texts,
        'sentiment':   labels,
        'category':    categories,
        'date':        [d.strftime('%Y-%m-%d') for d in dates],
    })

    # Missing values
    for col, pct in [('category', 0.05), ('date', 0.03)]:
        idx = np.random.choice(n, size=int(n * pct), replace=False)
        df.loc[idx, col] = np.nan

    # Duplicate rows (12)
    dup_idx = np.random.choice(n, size=12, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    churn = generate_churn_dataset()
    churn.to_csv(os.path.join(base, 'churn_raw.csv'), index=False)
    print(f"✅ Churn dataset: {churn.shape}  | missing: {churn.isnull().sum().sum()} | duplicates: {churn.duplicated().sum()}")

    nlp = generate_nlp_dataset()
    nlp.to_csv(os.path.join(base, 'nlp_raw.csv'), index=False)
    print(f"✅ NLP dataset:   {nlp.shape}   | missing: {nlp.isnull().sum().sum()} | duplicates: {nlp.duplicated().sum()}")
