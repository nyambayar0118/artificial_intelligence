import pandas as pd
import numpy as np
# import math

def plurality_value(examples, target):
    counts = examples[target].value_counts()
    return counts.idxmax()

def entropy(examples, target):
    values, counts = np.unique(examples[target], return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def remainder(examples, attr, target, threshold=None):
    if threshold is None:
        # Ангилалтай үед
        values, counts = np.unique(examples[attr], return_counts=True)
        total = len(examples)
        rem = 0.0
        for v, c in zip(values, counts):
            subset = examples[examples[attr] == v]
            rem += (c / total) * entropy(subset, target)
        return rem
    else:
        # Тоон утгатай үед
        left = examples[examples[attr] <= threshold]
        right = examples[examples[attr] > threshold]
        total = len(examples)
        rem = (len(left)/total) * entropy(left, target) + (len(right)/total) * entropy(right, target)
        return rem

def information_gain(examples, attr, target):
    base_entropy = entropy(examples, target)
    if np.issubdtype(examples[attr].dtype, np.number):
        # Continuous үед хамгийн сайн threshold утгыг олно
        values = sorted(examples[attr].unique())
        best_gain, best_threshold = -1, None
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            gain = base_entropy - remainder(examples, attr, target, threshold)
            if gain > best_gain:
                best_gain, best_threshold = gain, threshold
        return best_gain, best_threshold
    else:
        gain = base_entropy - remainder(examples, attr, target)
        return gain, None

def decision_tree_learning(examples, attributes, target, parent_examples=None):
    if len(examples) == 0:
        return plurality_value(parent_examples, target)
    if len(np.unique(examples[target])) == 1:
        return np.unique(examples[target])[0]
    if len(attributes) == 0:
        return plurality_value(examples, target)
    
    gains, thresholds = {}, {}
    for attr in attributes:
        gain, threshold = information_gain(examples, attr, target)
        gains[attr] = gain
        thresholds[attr] = threshold

    best_attr = max(gains, key=gains.get)
    best_threshold = thresholds[best_attr]
    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]

    if best_threshold is not None:
        left = examples[examples[best_attr] <= best_threshold]
        right = examples[examples[best_attr] > best_threshold]
        tree[best_attr][f'<= {round(best_threshold, 2)}'] = decision_tree_learning(left, remaining_attrs, target, examples)
        tree[best_attr][f'> {round(best_threshold, 2)}'] = decision_tree_learning(right, remaining_attrs, target, examples)
    else:
        for v in np.unique(examples[best_attr]):
            exs = examples[examples[best_attr] == v]
            subtree = decision_tree_learning(exs, remaining_attrs, target, examples)
            tree[best_attr][v] = subtree
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample[attr]
    for condition, branch in tree[attr].items():
        if condition.startswith('<='):
            threshold = float(condition.split('<= ')[1])
            if value <= threshold:
                return predict(branch, sample)
        elif condition.startswith('>'):
            threshold = float(condition.split('> ')[1])
            if value > threshold:
                return predict(branch, sample)
        elif str(value) == str(condition):
            return predict(branch, sample)
    return None

# =========================================================
#   Decision tree printer
# =========================================================

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    for attr, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{attr} = {value}]")
            print_tree(subtree, indent + "   ")

# =========================================================
#   MAIN
# =========================================================

df = pd.read_csv("./data/loan_train.csv")
df.columns = df.columns.str.strip() 

# Ангилалтай болон тоон утгатай багануудын ангилал
categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Area', 'Status']
numeric_cols = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Term', 'Credit_History', 'Dependents']

# Category column to string
for col in categorical:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Numeric column to float
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Утга байхгүй байвал 0 болгоно  
        df[col] = df[col].fillna(0)                        

print("\nData types after cleaning:")
print(df.dtypes)

print("\nUnique sample values:")
for c in df.columns:
    print(f"{c}: {df[c].unique()[:5]}") 

target = 'Status'
attributes = [c for c in df.columns if c != target]

tree = decision_tree_learning(df, attributes, target)

print("\n--- Pretty Printed Decision Tree ---")
print_tree(tree)
