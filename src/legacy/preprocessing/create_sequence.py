import os
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/paysim/paysim.csv"
SEQ_LEN = 2


def encode_tx_type(tx_type):
    mapping = {
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "CASH_IN": 3,
        "DEBIT": 4
    }
    return mapping.get(tx_type, -1)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, nrows=100000)

    print("Encoding transaction types...")
    df["type_encoded"] = df["type"].apply(encode_tx_type)

    sequences = []
    labels = []

    print("Building account sequences...")
    grouped = df.groupby("nameOrig")

    for account, group in grouped:
        group = group.sort_values("step")

        seq = group[["amount", "type_encoded", "step"]].values

    if len(seq) < SEQ_LEN:
        
      padding = np.zeros((SEQ_LEN - len(seq), 3))
      seq = np.vstack([seq, padding])
    else:
        seq = seq[:SEQ_LEN]                            

        label = int(group["isFraud"].max())

        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/sequences.npy", sequences)
    np.save("data/processed/sequence_labels.npy", labels)

    print("Sequences shape:", sequences.shape)
    print("Labels shape:", labels.shape)
    print("Saved sequence dataset.")


if __name__ == "__main__":
    main()