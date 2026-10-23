import urllib.request
import zipfile

import pandas as pd

from .config import DATA_DIR


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction.")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    original_file_path.rename(data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df


def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def main():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = DATA_DIR / "sms_spam_collection.zip"
    extracted_path = DATA_DIR / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(
        data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv(DATA_DIR / "train.csv", index=None)
    validation_df.to_csv(DATA_DIR / "validation.csv", index=None)
    test_df.to_csv(DATA_DIR / "test.csv", index=None)
    print(f"Train: {len(train_df)}, Validation: {len(validation_df)}, "
          f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()
