import json
import urllib.request

from .config import DATA_FILE_PATH, DATA_URL


def download_and_load_file(file_path, url):
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def split_data(data, train_frac=0.85, test_frac=0.1):
    train_portion = int(len(data) * train_frac)
    test_portion = int(len(data) * test_frac)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, val_data, test_data


def load_splits():
    data = download_and_load_file(DATA_FILE_PATH, DATA_URL)
    return split_data(data)


def main():
    train_data, val_data, test_data = load_splits()
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))


if __name__ == "__main__":
    main()
