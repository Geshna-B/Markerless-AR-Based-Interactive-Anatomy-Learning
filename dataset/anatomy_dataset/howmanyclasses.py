import os
from collections import Counter

base_path = "D:/Desktop/6th Sem/CV/Project/dataset/anatomy_dataset/labels"
splits = ["train", "val", "test"]

class_names = ["heart", "brain", "lungs"]

for split in splits:
    label_dir = os.path.join(base_path, split)
    counter = Counter()

    if not os.path.exists(label_dir):
        print(f"\n{split.upper()} → folder not found")
        continue

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

    print(f"\n{split.upper()} SET:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {counter[i]}")
