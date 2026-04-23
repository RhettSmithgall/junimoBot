import os

def remap_labels(folder):
    mapping = {
        "16": "0",
        "15": "1",
        "17": "2"
    }

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            # Replace first number if it's in mapping
            if parts[0] in mapping:
                parts[0] = mapping[parts[0]]

            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

remap_labels("trackTraining\\labels\\train")