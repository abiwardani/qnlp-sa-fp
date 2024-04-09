names = ["dev", "test", "train"]

targets = ["good", "great", "bad", "awful", "tasty", "bland", "fast", "slow"]
replaces = ["amazing", "excellent", "horrible", "lousy", "yummy", "unpleasant", "kind", "poor"]

for name in names:
    with open(f"./restaurant_v4_{name}.txt", "r") as f:
        data = [line for line in f]
        if "\n" not in data[-1]:
            data[-1] += "\n"

    new_data = []

    for target, replace in zip(targets, replaces):
        new_data.extend([d.replace(target, replace) for d in data if target in d])

    with open(f"./restaurant_v5_{name}.txt", "w") as f:
        data.extend(new_data)
        data[-1] = data[-1].replace("\n", "")

        f.writelines(data)
    
    