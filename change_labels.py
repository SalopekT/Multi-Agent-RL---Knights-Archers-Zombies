import os

label_folder = "C:\Users\tinsa\KULeuven\ml-project-2025-2026-main\ml-project-2025-2026-main\dataset\labels\train" 

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_folder, filename)

        new_lines = []

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) >= 5:
                cls = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                w = parts[3]
                h = parts[4]

                # convert coordinates
                x = (((x*1280) / 256 ) + 1) /2
                y = (((y*720) / 256 ) + 1) /2

                new_line = f"{cls} {x} {y} {w} {h}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("Done converting labels.")