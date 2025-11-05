input_file = r"Capstone-Project\pile_uncopyrighted_100k.txt"
output_file = r"Capstone-Project\pile_uncopyrighted_50MB.txt"

max_size_bytes = 50 * 1024 * 1024  # 50 MB
current_size = 0

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        line_size = len(line.encode("utf-8"))
        if current_size + line_size > max_size_bytes:
            break
        outfile.write(line)
        current_size += line_size

print(f"New file created: {output_file} ({current_size/1024/1024:.2f} MB)")