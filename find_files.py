import os

def find_files(directory, pattern):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern in file.lower():
                result.append(os.path.join(root, file))
    return result

if __name__ == "__main__":
    files = find_files(".", "fastscape")
    for file in files:
        print(file)
