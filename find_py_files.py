import os

def find_files(directory, extension):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                result.append(os.path.join(root, file))
    return result

if __name__ == "__main__":
    files = find_files(".", ".py")
    for file in files:
        print(file)
