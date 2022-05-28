# coding=utf-8

import json
import sys

def main():
    data_file = sys.argv[1]
    with open(data_file, 'r', encoding='utf-8') as data_handle:
        for line in data_handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            print("{}\t{}".format(data["sentence"], data["label_desc"]))
    
if __name__ == "__main__":
    main()
