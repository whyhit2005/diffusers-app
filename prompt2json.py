import os, sys
import argparse
import json
from pathlib import Path
import time, random

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example convert prompt to json.")
    parser.add_argument("--prompt_file", type=str,
                        default=None, required=True,
                        help="prompt file")
    parser.add_argument("--output", type=str,
                        default="prompt_out.json",
                        help="output path")
    parser.add_argument("--seed", type=int,
                        default=None,
                        help="seed")
    parser.add_argument("--replace_with", type=str,
                        default=None,
                        help="replace string")
    parser.add_argument("--replace_to", type=str,
                        default=None,
                        help="replace to string")
    return parser.parse_args(input_args)

def main(args):
    if args.seed is None:
        random.seed(time.time())
    else:
        random.seed(args.seed)
    replace_list = []
    if args.replace_with is not None:
        replace_list = args.replace_with.split(",")
    replace_to_list = []
    if args.replace_to is not None:
        replace_to_list = args.replace_to.split(",")
    if len(replace_list) != len(replace_to_list):
        raise ValueError("replace_with and replace_to must have the same length")
    
    prompt_file = Path(args.prompt_file)
    output = Path(args.output)
    plist = []
    with open(prompt_file, "r") as infile:
        for line in infile:
            line = line.strip()
            for i in range(len(replace_list)):
                line = line.replace(replace_list[i], replace_to_list[i])
            seed = random.randint(int(1e9), int(1e10)-1)
            t = {"prompt": line, "seed": seed}
            plist.append(t)
    with output.open("w") as outfile:
        for t in plist:
            json.dump(t, outfile)
            outfile.write("\n")
    return 0

if __name__ == "__main__":
    args = parse_args()
    ret = main(args)
    sys.exit(ret)