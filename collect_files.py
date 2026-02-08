#!/usr/bin/env python3
"""
collect_files.py

Collects the contents of all files in the current folder into
`all_files_contents.txt` so you can copy or move them elsewhere.

Usage:
  python collect_files.py        # non-recursive (top-level files only)
  python collect_files.py --recursive

The script skips itself and the output file, and avoids writing binary files.
"""
import os
import sys

OUTPUT = "all_files_contents.txt"
SKIP = {OUTPUT, os.path.basename(__file__)}

def is_text_file(path):
    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
            if b'\0' in chunk:
                return False
    except Exception:
        return False
    return True

def gather_files(root='.', recursive=False):
    files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                files.append(p)
    return sorted(files)

def main():
    recursive = '--recursive' in sys.argv[1:]
    files = gather_files('.', recursive=recursive)

    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for path in files:
            name = os.path.relpath(path, '.')
            if os.path.basename(path) in SKIP:
                continue
            if not is_text_file(path):
                print(f"Skipping binary: {name}")
                continue

            print(f"Adding: {name}")
            out.write(f"\n{'='*80}\n")
            out.write(f"FILE: {name}\n")
            out.write(f"{'='*80}\n\n")

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"[ERROR reading file: {e}]\n")

            out.write("\n\n")

    print(f"\nDone! All contents written to: {OUTPUT}")

if __name__ == '__main__':
    main()
