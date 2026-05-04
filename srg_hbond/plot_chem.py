from __future__ import annotations
import argparse
from .figures import load_history, generate_png_report


def main():
    ap = argparse.ArgumentParser(description='Generate PNG figures from SRG-Chem frames/history JSON.')
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()
    hist = load_history(args.run_dir)
    written = generate_png_report(args.run_dir, hist)
    print('wrote figures:')
    for p in written:
        print(' ', p)

if __name__ == '__main__':
    main()
