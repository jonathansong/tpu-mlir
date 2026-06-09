#!/usr/bin/env python3
"""Extract the IR printed immediately after one MLIR pass.

`tpu-mlir`'s tpuc-opt always runs its Deinit step before writing `-o`.  For the
Step 5 tiny LLM smoke we need the IR produced by convert-tpu-to-rxops itself,
because Deinit intentionally folds away unobserved temporary-buffer side effects.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpuc-opt', required=True)
    parser.add_argument('--pass-arg', required=True)
    parser.add_argument('--pass-name', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deinit-output', required=True)
    args = parser.parse_args()

    cmd = [
        args.tpuc_opt,
        args.pass_arg,
        '--mlir-disable-threading',
        f'--mlir-print-ir-after={args.pass_name}',
        '--mlir-print-ir-module-scope',
        args.input,
        '-o',
        args.deinit_output,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        return proc.returncode

    marker = f'// -----// IR Dump After '
    lines = proc.stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(marker) and f'({args.pass_name})' in line:
            start = i + 1
            break
    if start is None:
        sys.stderr.write(f'could not find IR dump after pass {args.pass_name}\n')
        sys.stdout.write(proc.stdout)
        return 1

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith('// -----// IR Dump After ') or \
           lines[i].startswith('// -----// IR Dump Before '):
            end = i
            break

    body = '\n'.join(lines[start:end]).strip() + '\n'
    if 'module ' not in body:
        sys.stderr.write(f'extracted dump after {args.pass_name} does not contain a module\n')
        sys.stdout.write(proc.stdout)
        return 1

    Path(args.output).write_text(body)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
