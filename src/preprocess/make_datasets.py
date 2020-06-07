import argparse
import os
import random
import re

_DISPATCH = {}


def register_make(fn):
    fn_name = fn.__name__
    res = re.fullmatch(r'make\_(.*)', fn_name)
    assert res is not None, f'Function name \"{fn_name}\" does not match \"make_\" format'
    _DISPATCH[res.group(1).replace('_', '-')] = fn


def make_chunks(text, lengths):
    lines = [line.strip() for line in text.strip().split('\n')]
    assert sum(lengths) <= len(lines)

    splits = [0]
    for length in lengths:
        splits.append(splits[-1] + length)
    splits += [len(lines)]

    chunks = [
        lines[splits[idx]:splits[idx + 1]] for idx in range(len(splits) - 1)
    ]
    return chunks


@register_make
def make_covtype(data_path):
    text = open(data_path).read()
    data_dir = os.path.dirname(data_path)
    chunks = make_chunks(text, [11340, 3780])
    for name, chunk in zip(['covtype.train', 'covtype.dev', 'covtype.test'],
                           chunks):
        with open(os.path.join(data_dir, name), 'w') as out_f:
            out_f.write('\n'.join(chunk))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data setup paths and type')
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help='Path to data file')
    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        help='Type of dataset we\'re doing preprocessing on')

    args = parser.parse_args()

    _DISPATCH[args.mode](args.path)
