#!/usr/bin/env python3
"""Render an effective run configuration.

The training entry point parses the command line FIRST and then applies the YAML
wholesale, so a command-line value is always overwritten by the file.  Anything
that has to override the file must therefore be merged into the file itself.
That is the whole job of this script: take one arm config, apply overrides, write
the effective config, and optionally describe or validate it.
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_value(text):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base arm config")
    ap.add_argument("--out", default="", help="where to write the effective config")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE", help="repeatable override")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the effective values")
    ap.add_argument("--require-paths", action="store_true",
                    help="fail unless the dataset directory exists")
    a = ap.parse_args()

    base = Path(a.config)
    if not base.is_file():
        print("error: config not found: %s" % base)
        return 2
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

    for item in a.set:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            print("error: --set expects KEY=VALUE, got %r" % item)
            return 2
        cfg[key] = {"desc": None, "value": parse_value(raw)}

    missing = [k for k in ("dataset", "source_path", "save_path") if k not in cfg]
    if missing:
        print("error: config lacks %s" % ", ".join(missing))
        return 2

    src = str(cfg["source_path"]["value"])
    dataset = str(cfg["dataset"]["value"])
    data_root = Path(src) / dataset
    if a.require_paths:
        if not Path(src).is_dir():
            print("error: source_path does not exist: %s" % src)
            return 3
        if not data_root.is_dir():
            print("error: dataset directory does not exist: %s" % data_root)
            print("       expected layout: <source_path>/<dataset>/images/...")
            return 3

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding="utf-8")
        print("effective config: %s" % out)

    if a.show:
        print("dataset      : %s  (%s)" % (dataset, data_root))
        for key in sorted(cfg):
            value = cfg[key]["value"] if isinstance(cfg[key], dict) else cfg[key]
            print("  %-30s %s" % (key, value))

    return 0


if __name__ == "__main__":
    sys.exit(main())
