"""Command-line entry point: ``python -m ayurner serve | build-index | extract``."""

from __future__ import annotations

import argparse
import sys


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("the server needs extra dependencies: pip install 'ayurner[server]'", file=sys.stderr)
        return 1
    # Build (or load) the cached index once so every worker starts instantly.
    import ayurner

    ayurner.load()
    uvicorn.run(
        "ayurner.server.app:get_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0


def _build_index(args: argparse.Namespace) -> int:
    import ayurner

    ner = ayurner.load(lexicons=args.lexicons or None)
    print(f"index ready: {len(ner.index)} entries, lexicon_version={ner.lexicon_version}")
    if args.collisions:
        print(ner.index.collision_report())
    return 0


def _extract(args: argparse.Namespace) -> int:
    import ayurner

    ner = ayurner.load(lexicons=args.lexicons or None)
    src = open(args.input, encoding="utf-8") if args.input != "-" else sys.stdin
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        lines = (line.rstrip("\n") for line in src)
        for doc in ner.pipe(lines, n_process=args.n_process):
            out.write(doc.to_json() + "\n")
    finally:
        if src is not sys.stdin:
            src.close()
        if out is not sys.stdout:
            out.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="ayurner", description="Ayurvedic NER for Sanskrit / Hindi"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=_serve)

    p = sub.add_parser("build-index", help="compile and cache the lexicon index")
    p.add_argument("lexicons", nargs="*", help="extra lexicon files")
    p.add_argument("--collisions", action="store_true", help="print cross-label key collisions")
    p.set_defaults(func=_build_index)

    p = sub.add_parser("extract", help="tag one text per line, write JSONL")
    p.add_argument("input", help="input file (one text per line) or - for stdin")
    p.add_argument("-o", "--output")
    p.add_argument("-l", "--lexicons", nargs="*", default=[])
    p.add_argument("-j", "--n-process", type=int, default=1)
    p.set_defaults(func=_extract)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
