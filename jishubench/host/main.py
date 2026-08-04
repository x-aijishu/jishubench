"""Console entry point: jishubench <command>."""

from __future__ import annotations

from host.cli.commands import HANDLERS
from host.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
