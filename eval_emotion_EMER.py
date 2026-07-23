"""Backward-compatible EMER reasoning evaluation entry point."""

from minigpt4.evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main(default_task="reasoning"))
