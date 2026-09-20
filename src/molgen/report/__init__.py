"""Run reporting: CSV tables and self-contained HTML."""

from .writer import write_csv, write_html, write_reports

__all__ = ["write_reports", "write_csv", "write_html"]
