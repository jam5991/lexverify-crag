"""
Structured Logging Configuration.

Sets up JSON-formatted structured logging via structlog for
observability and debugging of the CRAG pipeline.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, output logs as JSON. If False, use colored console output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    # Choose renderer based on output preference
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Structured logger instance.
    """
    return structlog.get_logger(name)
