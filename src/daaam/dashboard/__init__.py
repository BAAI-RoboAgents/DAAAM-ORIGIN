"""Local control and observability dashboard for DAAAM mapping workflows."""

from __future__ import annotations


def create_dashboard_app(*args, **kwargs):
    """Import FastAPI lazily so pure workflow readers keep no web dependency."""

    from .api import create_dashboard_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_dashboard_app"]
