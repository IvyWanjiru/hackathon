"""Shared startup settings for Render and the existing Docker image."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"
workers = 1
threads = 4
timeout = 120
# Load the application once so signing keys also remain consistent if the
# service overrides the worker count. Database connections are request-scoped.
preload_app = True
accesslog = '-'
errorlog = '-'


def when_ready(server):
    from app import bootstrap_models
    bootstrap_models()


def post_worker_init(worker):
    from app import start_channel_listeners
    start_channel_listeners()
