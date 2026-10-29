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
