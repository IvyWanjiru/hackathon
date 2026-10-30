---
title: GhostDemand PesaShield
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# GhostDemand AI

Capture retail enquiries, identify missed demand and organise follow-ups. Extraction and classification run on the backend; the dashboard shows business activity and evidence.

- [WhatsApp Business and Gmail live-demo setup](LIVE_DEMO_SETUP.md)
- [Automatic intake and other business-channel relays](BACKEND_AUTOMATION.md)

Run the app with `python app.py` or `gunicorn --config gunicorn.conf.py app:app` after installing `requirements.txt`. Open `/demo#activity` for synthetic examples or `/dashboard` for the live workspace. Configure private workspace access before connecting real customer channels, and keep `.env` out of Git.
