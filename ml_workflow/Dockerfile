# Optional baseline inference deployment. Build from the PROJECT ROOT.
FROM python:3.12-slim
WORKDIR /srv/ghostdemand
COPY ml_workflow/requirements.txt /tmp/ml-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ml-requirements.txt gunicorn
COPY demand_ml.py ./
COPY ml_workflow ./ml_workflow
# Mount a trusted candidate directory read-only at /model when running.
ENV GHOST_REASON_CONFIG=/model/serving.json
USER 10001
EXPOSE 5061
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5061", "--timeout", "30", "ml_workflow.service:configured_app()"]
