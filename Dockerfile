FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8765
EXPOSE 8765

CMD ["sh", "-lc", "python3 render_dashboard_lite.py --base-summary-path ligand_outputs/results_summary.json --bo2-output-dir yield_bo_pyg_outputs --bo2-candidate-data data/bh-reactions.csv --host 0.0.0.0 --port ${PORT}"]
