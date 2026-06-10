FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8765
EXPOSE 8765

RUN if [ ! -f yield_bo_pyg_outputs/meta.json ]; then python3 yield_bo_rdkit_pyg_pipeline.py train --data-path data/bh-reactions.csv --output-dir yield_bo_pyg_outputs --reaction-id 0 --kfolds 2 --n-bootstrap 6 --gnn-epochs 4 --bo-iter 4 --top-k 8; fi

CMD ["sh", "-lc", "python3 ligand_interactive_dashboard.py --model-dir ligand_outputs --excel-path \"Kraken monophosphine coordinates AD Descriptors.xlsx\" --bo2-output-dir yield_bo_pyg_outputs --bo2-candidate-data data/bh-reactions.csv --host 0.0.0.0 --port ${PORT}"]
