FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8765
EXPOSE 8765

CMD ["sh", "-c", "python3 ligand_interactive_dashboard.py --model-dir ligand_outputs --excel-path \"Kraken monophosphine coordinates AD Descriptors.xlsx\" --host 0.0.0.0 --port ${PORT}"]
