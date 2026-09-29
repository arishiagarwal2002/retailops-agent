FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "streamlit>=1.64" "requests>=2.34"
COPY ui/ ui/
EXPOSE 8501
CMD ["sh", "-c", "streamlit run ui/streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
