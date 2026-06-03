FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

LABEL maintainer="NexionDeve"
LABEL description="cuda2hip-porting-kit"

WORKDIR /workspace/cuda2hip-porting-kit

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/port.py", "--help"]
