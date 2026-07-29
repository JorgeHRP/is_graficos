# Usa a versão exata do Python do seu projeto
FROM python:3.12.7-slim

# Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# Copia apenas o arquivo de dependências primeiro (para otimizar o cache)
COPY requirements.txt .

# Instala as dependências sem guardar cache para deixar a imagem mais leve
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da sua aplicação para dentro do contêiner
COPY . .

# Expõe a porta que a aplicação vai usar
EXPOSE 80

# Comando exato do seu Procfile
CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:80", "app:app"]