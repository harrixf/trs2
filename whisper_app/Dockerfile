# Usar una imagen base de Python
FROM python:3.9-slim

# Instalar dependencias del sistema para ffmpeg y otros
RUN apt-get update && apt-get install -y ffmpeg

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . .

# Instalar las dependencias de Python
RUN pip install -r requirements.txt

# Exponer el puerto en el que corre Flask
EXPOSE 5000

# Comando para correr la aplicación
CMD ["python", "app.py"]
