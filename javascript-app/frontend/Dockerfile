FROM node:18-alpine
WORKDIR /app
RUN npm install -g http-server@14.1.1

# Copy static assets
COPY frontend/ /app/
COPY config.yaml /app/config.yaml

EXPOSE 4173
CMD ["http-server", "/app", "-p", "4173", "-c-1"]
