FROM golang:1.20-alpine AS builder
RUN apk add --no-cache bash
WORKDIR /app
COPY . .
RUN go mod init api-gateway
RUN go mod tidy
RUN go build -o /api-gateway

FROM alpine:latest
COPY --from=builder /api-gateway /api-gateway
EXPOSE 3000
CMD ["/api-gateway"]
