FROM golang:1.25.1-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o server .

FROM scratch
WORKDIR /app
COPY --from=builder /app/server /app/server
COPY --from=builder /app/frontend /app/frontend
EXPOSE 8080
ENTRYPOINT ["/app/server"]