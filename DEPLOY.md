# Hướng dẫn Deploy Anti-Cheat Service

## ✅ Có thể deploy được!

Service này đã được chuẩn bị để deploy với Docker. Service nhận frame và audio qua HTTP upload (không cần camera/microphone trực tiếp trên server).

## Cách Deploy

### 1. Build Docker Image

```bash
cd anti-cheat
docker build -t anti-cheat-service .
```

### 2. Chạy với Docker

```bash
# Chạy đơn giản
docker run -d \
  --name anti-cheat \
  -p 8081:8081 \
  -e ANTICHEAT_BEARER_TOKEN=your-secret-token \
  anti-cheat-service

# Hoặc không cần token (dev mode)
docker run -d \
  --name anti-cheat \
  -p 8081:8081 \
  anti-cheat-service
```

### 3. Chạy với Docker Compose

```bash
# Tạo file .env (optional)
echo "ANTICHEAT_BEARER_TOKEN=your-secret-token" > .env

# Chạy service
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng service
docker-compose down
```

### 4. Kiểm tra Health

```bash
curl http://localhost:8081/health
```

Kết quả mong đợi:
```json
{"status": "ok"}
```

## Deploy lên Production

### Railway / Render / Heroku

1. **Railway:**
   - Kết nối GitHub repo
   - Railway sẽ tự động detect Dockerfile
   - Set environment variable: `ANTICHEAT_BEARER_TOKEN`

2. **Render:**
   - Tạo Web Service
   - Chọn Dockerfile
   - Set port: `8081`
   - Set environment variables

3. **Heroku:**
   - Cần thêm `Procfile`:
     ```
     web: python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```

### Environment Variables

- `ANTICHEAT_BEARER_TOKEN` (optional): Token để bảo vệ API endpoints
- `PORT` (optional): Port để chạy service (default: 8081)

## Lưu ý

1. ✅ **Service không cần camera/microphone trên server** - nhận data qua HTTP upload
2. ✅ **OpenCV và MediaPipe** đã được cài đặt trong Dockerfile
3. ✅ **Health check** đã được cấu hình
4. ⚠️ **Voice detector** đang bị comment - có thể bật lại nếu cần
5. ⚠️ **CORS** hiện tại cho phép tất cả origins (`*`) - nên giới hạn trong production

## Test sau khi deploy

```bash
# Health check
curl http://your-server:8081/health

# Start session (với token)
curl -X POST http://your-server:8081/api/anti-cheat/session/start \
  -H "Authorization: Bearer your-secret-token"

# Response: {"sessionId": "session_xxx"}
```

## Troubleshooting

1. **Container không start:**
   ```bash
   docker logs anti-cheat
   ```

2. **Port đã được sử dụng:**
   - Đổi port trong docker-compose.yml: `"8082:8081"`

3. **Memory issues:**
   - MediaPipe và OpenCV cần memory
   - Tăng memory limit: `docker run --memory="512m" ...`

4. **Build fails:**
   - Kiểm tra internet connection
   - Kiểm tra Dockerfile có đúng không

