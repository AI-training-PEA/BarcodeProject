# ใช้ Python 3.10 ตัวเล็ก (Slim) เป็นฐาน
FROM python:3.10-slim

# --- [จุดที่แก้ไข] ---
# เปลี่ยน libgl1-mesa-glx เป็น libgl1
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ตั้งโฟลเดอร์ทำงานข้างใน Container
WORKDIR /app

# ก๊อปปี้ไฟล์รายการ Library ไปติดตั้งก่อน
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดที่เหลือทั้งหมด (รวมถึงโมเดล best.pt และโฟลเดอร์ templates) เข้าไป
COPY . .

# บอกว่า Container นี้จะเปิดพอร์ต 5000
EXPOSE 5000

# คำสั่งรัน Server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]