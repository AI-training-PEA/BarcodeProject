from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

# เริ่มต้นแอป Flask
app = Flask(__name__)

# --- ส่วนตั้งค่า AI ---
MODEL_PATH = 'best.pt'  # ชื่อไฟล์โมเดลของคุณ
print(f"⏳ กำลังโหลดโมเดล AI จาก: {MODEL_PATH} ...")
try:
    # โหลดโมเดลเตรียมไว้แค่ครั้งเดียวตอนเริ่มแอป
    model = YOLO(MODEL_PATH)
    print("✅ โหลดโมเดลสำเร็จ พร้อมใช้งาน!")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    print("กรุณาตรวจสอบว่าไฟล์ best.pt วางอยู่ถูกที่หรือไม่")
    exit(1) # ปิดโปรแกรมถ้าโหลดโมเดลไม่ได้

# --- ส่วนเส้นทางเว็บ (Routes) ---

@app.route('/')
def home():
    """แสดงหน้าเว็บหลัก (index.html)"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API สำหรับรับรูปภาพและส่งคืนผลลัพธ์"""
    if 'file' not in request.files:
        return "ไม่พบไฟล์ที่อัปโหลด", 400
    
    file = request.files['file']
    if file.filename == '':
        return "ไม่ได้เลือกไฟล์", 400

    try:
        # 1. อ่านไฟล์รูปภาพที่อัปโหลดเข้ามา
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # 2. ส่งให้ AI ตรวจจับ (ปรับค่า conf=0.4 คือความมั่นใจ 40% ขึ้นไปถึงจะโชว์)
        results = model.predict(img, conf=0.4)

        # 3. ดึงรูปภาพผลลัพธ์ที่วาดกรอบแล้วออกมา (อยู่ในรูปแบบ numpy array - BGR)
        result_array_bgr = results[0].plot()
        
        # 4. แปลงจาก BGR เป็น RGB เพื่อให้สีถูกต้องสำหรับเว็บ
        result_array_rgb = result_array_bgr[..., ::-1]
        result_img_pil = Image.fromarray(result_array_rgb)

        # 5. แปลงรูปภาพกลับเป็น Bytes เพื่อส่งกลับไปให้หน้าเว็บ
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)

        # ส่งไฟล์รูปภาพกลับไปเป็น response
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {e}", 500

# สำหรับรันตอนพัฒนา (ถ้าใช้ Gunicorn ใน Docker บรรทัดนี้จะไม่ถูกเรียก)
if __name__ == '__main__':
    app.run(debug=True, port=5000)