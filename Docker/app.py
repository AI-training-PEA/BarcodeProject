from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# โหลดโมเดล
try:
    model = YOLO('best.pt') 
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    # ใช้โมเดลมาตรฐานแก้ขัดไปก่อนถ้าหาไฟล์ไม่เจอ (สำหรับการทดสอบ)
    model = YOLO('yolov8n.pt') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    results_data = []

    for file in files:
        if file.filename == '':
            continue
            
        try:
            # 1. อ่านและประมวลผลภาพ
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # 2. ให้ AI ตรวจจับ
            results = model.predict(img, conf=0.4)
            result = results[0]
            
            # 3. นับจำนวนสิ่งที่เจอ (สมมติว่าเป็นบาร์โค้ด)
            detections_count = len(result.boxes)
            
            # 4. แปลงภาพผลลัพธ์เป็น Base64 (เพื่อส่งกลับไปโชว์หน้าเว็บโดยไม่ต้องเซฟลงเครื่อง)
            res_plotted = result.plot()
            res_plotted_rgb = res_plotted[..., ::-1]
            res_pil = Image.fromarray(res_plotted_rgb)
            
            buffered = io.BytesIO()
            res_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 5. เก็บข้อมูลลง List
            results_data.append({
                'filename': file.filename,
                'count': detections_count,
                'image_data': img_str,
                'status': '✅ ผ่าน' if detections_count > 0 else '❌ ไม่พบ'
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results_data.append({
                'filename': file.filename,
                'count': 0,
                'image_data': None,
                'status': '⚠️ Error'
            })

    # ส่งข้อมูลกลับเป็น JSON ให้หน้าเว็บวาดตาราง
    return jsonify({'results': results_data})

if __name__ == '__main__':
    app.run(debug=True, port=5000)