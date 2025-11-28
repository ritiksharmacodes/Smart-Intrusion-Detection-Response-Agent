# 🚨 Smart-Intrusion-Detection-Response-Agent  
### Real-Time AI-Powered Behavior Analysis & Intrusion Detection System

A high-performance, real-time AI surveillance system designed to detect **loitering**, **suspicious pacing**, and **abnormal human behavior** using **YOLOv8** and advanced motion-analysis algorithms.  
Built for smart campuses, malls, offices, parking areas, and public safety ecosystems.

---

## 📌 Key Features

- 🔍 **Real-time person detection** using YOLOv8  
- 🎯 **Behavior recognition:**  
  - Loitering detection (🔴 red bounding box)  
  - Suspicious pacing (🟡 yellow bounding box)  
- ⚡ **Fast Flask-based video streaming** (low latency)  
- 📊 **Real-time dashboard** with live detections  
- 🧠 **Custom behavior algorithms** (time + movement analysis)  
- 🎥 **Web-based UI** with clean design  
- 🔧 **Lightweight & hardware-efficient**

---

## 🧠 How It Works (Architecture)

┌─────────────────────────┐
│ Live Camera Feed │
└─────────────┬───────────┘
│
▼
┌─────────────────────────┐
│ YOLOv8 Detector │
│ (Person Class Only) │
└─────────────┬───────────┘
│
▼
┌──────────────────────────────┐
│ Behavior Analysis Module │
│ - Loitering Timer │
│ - Pacing Movement Pattern │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ Colored Bounding Boxes │
│ (Red = Loitering, Yellow = Pacing)
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ Flask Live Stream Server │
└─────────────┬────────────────┘
│
▼
┌──────────────────────────────┐
│ Web Dashboard (UI) │
└──────────────────────────────┘


---

## 🛠️ Tech Stack

**Backend / AI**
- Python  
- Flask  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  

**Frontend**
- HTML  
- CSS  
- JavaScript  

---

## 📂 Project Structure

Smart-Intrusion-Detection-Response-Agent/
│
├── app.py # Main Flask app
├── detection/ # YOLO model & behavior logic
├── static/ # CSS, JS, icons
├── templates/ # HTML dashboard
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Installation & Setup (Simple Version)


```bash
1️⃣ Clone the repository

git clone https://github.com/ritiksharmacodes/Smart-Intrusion-Detection-Response-Agent.git
cd Smart-Intrusion-Detection-Response-Agent


2️⃣ Install dependencies
pip install -r requirements.txt


3️⃣ Run the application
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000

```
## **▶️ Usage**

- Launch the Flask server

- Open the live dashboard in your browser

- The system will automatically detect people

- Behavior detection is visualized as:

  - Red Box → Loitering

  - Yellow Box → Suspicious pacing

- View logs & FPS for debugging

## **🚀 Future Enhancements**

- Restricted-area intrusion alerts

- Multi-camera central dashboard

- Audio/visual alert integration

- Notification system (SMS/Email/WhatsApp)

- Cloud deployment support

## **🙌 Team Members**

Ritik Sharma

Navneet Singh Rawat


## **⭐ If you like this project, consider giving it a star on GitHub!**