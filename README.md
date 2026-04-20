# 🤟 SignVerse — Indian Sign Language Translator

SignVerse is a real-time Indian Sign Language (ISL) recognition system that translates hand gestures into text using computer vision and machine learning. The application is fully web-based and deployed, allowing users to access it directly from their mobile devices.

---

## 🌐 Live Demo

🔗 https://tor-2wum.vercel.app *(replace if needed)*

📱 Works seamlessly on mobile browsers.

---

## ✨ Features

* 🎥 Real-time hand gesture detection via webcam
* 🧠 Multiple ML models:

  * MLP (default)
  * Random Forest
  * KNN
* 📊 Live detection confidence tracking
* 🔄 Switch models dynamically from UI
* 🔤 Translation buffer to form words (e.g., HELLO)
* 🧾 Recent detection history with confidence %
* 📱 Fully responsive (mobile-friendly UI)
* 🟢 Connection status indicator

---

## 🖥️ UI Preview

### 📸 Main Interface

* Displays live camera feed
* Shows **"No Hand Detected"** when idle
* Highlights detection confidence and active model

### 🧠 Model Selection Panel

* Switch between:

  * MLP
  * Random Forest
  * KNN

### 🔤 Translation Buffer

* Builds words from detected gestures
* Includes:

  * Space
  * Delete
  * Clear

### 📊 Recent Detections

* Shows last recognized letters with confidence scores

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript (Deployed on Vercel)
* **Backend:** Flask (Python)
* **Machine Learning:**

  * MLP (Multi-Layer Perceptron)
  * Random Forest
  * KNN
* **Computer Vision:** MediaPipe (Hand Tracking)

---

## 🧠 How It Works

1. Captures real-time video from camera
2. Detects hand landmarks using MediaPipe
3. Converts landmarks into numerical feature vectors
4. Passes features to trained ML models
5. Predicts gesture → converts into text
6. Builds words using translation buffer

---

## 📊 Model Performance

| Model         | Description                                |
| ------------- | ------------------------------------------ |
| MLP           | Neural network-based, balanced performance |
| Random Forest | Highest accuracy, robust predictions       |
| KNN           | Simple and interpretable                   |

---

## 🎯 Use Cases

* Assist communication for deaf and mute individuals
* Real-time ISL learning tool
* Accessibility enhancement using AI

---

## 🔮 Future Improvements

* Dynamic gesture recognition (full words/sentences)
* Speech output (text-to-speech)
* More ISL vocabulary support
* Dark mode & UI enhancements
* Offline/mobile app version

---

## ⚙️ Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
python app.py
```

---

## 👨‍💻 Author

* Krinna Anandpara

---

## ⭐ Acknowledgements

* Google MediaPipe
* Open-source ML libraries
* ISL research community

---
