# 🎨 Interactive Evolutionary Art System
An interactive web application that combines evolutionary algorithms with human selection to generate unique artwork through real-time collaboration between users and AI.

## ✨ Features
-  Interactive Evolution: Guide art generation through visual selection
-  Hybrid AI-Human Evaluation: Combines CLIP/LAION AI scoring with human preferences  
-  Hardware Adaptive: Runs on GPU, CPU, or basic mode without AI models
-  Real-time Analytics: Track evolution progress with comprehensive statistics
-  Generation History: Browse and navigate through evolution timelines

## 📋 Prerequisites
🖥 System Requirements:
| Component | Requirement |
|-----------|------------|
| Python | Version 3.7 or higher |
|  OS | Windows, macOS, or Linux |
|  Memory | 4GB RAM minimum (8GB+ recommended) |
|  Storage | 2GB free space |

🔧 Hardware Options:
| Mode | Requirements | Performance |
|------|-------------|-------------|
| 🚀 Full AI Mode | CUDA GPU with 8GB+ VRAM | ⭐ Recommended |
| ⚙️ CPU Mode | Modern CPU | ⭐ Functional but slower |
| 🔧 Basic Mode | Any hardware | ⭐ No AI models required |

##  Installation
1️⃣  Clone the Repository
git clone https://github.com/jujuGthb/emlart_project
cd emlart-gp-tutorial

2️⃣  Create Virtual Environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

3️⃣  Install Dependencies
 For Full AI Experience:
pip install gradio numpy pillow matplotlib tensorflow
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git

4️⃣  Download AI Models (Optional but Recommended)
 Download the pre-trained models from: https://github.com/cdvetal/emlart-gp-tutorial
- Follow the model installation instructions in the repository
- Place models in the appropriate directory as specified

⚡ For Basic/Testing Mode:
pip install gradio numpy pillow matplotlib tensorflow

## Usage
1️⃣ ▶️ Run the Application
python emlart_gp.py  # 🚀 Full Version
python test.py       # 🧪 Test Version

2️⃣ 🌐 Access the Interface
-  Open web browser
- 📍 Navigate to: http://localhost:7860

3️⃣ ⚙️ Configure Evolution (Setup Tab)
-  Set population size (10-200)
-  Choose generations (1-100)
- 🖼 Select image resolution (224x224 to 512x512)
-  Enter descriptive text prompt
-  Click "Initialize Enhanced Evolution"

4️⃣ 🎨 Interact with Evolution (Evolution Tab)
-  View generated population in grid layout
-  Click images you find appealing to select them
-  Click "Submit Selection & Next Generation" to evolve
-  Use navigation controls to browse evolution history

5️⃣ 📈 Monitor Progress (Statistics Tab)
-  View real-time evolution analytics
-  Track fitness progression and complexity metrics
-  Analyze selection patterns and trends

## 📤 Expected Output
| Status | Output |
|--------|---------|
| ✅ With AI Models | Full CLIP + LAION aesthetic model loaded successfully on cuda |
|  Test Mode | CLIP not available - running in basic mode |
|  Both Modes | Running on local URL: http://localhost:7860 |

## 📁 Project Structure After Running
 interactive_evolution/
├── 📂 run_YYYYMMDD_HHMMSS/
│   ├── generations/
│   ├── selected/


## 🛠️ Troubleshooting
| Issue | Solution |
|-------|----------|
| ❌ Missing Dependencies | pip install [missing-package-name] |
|  Port Already in Use | demo.launch(server_port=7861) |
| ⚠️ GPU Issues | System automatically falls back to CPU/basic mode |
|  Model Loading Issues | Download models from https://github.com/cdvetal/emlart-gp-tutorial |

💡 Performance Tips:
- Start with smaller population sizes (20-30)
-  Use lower resolutions for faster generation
-  Use test.py for systems without GPU

## ⚙️ Configuration Options
| Setting | Range | Description |
|---------|-------|-------------|
| Population Size | 10-200 | Number of individuals per generation |
|  Generations | 1-100 | Evolution cycles to run |
|  Resolution | 224x224, 256x256, 512x512 | Output image size |
| Text Prompts | Any text | Descriptive guidance for evolution |

## 🔬 Technical Stack
| Component | Technology |
|-----------|-----------|
|  Framework | TensorGP for genetic programming |
|  Interface | Gradio web application |
|  AI Models | CLIP + LAION for aesthetic scoring |
|  Visualization | Matplotlib for real-time statistics |

## 📚 Documentation
| Resource | Description |
|----------|-------------|
| 📄 Project Report | Complete system documentation (main branch) |
| 💻 Source Code | Detailed inline documentation and comments |
