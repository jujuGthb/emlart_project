# Interactive Evolutionary Art System

An interactive web application that combines evolutionary algorithms with human selection to generate unique artwork through real-time collaboration between users and AI.

## Features

- **Interactive Evolution**: Guide art generation through visual selection
- **Hybrid AI-Human Evaluation**: Combines CLIP/LAION AI scoring with human preferences
- **Hardware Adaptive**: Runs on GPU, CPU, or basic mode without AI models
- **Real-time Analytics**: Track evolution progress with comprehensive statistics
- **Generation History**: Browse and navigate through evolution timelines

## Prerequisites

### System Requirements
- **Python**: Version 3.7 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 2GB free space

### Hardware Options
- **Full AI Mode**: CUDA GPU with 8GB+ VRAM (recommended)
- **CPU Mode**: Modern CPU (functional but slower)
- **Basic Mode**: Any hardware (no AI models required)

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/jujuGthb/emlart_project
cd emlart-gp-tutorial

2. **Creaate Virtual Environment**
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
3. **Install dependancies**

- For Full AI Project with AI Experience
pip install gradio numpy pillow matplotlib tensorflow
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install laion-aesthetics

- For basic or Testing mode
pip install gradio numpy pillow matplotlib tensorflow

## Usage
1. Run the Application
Full Version:

python emlart_gp.py

To test:
python test.py
2. Access the Interface
- Open Web browser
- Navigate to http://localhost:7860

3. Configure Evolution (Setup Tab)

- Set population size (10-200)

- Choose generations (1-100)

- Select image resolution (224x224 to 512x512)

- Enter text prompt

- Click "Initialize Enhanced Evolution"

4. Interact with Evolution (Evolution Tab)
5. Monitor Progress (Statistics Tab)

## Expected Output
Successful with AI Models:

Full CLIP + LAION aesthetic model loaded successfully on cuda
Running on local URL: http://localhost:7860

Test Mode:
CLIP not available - running in basic mode
Running on local URL: http://localhost:7860

## Project Structure After Running
interactive_evolution/
├── run_YYYYMMDD_HHMMSS/
│   ├── generations/
│   ├── selected/

## Troubleshooting

**Missing Dependencies:**
pip install [missing-package-name]

**Port Already in Use:**
demo.launch(server_port=7861)

## Documentation:
Project Report: Complete system documentation
