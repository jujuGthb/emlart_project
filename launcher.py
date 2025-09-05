#!/usr/bin/env python3
"""
Interactive Evolutionary Art System Launcher
Simplified launcher for the interactive evolutionary art system
"""

import sys
import os
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import PIL
    except ImportError:
        missing.append("pillow")
    
    try:
        import tensorflow
    except ImportError:
        missing.append("tensorflow")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Please install with: pip install {' '.join(missing)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Launch Interactive Evolutionary Art System')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--population', type=int, default=16, help='Population size')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--resolution', type=str, default='256x256', help='Image resolution (e.g., 256x256)')
    parser.add_argument('--no-clip', action='store_true', help='Run without CLIP model (pure interactive)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import and configure based on mode
    if args.no_clip:
        print("Running in pure interactive mode (no CLIP model)")
        # Import the interactive version without CLIP dependencies
        import interactive_evo_art_simple as evo_art
    else:
        # Check for CLIP dependencies
        try:
            import clip
            import torch
            import emlart_gp as evo_art
        except ImportError:
            print("CLIP/Torch not found. Running in simple mode.")
            import emlart_gp as evo_art
    
    # Launch the interface
    print(f"Starting Interactive Evolutionary Art System...")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Resolution: {args.resolution}")
    print(f"Port: {args.port}")
    
    # Initialize and launch
    evo_art.demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()