from tensorgp.engine import *
import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import time
from typing import List, Tuple
import sys

# MODIFIED SECTION 1: Add LAION aesthetics import
try:
    from laion_aesthetics import MLP, normalizer, init_laion
    LAION_AVAILABLE = True
except ImportError:
    LAION_AVAILABLE = False
    print("LAION aesthetics model not available - using basic aesthetics fallback")

#Check device and CLIP availability 
CLIP_AVAILABLE = False
device = "cpu"  # Default to CPU

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CLIP loaded successfully - semantic text guidance available")
except Exception as e:
    print(f"CLIP not available ({type(e).__name__}): {e}")
    print("Running in pure interactive mode with basic text guidance.")
    CLIP_AVAILABLE = False
    device = "cpu"

# MODIFIED SECTION 2: Initialize both CLIP and LAION models
aesthetic_model = None
vit_model = None
preprocess = None

def initialize_models():
    """Initialize both CLIP and LAION aesthetic models if available"""
    global aesthetic_model, vit_model, preprocess
    
    if CLIP_AVAILABLE and LAION_AVAILABLE:
        try:
            aesthetic_model, vit_model, preprocess = init_laion(device)
            print(f"Both CLIP and LAION aesthetic models loaded successfully on {device}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    elif CLIP_AVAILABLE:
        try:
            vit_model, preprocess = clip.load("ViT-L-14", device=device)
            print("CLIP model loaded (no aesthetic model available)")
            return True
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            return False
    return False
    
# Global variables for managing evolution state
current_engine = None
current_population = None
current_tensors = None
current_generation = 0
selected_indices = []
work_directory = None
grid_size = 16
text_features = None  # For CLIP text prompt
current_text_prompt = None  # Store current prompt globally

def initialize_clip():
    """Initialize CLIP model if available"""
    return initialize_models()

def encode_text_prompt(prompt):
    """Encode text prompt using CLIP"""
    global text_features
    if CLIP_AVAILABLE and vit_model is not None and prompt:
        text_inputs = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_features = vit_model.encode_text(text_inputs)
        return True
    return False

def create_work_directory(base_dir="interactive_evolution"):
    """Create a timestamped work directory for this evolution run"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "generations"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "selected"), exist_ok=True)
    return work_dir

def save_generation_images(tensors, generation, directory):
    """Save all images from current generation to disk"""
    gen_dir = os.path.join(directory, "generations", f"gen_{generation:04d}")
    os.makedirs(gen_dir, exist_ok=True)
    
    image_paths = []
    for idx, tensor in enumerate(tensors):
        img_array = tensor.numpy()
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array)
        
        img_path = os.path.join(gen_dir, f"ind_{idx:04d}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths

def save_selected_images(tensors, indices, generation, directory):
    """Save selected images to a separate folder"""
    selected_dir = os.path.join(directory, "selected", f"gen_{generation:04d}")
    os.makedirs(selected_dir, exist_ok=True)
    
    for idx in indices:
        img_array = tensors[idx].numpy()
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array)
        
        img_path = os.path.join(selected_dir, f"selected_{idx:04d}.png")
        img.save(img_path)

def save_generation_metadata(population, selected, generation, directory, prompt=None):
    """Save metadata about the generation including expressions and fitness"""
    meta_file = os.path.join(directory, "generations", f"gen_{generation:04d}", "metadata.json")
    
    metadata = {
        "generation": generation,
        "population_size": len(population),
        "selected_indices": selected,
        "text_prompt": prompt,
        "individuals": []
    }
    
    for idx, ind in enumerate(population):
        metadata["individuals"].append({
            "index": idx,
            "expression": ind['tree'].get_str(),
            "fitness": ind['fitness'],
            "depth": ind['depth'],
            "nodes": ind['nodes'],
            "selected": idx in selected
        })
    
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

# MODIFIED SECTION 3: Enhanced fitness evaluation using original logic
def calculate_clip_similarity(tensor):
    """Calculate CLIP similarity between image and text prompt"""
    if not CLIP_AVAILABLE or text_features is None:
        return 0.0
    
    img_array = tensor.numpy()
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        pil_image = Image.fromarray(img_array, mode='L').convert('RGB')
    else:
        pil_image = Image.fromarray(img_array)
    
    image = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = vit_model.encode_image(image)
        similarity = torch.cosine_similarity(text_features, image_features, dim=-1).mean()
    
    return similarity.item()

def calculate_aesthetic_score(tensor):
    """Calculate LAION aesthetic score for the image"""
    if not (CLIP_AVAILABLE and LAION_AVAILABLE and aesthetic_model is not None):
        return 0.0
    
    img_array = tensor.numpy()
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        pil_image = Image.fromarray(img_array, mode='L').convert('RGB')
    else:
        pil_image = Image.fromarray(img_array)
    
    image = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = vit_model.encode_image(image)
        im_emb_arr = normalizer(image_features.cpu().detach().numpy())
        prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float))
        aesthetic_score = prediction.item()
    
    return aesthetic_score

def text_prompt_fitness(tensor, prompt_text):
    """
    Simple text-prompt fitness function that works without CLIP.
    This is a fallback that uses basic image properties to approximate text descriptions.
    """
    if not prompt_text:
        return 0.0
    
    # Convert tensor to numpy
    img_array = tensor.numpy()
    if len(img_array.shape) == 3:
        # Convert RGB to grayscale for analysis
        grayscale = np.mean(img_array, axis=2)
    else:
        grayscale = img_array
    
    # Normalize to 0-1 range
    normalized = grayscale / 255.0
    
    # Basic prompt-based scoring (very simplified)
    score = 0.0
    prompt_lower = prompt_text.lower()
    
    # Brightness-based keywords
    avg_brightness = np.mean(normalized)
    if any(word in prompt_lower for word in ['bright', 'light', 'white', 'sun']):
        score += avg_brightness
    elif any(word in prompt_lower for word in ['dark', 'black', 'night', 'shadow']):
        score += (1.0 - avg_brightness)
    
    # Contrast-based keywords
    contrast = np.std(normalized)
    if any(word in prompt_lower for word in ['sharp', 'contrast', 'edge', 'geometric']):
        score += contrast
    elif any(word in prompt_lower for word in ['smooth', 'soft', 'blur', 'gentle']):
        score += (1.0 - contrast)
    
    # Complexity-based keywords
    # Calculate edges using simple gradient
    grad_x = np.abs(np.gradient(normalized, axis=1))
    grad_y = np.abs(np.gradient(normalized, axis=0))
    edge_density = np.mean(grad_x + grad_y)
    
    if any(word in prompt_lower for word in ['complex', 'detailed', 'intricate', 'pattern']):
        score += edge_density
    elif any(word in prompt_lower for word in ['simple', 'minimal', 'clean', 'plain']):
        score += (1.0 - edge_density)
    
    # Add small random component to avoid ties
    score += 0.01 * np.random.random()
    
    return max(0.0, min(1.0, score))  # Clamp to [0,1]

# MODIFIED SECTION 4: Updated fitness evaluation following original EML-Art strategy
def user_guided_evaluation(**kwargs):
    """
    Enhanced fitness evaluation based on original EML-Art strategy:
    - User selection (interactive component)
    - CLIP semantic similarity (text-image alignment) 
    - LAION aesthetic quality assessment
    """
    global current_population, current_tensors, selected_indices, current_text_prompt
    
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    
    current_population = population
    current_tensors = tensors
    
    best_ind = 0
    best_fitness = 0
    
    # Calculate fitness for each individual using the original EML-Art approach
    for idx in range(len(population)):
        fitness = 0.01 * np.random.random()  # Base random fitness
        
        # Component 1: User selection (interactive guidance)
        user_selection_score = 0.0
        if idx in selected_indices:
            user_selection_score = 1.0
        
        # Component 2: CLIP semantic similarity (text-image alignment)
        clip_similarity_score = 0.0
        if current_text_prompt and CLIP_AVAILABLE and text_features is not None:
            clip_similarity_score = calculate_clip_similarity(tensors[idx])
        elif current_text_prompt:
            # Fallback to basic text fitness
            clip_similarity_score = text_prompt_fitness(tensors[idx], current_text_prompt)
        
        # Component 3: LAION aesthetic score (image quality)
        aesthetic_score = 0.0
        if CLIP_AVAILABLE and LAION_AVAILABLE and aesthetic_model is not None:
            aesthetic_score = calculate_aesthetic_score(tensors[idx])
            # Normalize aesthetic score (LAION typically ranges around 2-10)
            aesthetic_score = max(0.0, min(1.0, (aesthetic_score - 2.0) / 8.0))
        
        # MODIFIED SECTION 5: Fitness combination following original strategy
        if len(selected_indices) > 0:
            # When user has made selections, combine all three components
            # Following original: similarity + aesthetics/200 (but normalized here)
            fitness = (user_selection_score * 1.0 +           # User preference
                      clip_similarity_score * 0.8 +           # Semantic alignment  
                      aesthetic_score * 0.3)                  # Aesthetic quality
        else:
            # First generation or no user selection: focus on text + aesthetics
            if current_text_prompt:
                # Original strategy: use just similarity for text-only mode
                # But we enhance it with aesthetics
                fitness = (clip_similarity_score * 1.0 +     # Primary: semantic alignment
                          aesthetic_score * 0.2)              # Secondary: aesthetic quality
            else:
                # No prompt, no selection: pure aesthetic + random
                fitness = aesthetic_score * 0.5 + 0.5 * np.random.random()
        
        population[idx]['fitness'] = fitness
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_ind = idx
    
    return population, best_ind

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    img_array = tensor.numpy()
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array, mode='L')
    else:
        return Image.fromarray(img_array)

def get_current_images(start_idx=0):
    """Get current generation images for display"""
    global current_tensors
    
    if current_tensors is None:
        return []
    
    images = []
    end_idx = min(start_idx + grid_size, len(current_tensors))
    
    for i in range(start_idx, end_idx):
        img = tensor_to_pil(current_tensors[i])
        images.append((img, f"Image {i}"))
    
    return images

def on_image_select(evt: gr.SelectData, gallery_state):
    """Handle image selection in gallery"""
    if gallery_state is None:
        gallery_state = []
    
    # Toggle selection
    if evt.index in gallery_state:
        gallery_state.remove(evt.index)
    else:
        gallery_state.append(evt.index)
    
    return gallery_state, f"Selected: {gallery_state}"

def submit_selection(gallery_state, text_prompt):
    """Submit user selection and proceed to next generation"""
    global selected_indices, current_generation, current_engine, work_directory, current_text_prompt
    
    if gallery_state is None:
        gallery_state = []
    
    selected_indices = gallery_state.copy()
    current_text_prompt = text_prompt  # Update global prompt
    
    # Save current generation data
    if current_tensors is not None and work_directory is not None:
        save_generation_images(current_tensors, current_generation, work_directory)
        if len(selected_indices) > 0:
            save_selected_images(current_tensors, selected_indices, current_generation, work_directory)
        save_generation_metadata(current_population, selected_indices, current_generation, work_directory, text_prompt)
    
    current_generation += 1
    
    # Run next generation if engine exists
    if current_engine is not None and current_generation <= current_engine.stop_value:
        status_msg = f"Generation {current_generation} starting..."
        new_gallery_state = []  # Reset selection
        images = get_current_images()
        return new_gallery_state, status_msg, images
    else:
        return [], "Evolution complete!", []

# MODIFIED SECTION 6: Enhanced initialization with model loading
def initialize_evolution(pop_size, num_gens, resolution, seed, text_prompt):
    """Initialize the evolutionary engine with enhanced model support"""
    global current_engine, current_generation, work_directory, current_text_prompt
    
    # Store the text prompt globally
    current_text_prompt = text_prompt
    
    # Initialize models if available and prompt provided
    models_loaded = False
    if text_prompt:
        models_loaded = initialize_models()
        if models_loaded and CLIP_AVAILABLE:
            encode_text_prompt(text_prompt)
            if LAION_AVAILABLE:
                print(f"Using CLIP semantic similarity + LAION aesthetics with prompt: {text_prompt}")
            else:
                print(f"Using CLIP semantic similarity (no aesthetics) with prompt: {text_prompt}")
        else:
            print(f"Using basic text guidance with prompt: {text_prompt}")
    
    # Parse resolution
    res_parts = resolution.split('x')
    if len(res_parts) == 2:
        image_resolution = [int(res_parts[0]), int(res_parts[1]), 3]
    else:
        image_resolution = [256, 256, 3]
    
    # Create work directory
    work_directory = create_work_directory()
    
    # Save initial configuration
    config = {
        "population_size": int(pop_size),
        "generations": int(num_gens),
        "resolution": resolution,
        "seed": int(seed) if seed else None,
        "text_prompt": text_prompt,
        "clip_available": CLIP_AVAILABLE,
        "laion_available": LAION_AVAILABLE,
        "models_loaded": models_loaded
    }
    
    with open(os.path.join(work_directory, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Function set
    fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'log',
            'max', 'mdist', 'min', 'mult', 'neg', 'or', 'pow', 'sin', 'sqrt',
            'sub', 'tan', 'warp', 'xor'}
    
    # Create engine
    current_engine = Engine(
        fitness_func=user_guided_evaluation,
        population_size=int(pop_size),
        tournament_size=5,
        mutation_rate=0.9,
        crossover_rate=0.5,
        elitism=max(2, int(pop_size) // 10),
        
        terminal_prob=0.2,
        scalar_prob=0.50,
        uniform_scalar_prob=1,
        max_retries=20,
        
        mutation_funcs=[Engine.point_mutation, Engine.subtree_mutation, 
                       Engine.insert_mutation, Engine.delete_mutation],
        mutation_probs=[0.25, 0.3, 0.2, 0.25],
        
        method='ramped half-and-half',
        max_tree_depth=12,
        min_tree_depth=-1,
        min_init_depth=1,
        max_init_depth=6,
        
        bloat_control='weak',
        bloat_mode='depth',
        dynamic_limit=5,
        min_overall_size=1,
        max_overall_size=12,
        
        domain=[-1, 1],
        codomain=[-1, 1],
        do_final_transform=True,
        final_transform=[0, 255],
        
        stop_value=int(num_gens),
        effective_dims=3,
        seed=int(seed) if seed else None,
        operators=fset,
        debug=0,
        save_to_file=1,
        target_dims=image_resolution,
        objective='maximizing',
        device='/cpu:0',  # Force CPU device
        stop_criteria='generation',
        exp_prefix='interactive',
        save_image_pop=False,
        save_image_best=False,
        run_dir_path=work_directory
    )
    
    current_generation = 0
    
    # Initialize population
    current_engine.experiment.set_generation_directory(0, lambda: False)
    population, best = current_engine.initialize_population(
        max_depth=current_engine.max_init_depth,
        min_depth=current_engine.min_init_depth,
        individuals=current_engine.population_size,
        method=current_engine.method,
        max_nodes=current_engine.max_nodes
    )
    
    # Enhanced initial evaluation with models if available
    if text_prompt and models_loaded:
        print("Evaluating initial population with enhanced text + aesthetic guidance...")
        population, best = current_engine.fitness_func_wrap(
            population=population,
            f_path=current_engine.experiment.cur_image_directory
        )
        # Sort by fitness and keep best individuals  
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]
    
    current_engine.population = population
    current_engine.best = best
    current_engine.best_overall = current_engine.deep_shallow_copy(best)
    
    status_msg = f"Evolution initialized: {pop_size} individuals, {num_gens} generations"
    if text_prompt:
        if models_loaded and CLIP_AVAILABLE and LAION_AVAILABLE:
            status_msg += f"\nCLIP + LAION guidance: '{text_prompt}'"
        elif models_loaded and CLIP_AVAILABLE:
            status_msg += f"\nCLIP guidance: '{text_prompt}'"
        else:
            status_msg += f"\nBasic text guidance: '{text_prompt}'"
    
    return status_msg, get_current_images()

def run_single_generation():
    """Run a single generation of evolution"""
    global current_engine, selected_indices
    
    if current_engine is None or not current_engine.condition():
        return "Evolution complete or not initialized", []
    
    # Create new population
    new_population = current_engine.get_n_best_from_pop(
        population=current_engine.population, 
        n=current_engine.elitism
    )
    
    # Generate offspring
    temp_population = []
    for _ in range(current_engine.population_size - current_engine.elitism):
        indiv_temp, _, plist = current_engine.selection()
        member_depth, member_nodes = indiv_temp.get_depth()
        temp_population.append(
            new_individual(indiv_temp, fitness=0, depth=member_depth, 
                          nodes=member_nodes, valid=False, parents=plist)
        )
    
    # Evaluate new population
    if len(temp_population) > 0:
        temp_population, _ = current_engine.fitness_func_wrap(
            population=temp_population, 
            f_path=current_engine.experiment.current_directory
        )
    
    new_population += temp_population
    current_engine.population = new_population
    
    # Update best
    current_engine.best = current_engine.get_n_best_from_pop(
        population=current_engine.population, n=1
    )[0]
    
    if current_engine.condition_overall(current_engine.best['fitness']):
        current_engine.best_overall = current_engine.deep_shallow_copy(current_engine.best)
    
    current_engine.current_generation += 1
    
    return f"Generation {current_engine.current_generation} complete", get_current_images()

# Create Gradio interface
with gr.Blocks(title="Interactive Evolutionary Art") as demo:
    gr.Markdown("# Interactive EML-Art System")
    gr.Markdown("Guide evolution using text descriptions and manual selection, enhanced with CLIP semantic understanding and LAION aesthetic assessment")
    
    with gr.Tab("Setup"):
        with gr.Row():
            with gr.Column():
                pop_size_input = gr.Number(value=16, label="Population Size", minimum=4, maximum=100)
                num_gens_input = gr.Number(value=20, label="Number of Generations", minimum=1, maximum=100)
            with gr.Column():
                resolution_input = gr.Dropdown(
                    choices=["128x128", "256x256", "512x512"],
                    value="256x256",
                    label="Image Resolution"
                )
                seed_input = gr.Number(value=42, label="Random Seed (optional)")
        
        text_prompt_input = gr.Textbox(
            label="Text Prompt (Enhanced with CLIP + LAION)",
            placeholder="e.g., 'sunset with bright colors' or 'abstract geometric patterns' or 'serene landscape'",
            value="",
            info="Describe what you want to evolve. Uses CLIP for semantic understanding and LAION for aesthetic quality."
        )
        
        init_button = gr.Button("Initialize Evolution", variant="primary")
        init_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("Evolution"):
        gen_status = gr.Textbox(label="Generation Status", value="Not started", interactive=False)
        
        gallery = gr.Gallery(
            label="Current Generation",
            show_label=True,
            elem_id="gallery",
            columns=4,
            rows=4,
            object_fit="contain",
            height="auto",
            interactive=True
        )
        
        gallery_state = gr.State([])
        selection_status = gr.Textbox(label="Selected Images", value="Selected: []", interactive=False)
        
        with gr.Row():
            submit_button = gr.Button("Submit Selection & Next Generation", variant="primary")
            auto_run_button = gr.Button("Run Generation (Auto-select best)", variant="secondary")
        
        gr.Markdown("### Instructions:")
        gr.Markdown("1. **Enter a text description** to guide evolution with CLIP + LAION")
        gr.Markdown("2. Click images to select favorites (combines user preference + AI evaluation)")
        gr.Markdown("3. System uses: User Selection + CLIP Semantic Similarity + LAION Aesthetics")
        gr.Markdown("4. First generation evaluated primarily on text-image alignment + quality")
        gr.Markdown("5. Subsequent generations balance all three components")
    
    # Event handlers
    init_button.click(
        fn=initialize_evolution,
        inputs=[pop_size_input, num_gens_input, resolution_input, seed_input, text_prompt_input],
        outputs=[init_status, gallery]
    )
    
    gallery.select(
        fn=on_image_select,
        inputs=[gallery_state],
        outputs=[gallery_state, selection_status]
    )
    
    def handle_submit(state, prompt):
        """Handle submit button click"""
        new_state, status, images = submit_selection(state, prompt)
        gen_status, new_images = run_single_generation()
        return new_state, gen_status, new_images
    
    submit_button.click(
        fn=handle_submit,
        inputs=[gallery_state, text_prompt_input],
        outputs=[gallery_state, gen_status, gallery]
    ).then(
        fn=lambda: ([], "Selected: []"),
        outputs=[gallery_state, selection_status]
    )
    
    def handle_auto_run(prompt):
        """Auto-select top individuals and run generation"""
        global current_population, current_text_prompt
        current_text_prompt = prompt  # Update prompt
        if current_population:
            # Auto-select top 25% of population
            n_select = max(1, len(current_population) // 4)
            auto_selected = list(range(n_select))
            new_state, status, images = submit_selection(auto_selected, prompt)
            gen_status, new_images = run_single_generation()
            return [], gen_status, new_images
        return [], "No population to evolve", []
    
    auto_run_button.click(
        fn=handle_auto_run,
        inputs=[text_prompt_input],
        outputs=[gallery_state, gen_status, gallery]
    )

# Command-line interface support
if __name__ == "__main__":
    # Check if running from command line with arguments
    if len(sys.argv) > 1:
        # Parse command line arguments
        # Format: python script.py [seed] [num_runs] [generations] ["prompt"]
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        generations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        prompt = sys.argv[4] if len(sys.argv) > 4 else ""
        
        print(f"Command line mode:")
        print(f"  Seed: {seed}")
        print(f"  Runs: {num_runs}")
        print(f"  Generations: {generations}")
        print(f"  Prompt: {prompt}")
        
        # You could run automated evolution here if desired
        # For now, just set defaults and launch GUI
        
    # Launch Gradio interface
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)