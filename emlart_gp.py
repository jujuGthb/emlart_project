from tensorgp.engine import *
import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import time
from typing import List, Tuple
import sys
import copy
import io


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
    
    
# Global variables for managing evolution state
current_engine = None
current_population = None
current_tensors = None
current_generation = 0
selected_indices = []
work_directory = None
grid_size = 16
text_features = None  # For CLIP text prompt
vit_model = None  # CLIP model
preprocess = None  # CLIP preprocessing
current_text_prompt = None  # Store current prompt globally
generation_history = {}  # Store all generations: {gen_num: {'population': pop, 'tensors': tensors, 'metadata': dict}}
current_display_generation = 0
navigation_enabled = False
stats_data = []  # Store statistics over time

def initialize_clip():
    """Initialize CLIP model if available"""
    global vit_model, preprocess
    if CLIP_AVAILABLE:
        vit_model, preprocess = clip.load("ViT-L-14", device=device)
        return True
    return False

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
            "fitness": float(ind['fitness']),
            "depth": int(ind['depth']),
            "nodes": int(ind['nodes']),
            "selected": idx in selected
        })
    
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

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

def user_guided_evaluation(**kwargs):
    """Fitness evaluation based on user selection and optionally text prompt"""
    global current_population, current_tensors, selected_indices, current_text_prompt
    
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    
    current_population = population
    current_tensors = tensors
    
    best_ind = 0
    best_fitness = 0
    
    # Calculate fitness for each individual
    for idx in range(len(population)):
        fitness = 0.01 * np.random.random()  # Base random fitness
        
        # Add user selection component
        if idx in selected_indices:
            fitness += 1.0
        
        # Add text prompt component if available
        if current_text_prompt:
            if CLIP_AVAILABLE and text_features is not None:
             
                clip_score = calculate_clip_similarity(tensors[idx])
                # Higher weight for generation 0 when no manual selection exists
                clip_weight = 1.0 if (current_generation == 0 and len(selected_indices) == 0) else 0.5
                fitness += clip_score * clip_weight
            else:
                # Use fallback text fitness function
                text_score = text_prompt_fitness(tensors[idx], current_text_prompt)
                # Higher weight for generation 0 when no manual selection exists
                text_weight = 0.8 if (current_generation == 0 and len(selected_indices) == 0) else 0.3
                fitness += text_score * text_weight
        
        population[idx]['fitness'] = fitness
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_ind = idx
    
    # First generation: if no selection, use text prompt or random
    if len(selected_indices) == 0 and current_generation == 0:
        for idx in range(len(population)):
            if current_text_prompt:
                if CLIP_AVAILABLE and text_features is not None:
                    fitness = calculate_clip_similarity(tensors[idx])
                else:
                    fitness = text_prompt_fitness(tensors[idx], current_text_prompt)
            else:
                fitness = np.random.random()
            
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
    global selected_indices, current_generation, current_engine, work_directory, current_text_prompt, navigation_enabled, current_display_generation
    
    # Don't allow submission when navigating history
    if navigation_enabled:
        return gallery_state, "Cannot submit selection while viewing historical generation. Return to current generation first.", get_current_images()
    
    if gallery_state is None:
        gallery_state = []
    
    selected_indices = gallery_state.copy()
    current_text_prompt = text_prompt  # Update global prompt
    
    # Save current generation to history before proceeding
    save_generation_to_history()
    
    # Save current generation data
    if current_tensors is not None and work_directory is not None:
        save_generation_images(current_tensors, current_generation, work_directory)
        if len(selected_indices) > 0:
            save_selected_images(current_tensors, selected_indices, current_generation, work_directory)
        save_generation_metadata(current_population, selected_indices, current_generation, work_directory, text_prompt)
    
    current_generation += 1
    current_display_generation = current_generation  # Keep them in sync
    
    # Run next generation if engine exists
    if current_engine is not None and current_generation <= current_engine.stop_value:
        status_msg = f"Generation {current_generation} starting..."
        new_gallery_state = []  # Reset selection
        images = get_current_images()
        return new_gallery_state, status_msg, images
    else:
        return [], "Evolution complete!", []

def initialize_evolution(pop_size, num_gens, resolution, seed, text_prompt):
    """Initialize the evolutionary engine"""
    global current_engine, current_generation, work_directory, current_text_prompt, current_display_generation, stats_data, generation_history
    
    # Clear previous evolution data
    stats_data = []  
    generation_history = {} 
    
    current_text_prompt = text_prompt
    
    # Initialize CLIP if available and prompt provided
    if text_prompt and CLIP_AVAILABLE:
        if not vit_model:
            initialize_clip()
        encode_text_prompt(text_prompt)
        print(f"Using CLIP guidance with prompt: {text_prompt}")
    elif text_prompt:
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
        "clip_available": CLIP_AVAILABLE
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
        elitism=max(4, int(pop_size) // 3),
        
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
    current_display_generation = 0  # Initialize display generation
    
    # Initialize population
    current_engine.experiment.set_generation_directory(0, lambda: False)
    population, best = current_engine.initialize_population(
        max_depth=current_engine.max_init_depth,
        min_depth=current_engine.min_init_depth,
        individuals=current_engine.population_size,
        method=current_engine.method,
        max_nodes=current_engine.max_nodes
    )
    
    # If we have a text prompt, evaluate initial population based on it
    if text_prompt:
        print("Evaluating initial population with text guidance...")
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
        if CLIP_AVAILABLE:
            status_msg += f"\nCLIP guidance: '{text_prompt}'"
        else:
            status_msg += f"\nBasic text guidance: '{text_prompt}'"
    
    return status_msg, get_current_images()
def run_single_generation():
    """Run a single generation of evolution"""
    global current_engine, selected_indices, current_tensors, current_population
    
    if current_engine is None or not current_engine.condition():
        return "Evolution complete or not initialized", []
    
    # Create new population from elite
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
    
    # Evaluate offspring only (elite already has fitness)
    if len(temp_population) > 0:
        temp_population, _ = current_engine.fitness_func_wrap(
            population=temp_population, 
            f_path=current_engine.experiment.current_directory
        )
    
    # Combine elite + offspring for the complete new population
    complete_population = new_population + temp_population
    current_engine.population = complete_population
    
    # Update current population reference for display
    current_population = complete_population
    
    # Calculate tensors for the COMPLETE population (elite + offspring)
    with tf.device(device):
        complete_tensors, _ = current_engine.calculate_tensors(complete_population)
    
    # Update global tensors for display
    current_tensors = complete_tensors
    
    # Update best individual
    current_engine.best = current_engine.get_n_best_from_pop(
        population=current_engine.population, n=1
    )[0]
    
    if current_engine.condition_overall(current_engine.best['fitness']):
        current_engine.best_overall = current_engine.deep_shallow_copy(current_engine.best)
    
    current_engine.current_generation += 1
    
    status_msg = f"Generation {current_engine.current_generation} complete"
    return status_msg, get_current_images()

def save_generation_to_history():
    """Save current generation to history for navigation"""
    global generation_history, current_generation, current_population, current_tensors
    
    if current_population is not None and current_tensors is not None:
        # Calculate statistics - fix the syntax errors
        fitness_values = [float(ind['fitness']) for ind in current_population]
        depth_values = [int(ind['depth']) for ind in current_population]
        nodes_values = [int(ind['nodes']) for ind in current_population]
        
        metadata = {
            'generation': current_generation,
            'selected_indices': selected_indices.copy(),
            'text_prompt': current_text_prompt,
            'fitness_stats': {
                'avg': float(np.mean(fitness_values)) if fitness_values else 0.0,
                'std': float(np.std(fitness_values)) if fitness_values else 0.0,
                'max': float(np.max(fitness_values)) if fitness_values else 0.0,
                'min': float(np.min(fitness_values)) if fitness_values else 0.0
            },
            'depth_stats': {
                'avg': float(np.mean(depth_values)) if depth_values else 0.0,
                'max': int(np.max(depth_values)) if depth_values else 0,
                'min': int(np.min(depth_values)) if depth_values else 0
            },
            'nodes_stats': {
                'avg': float(np.mean(nodes_values)) if nodes_values else 0.0,
                'max': int(np.max(nodes_values)) if nodes_values else 0,
                'min': int(np.min(nodes_values)) if nodes_values else 0
            },
            'timestamp': time.time()
        }
        
        generation_history[current_generation] = {
            'population': copy.deepcopy(current_population),
            'tensors': [tensor.numpy().copy() for tensor in current_tensors],
            'metadata': metadata
        }
        
        # Update global stats
        stats_data.append([
            current_generation,
            metadata['fitness_stats']['avg'],
            metadata['fitness_stats']['std'],
            metadata['fitness_stats']['max'],
            metadata['depth_stats']['avg'],
            metadata['nodes_stats']['avg'],
            len(selected_indices)
        ])

def navigate_to_generation(target_gen):
    """Navigate to a specific generation"""
    global current_display_generation, navigation_enabled, current_tensors
    
    if target_gen in generation_history:
        current_display_generation = target_gen
        navigation_enabled = True
        
        # Update current_tensors to show historical images
        stored_tensors = generation_history[target_gen]['tensors']
        current_tensors = [tf.convert_to_tensor(tensor_array) for tensor_array in stored_tensors]
        
        # Get images from history
        images = []
        for i, tensor_array in enumerate(stored_tensors):
            img_array = np.clip(tensor_array, 0, 255).astype(np.uint8)
            if len(img_array.shape) == 2:
                img = Image.fromarray(img_array, mode='L')
            else:
                img = Image.fromarray(img_array)
            images.append((img, f"Gen {target_gen} - Image {i}"))
        
        metadata = generation_history[target_gen]['metadata']
        status = f"Viewing Generation {target_gen} (Historical)"
        selection_status = f"Previous selections: {metadata['selected_indices']}"
        
        return images, status, selection_status, target_gen
    else:
        return [], f"Generation {target_gen} not found in history", "Selected: []", current_display_generation

def go_back_generation():
    """Navigate to previous generation"""
    global current_display_generation
    
    available_gens = sorted(generation_history.keys())
    if available_gens:
        current_idx = available_gens.index(current_display_generation) if current_display_generation in available_gens else len(available_gens) - 1
        if current_idx > 0:
            return navigate_to_generation(available_gens[current_idx - 1])
    
    return get_current_images(), f"Already at earliest generation ({current_display_generation})", "Selected: []", current_display_generation

def go_forward_generation():
    """Navigate to next generation"""
    global current_display_generation
    
    available_gens = sorted(generation_history.keys())
    if available_gens:
        current_idx = available_gens.index(current_display_generation) if current_display_generation in available_gens else 0
        if current_idx < len(available_gens) - 1:
            return navigate_to_generation(available_gens[current_idx + 1])
    
    return get_current_images(), f"Already at latest generation ({current_display_generation})", "Selected: []", current_display_generation

def go_to_current_generation():
    """Return to the current active generation"""
    global navigation_enabled, current_display_generation, current_tensors
    
    navigation_enabled = False
    current_display_generation = current_generation
    
    # Restore current generation tensors
    if current_engine and current_engine.population:
        with tf.device(device):
            current_tensors, _ = current_engine.calculate_tensors(current_engine.population)
    
    images = get_current_images()
    status = f"Returned to current generation {current_generation}"
    selection_status = f"Selected: {selected_indices}"
    
    return images, status, selection_status, current_generation

def get_statistics_plot():
    """Generate statistics plot"""
    if not stats_data:
        return None
    
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    generations = [row[0] for row in stats_data]
    fitness_avg = [row[1] for row in stats_data]
    fitness_std = [row[2] for row in stats_data]
    fitness_max = [row[3] for row in stats_data]
    depth_avg = [row[4] for row in stats_data]
    nodes_avg = [row[5] for row in stats_data]
    selections = [row[6] for row in stats_data]
    
    # Fitness evolution
    ax1.plot(generations, fitness_avg, label='Average', color='blue')
    ax1.plot(generations, fitness_max, label='Maximum', color='red')
    ax1.fill_between(generations, 
                     [avg - std for avg, std in zip(fitness_avg, fitness_std)],
                     [avg + std for avg, std in zip(fitness_avg, fitness_std)],
                     alpha=0.3, color='blue')
    ax1.set_title('Fitness Evolution')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(True)
    
    # Complexity evolution
    ax2.plot(generations, depth_avg, label='Avg Depth', color='green')
    ax2.plot(generations, nodes_avg, label='Avg Nodes', color='orange')
    ax2.set_title('Complexity Evolution')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Complexity')
    ax2.legend()
    ax2.grid(True)
    
    # Selection pressure
    ax3.bar(generations, selections, alpha=0.7, color='purple')
    ax3.set_title('Selection Pressure')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Number of Selected Individuals')
    ax3.grid(True)
    
    # Summary statistics
    if generations:
        total_gens = len(generations)
        avg_fitness_overall = np.mean(fitness_avg)
        max_fitness_overall = np.max(fitness_max)
        avg_selections = np.mean(selections)
        
        ax4.text(0.1, 0.8, f"Total Generations: {total_gens}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Avg Fitness: {avg_fitness_overall:.3f}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Best Fitness: {max_fitness_overall:.3f}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f"Avg Selections: {avg_selections:.1f}", transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64 for display
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)
    
    return buffer.getvalue()

def get_generation_history_table():
    """Get complete generation history table data"""
    if not stats_data:
        return []
    
    table_data = []
    for row in stats_data:
        table_data.append([
            int(row[0]),          # Generation
            f"{row[1]:.3f}",      # Avg Fitness
            f"{row[3]:.3f}",      # Max Fitness
            f"{row[4]:.1f}",      # Avg Depth
            f"{row[5]:.1f}",      # Avg Nodes
            int(row[6])           # Selections
        ])
    
    return table_data

# Create Gradio interface
with gr.Blocks(title="Interactive Evolutionary Art", css="""
    /* Main title styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #2d3748;
        font-weight: bold;
        font-size: 1.2em;
        margin: 15px 0 10px 0;
        padding: 8px 12px;
        background: linear-gradient(90deg, #e2e8f0 0%, #f7fafc 100%);
        border-left: 4px solid #4299e1;
        border-radius: 4px;
    }
    
    /* Keep gallery images at smaller consistent size */
    .gallery-item img {
        width: 120px !important;
        height: 120px !important;
        object-fit: cover !important;
        border-radius: 8px;
        transition: all 0.2s ease;
        border: 2px solid #e2e8f0;
    }
    
    /* Selection styling - green border */
    .gallery-item.selected {
        border: 3px solid #00ff00 !important;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.5) !important;
    }
    
    /* Hover effect */
    .gallery-item:hover img {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-color: #4299e1;
    }
    
    /* Ensure gallery container is more compact */
    .gallery-container {
        min-height: 400px;
        max-height: 500px;
        border-radius: 8px;
        background: #f8f9fa;
        padding: 10px;
    }
    
    /* Fix selection indicators */
    .gallery-item::after {
        content: "✓";
        position: absolute;
        top: 5px;
        right: 5px;
        background: #00ff00;
        color: white;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        opacity: 0;
        transition: opacity 0.2s ease;
        font-size: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .gallery-item.selected::after {
        opacity: 1;
    }
    
    /* Button styling */
    .nav-button {
        margin: 2px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Status boxes */
    .status-box {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px;
    }
    
    /* History table styling */
    .history-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling */
    .tab-nav button {
        font-weight: 500;
        border-radius: 6px 6px 0 0;
    }
    
    /* Instructions accordion */
    .instructions {
        background: #fef7e0;
        border: 1px solid #f6e05e;
        border-radius: 6px;
        margin-top: 15px;
    }
""") as demo:
    
    # Main header with styling
    gr.HTML("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5em;">🎨 Interactive Evolutionary Art System</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                Guide evolution by selecting favorites and using text descriptions
            </p>
        </div>
    """)
    
    with gr.Tab("🚀 Setup"):
        gr.HTML('<div class="section-header">🔧 Evolution Parameters</div>')
        
        with gr.Row():
            with gr.Column():
                pop_size_input = gr.Number(
                    value=10, 
                    label="👥 Population Size", 
                    minimum=4, 
                    maximum=100,
                    info="Number of individuals per generation"
                )
                num_gens_input = gr.Number(
                    value=30, 
                    label="🔄 Number of Generations", 
                    minimum=1, 
                    maximum=100,
                    info="Maximum generations to evolve"
                )
            with gr.Column():
                resolution_input = gr.Dropdown(
                    choices=["128x128", "256x256", "512x512"],
                    value="256x256",
                    label="📐 Image Resolution",
                    info="Higher resolution = better quality, slower processing"
                )
                seed_input = gr.Number(
                    value=1, 
                    label="🎲 Random Seed (optional)",
                    info="Use same seed for reproducible results"
                )
        
        gr.HTML('<div class="section-header">💭 Text Guidance</div>')
        
        text_prompt_input = gr.Textbox(
            label="📝 Text Prompt (REQUIRED for guided evolution)",
            placeholder="e.g., 'bright colorful patterns' or 'dark geometric shapes' or 'smooth flowing curves'",
            value="",
            info="Enter a description to guide the initial generation. Even without CLIP, basic text guidance will be used.",
            lines=2
        )
        
        with gr.Row():
            init_button = gr.Button("🚀 Initialize Evolution", variant="primary", size="lg")
            
        init_status = gr.Textbox(label="📊 Status", interactive=False, elem_classes=["status-box"])
    
    with gr.Tab("🧬 Evolution"):
        # Generation status at the top
        gr.HTML('<div class="section-header">📈 Generation Status</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                gen_status = gr.Textbox(
                    label="🔄 Generation Status", 
                    value="Not started", 
                    interactive=False,
                    elem_classes=["status-box"]
                )
            with gr.Column(scale=1):
                current_gen_display = gr.Number(
                    label="📊 Current Generation", 
                    value=0, 
                    interactive=False
                )
        
        # Main content area 
        with gr.Row():
            # Evolution panel (left side)
            with gr.Column(scale=3):
                gr.HTML('<div class="section-header">🎨 Current Generation</div>')
                
                gallery = gr.Gallery(
                    label="Evolution Gallery",
                    show_label=False,
                    elem_id="gallery",
                    columns=4,
                    rows=3,
                    object_fit="cover",
                    height=400,
                    interactive=True,
                    container=True,
                    allow_preview=True,
                    elem_classes=["gallery-container"]
                )
                
                gallery_state = gr.State([])
                selection_status = gr.Textbox(
                    label="✅ Selected Images", 
                    value="Selected: []", 
                    interactive=False,
                    elem_classes=["status-box"]
                )
                
                # Action buttons 
                gr.HTML('<div class="section-header">⚡ Evolution Actions</div>')
                with gr.Row():
                    submit_button = gr.Button(
                        "🚀 Submit Selection & Next Generation", 
                        variant="primary", 
                        scale=2
                    )
                    auto_run_button = gr.Button(
                        "🤖 Auto-select Best", 
                        variant="secondary",
                        scale=1
                    )
            
            # Generation History panel 
            with gr.Column(scale=2):
                gr.HTML('<div class="section-header">📊 Generation History</div>')
                with gr.Group(elem_classes=["history-panel"]):
                    history_table = gr.Dataframe(
                        headers=["Gen", "Avg Fitness", "Max Fitness", "Avg Depth", "Avg Nodes", "Selections"],
                        datatype=["number", "number", "number", "number", "number", "number"],
                        label="📈 Statistics Over Time",
                        interactive=False,
                        show_label=False
                    )
                
                # Navigation controls
                gr.HTML('<div class="section-header">🧭 Navigation Controls</div>')
                with gr.Column():
                    back_button = gr.Button(
                        "⬅️ Previous Generation", 
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="sm"
                    )
                    current_button = gr.Button(
                        "🎯 Current Generation", 
                        variant="primary",
                        elem_classes=["nav-button"],
                        size="sm"
                    )
                    forward_button = gr.Button(
                        "➡️ Next Generation", 
                        variant="secondary",
                        elem_classes=["nav-button"],
                        size="sm"
                    )
        
        # Instructions at the bottom
        with gr.Accordion("📖 Instructions", open=False, elem_classes=["instructions"]):
            gr.Markdown("""
                ### 🎯 How to Use:
                
                **1. Setup Phase:**
                - 🔧 Configure evolution parameters in the Setup tab
                - 📝 **Enter a text description** to guide evolution (required)
                - 🚀 Click "Initialize Evolution"
                
                **2. Evolution Phase:**
                - 🖱️ **Click images** to select your favorites (they'll show green borders)
                - 🚀 **Submit Selection** to evolve to the next generation
                - 🧭 Use **navigation buttons** (on the right) to view previous generations
                - 📊 Watch the **generation history table** update automatically
                
                **3. Tips:**
                - 🎨 The first generation uses your text prompt for guidance
                - 🔄 Later generations combine your selections + text similarity
                - 🤖 Use "Auto-select best" for hands-off evolution
                - 📈 Monitor progress in the statistics table
            """)
    
    with gr.Tab("📊 Detailed Statistics"):
        gr.HTML('<div class="section-header">📈 Evolution Analytics</div>')
        
        stats_plot = gr.Image(
            label="📊 Evolution Statistics Plots", 
            type="pil",
            show_label=True
        )
        
        gr.HTML('<div class="section-header">🔍 Detailed Analysis</div>')
        gr.Markdown("""
            ### 📊 Comprehensive Statistical Analysis
            
            This tab provides detailed visualizations of the evolutionary process:
            
            - **📈 Fitness Evolution:** Track average and maximum fitness over generations
            - **🧠 Complexity Evolution:** Monitor tree depth and node count trends  
            - **👆 Selection Pressure:** Visualize user selection patterns
            - **📋 Summary Statistics:** Overall evolution metrics and trends
            
            Statistics update automatically when new generations are created!
        """)
        
        # Refresh button at the bottom of statistics tab
        refresh_stats_button = gr.Button(
            "🔄 Refresh Statistics", 
            variant="primary"
        )
    
    # Helper function for statistics updates
    def update_statistics_if_data():
        """Update statistics plot if data exists"""
        if stats_data:
            plot_bytes = get_statistics_plot()
            if plot_bytes:
                return Image.open(io.BytesIO(plot_bytes))
        return None
    
    # Event handlers
    init_button.click(
        fn=initialize_evolution,
        inputs=[pop_size_input, num_gens_input, resolution_input, seed_input, text_prompt_input],
        outputs=[init_status, gallery]
    ).then(
        fn=lambda: (0, get_generation_history_table(), update_statistics_if_data()),
        outputs=[current_gen_display, history_table, stats_plot]
    )
    
    gallery.select(
        fn=on_image_select,
        inputs=[gallery_state],
        outputs=[gallery_state, selection_status]
    )
    
    def handle_submit(state, prompt):
        """Handle submit button click"""
        new_state, status, images = submit_selection(state, prompt)
        if "Cannot submit" not in status:
            gen_status, new_images = run_single_generation()
            stats_img = update_statistics_if_data()
            return new_state, gen_status, new_images, current_display_generation, get_generation_history_table(), stats_img
        else:
            return new_state, status, images, current_display_generation, get_generation_history_table(), None
    
    submit_button.click(
        fn=handle_submit,
        inputs=[gallery_state, text_prompt_input],
        outputs=[gallery_state, gen_status, gallery, current_gen_display, history_table, stats_plot]
    ).then(
        fn=lambda: ([], "Selected: []"),
        outputs=[gallery_state, selection_status]
    )
    
    def handle_auto_run(prompt):
        """Auto-select top individuals and run generation"""
        global current_population, current_text_prompt, navigation_enabled
        
        if navigation_enabled:
            return [], "Cannot auto-run while viewing historical generation. Return to current generation first.", [], current_display_generation, get_generation_history_table(), None
        
        current_text_prompt = prompt
        if current_population:
            n_select = max(1, len(current_population) // 4)
            auto_selected = list(range(n_select))
            new_state, status, images = submit_selection(auto_selected, prompt)
            if "Cannot submit" not in status:
                gen_status, new_images = run_single_generation()
                stats_img = update_statistics_if_data()
                return [], gen_status, new_images, current_display_generation, get_generation_history_table(), stats_img
        return [], "No population to evolve", [], current_display_generation, get_generation_history_table(), None
    
    auto_run_button.click(
        fn=handle_auto_run,
        inputs=[text_prompt_input],
        outputs=[gallery_state, gen_status, gallery, current_gen_display, history_table, stats_plot]
    )
    
    # Manual refresh button functionality 
    def refresh_detailed_statistics():
        """Refresh the detailed statistics display"""
        return update_statistics_if_data()
    
    refresh_stats_button.click(
        fn=refresh_detailed_statistics,
        outputs=[stats_plot]
    )
    
    # Navigation event handlers 
    def handle_back_generation():
        """Handle back button with proper image and generation display updates"""
        images, status, selection_status_text, gen_display = go_back_generation()
        stats_img = update_statistics_if_data()
        return images, status, selection_status_text, gen_display, get_generation_history_table(), stats_img
    
    def handle_forward_generation():
        """Handle forward button with proper image and generation display updates"""
        images, status, selection_status_text, gen_display = go_forward_generation()
        stats_img = update_statistics_if_data()
        return images, status, selection_status_text, gen_display, get_generation_history_table(), stats_img
    
    def handle_current_generation():
        """Handle current button with proper image and generation display updates"""
        images, status, selection_status_text, gen_display = go_to_current_generation()
        stats_img = update_statistics_if_data()
        return images, status, selection_status_text, gen_display, get_generation_history_table(), stats_img
    
    back_button.click(
        fn=handle_back_generation,
        outputs=[gallery, gen_status, selection_status, current_gen_display, history_table, stats_plot]
    )
    
    forward_button.click(
        fn=handle_forward_generation,
        outputs=[gallery, gen_status, selection_status, current_gen_display, history_table, stats_plot]
    )
    
    current_button.click(
        fn=handle_current_generation,
        outputs=[gallery, gen_status, selection_status, current_gen_display, history_table, stats_plot]
    )

# Command-line interface support
if __name__ == "__main__":
   
    if len(sys.argv) > 1:
        # Format: python emlart_gp.py [seed] [num_runs] [generations] ["prompt"]
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
    demo.launch(share=False, server_name="localhost", server_port=7860)