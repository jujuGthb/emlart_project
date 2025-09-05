from tensorgp.engine import *
import gradio as gr
import numpy as np
from PIL import Image
import os
import json
import time
import copy
from typing import List, Tuple, Dict, Any
import sys

# Model imports with fallbacks
try:
    from laion_aesthetics import MLP, normalizer, init_laion
    LAION_AVAILABLE = True
except ImportError:
    LAION_AVAILABLE = False
    print("LAION aesthetics model not available - using basic aesthetics fallback")

CLIP_AVAILABLE = False
device = "cpu"

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"CLIP loaded successfully on {device} - semantic text guidance available")
except Exception as e:
    print(f"CLIP not available ({type(e).__name__}): {e}")
    print("Running in pure interactive mode with basic text guidance.")
    CLIP_AVAILABLE = False
    device = "cpu"

# Global variables
aesthetic_model = None
vit_model = None
preprocess = None
current_engine = None
current_population = None
current_tensors = None
current_generation = 0
selected_indices = []
work_directory = None
grid_size = 16
text_features = None
current_text_prompt = None
generation_history = {}
model_status = "Not initialized"

def initialize_models():
    """Initialize both CLIP and LAION aesthetic models if available"""
    global aesthetic_model, vit_model, preprocess, model_status
    
    if CLIP_AVAILABLE and LAION_AVAILABLE:
        try:
            aesthetic_model, vit_model, preprocess = init_laion(device)
            model_status = f"✓ CLIP + LAION loaded on {device}"
            print(f"Both CLIP and LAION aesthetic models loaded successfully on {device}")
            return True
        except Exception as e:
            model_status = f"⚠ Model loading failed: {str(e)}"
            print(f"Error loading models: {e}")
            return False
    elif CLIP_AVAILABLE:
        try:
            vit_model, preprocess = clip.load("ViT-L-14", device=device)
            model_status = f"✓ CLIP loaded on {device} (no aesthetics)"
            print("CLIP model loaded (no aesthetic model available)")
            return True
        except Exception as e:
            model_status = f"⚠ CLIP loading failed: {str(e)}"
            print(f"Error loading CLIP: {e}")
            return False
    
    model_status = "CPU only (no models)"
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
    os.makedirs(os.path.join(work_dir, "exports"), exist_ok=True)
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

def save_generation_to_history():
    """Save current generation to history"""
    global generation_history, current_generation, current_population, current_tensors
    
    if current_population and current_tensors:
        generation_history[current_generation] = {
            'population': copy.deepcopy(current_population),
            'tensors': [tensor.numpy().copy() for tensor in current_tensors],
            'timestamp': time.time(),
            'selected_indices': selected_indices.copy() if selected_indices else []
        }

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
    """Simple text-prompt fitness function that works without CLIP"""
    if not prompt_text:
        return 0.0
    
    img_array = tensor.numpy()
    if len(img_array.shape) == 3:
        grayscale = np.mean(img_array, axis=2)
    else:
        grayscale = img_array
    
    normalized = grayscale / 255.0
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
    grad_x = np.abs(np.gradient(normalized, axis=1))
    grad_y = np.abs(np.gradient(normalized, axis=0))
    edge_density = np.mean(grad_x + grad_y)
    
    if any(word in prompt_lower for word in ['complex', 'detailed', 'intricate', 'pattern']):
        score += edge_density
    elif any(word in prompt_lower for word in ['simple', 'minimal', 'clean', 'plain']):
        score += (1.0 - edge_density)
    
    score += 0.01 * np.random.random()
    return max(0.0, min(1.0, score))

def user_guided_evaluation(**kwargs):
    """Enhanced fitness evaluation with user selection, CLIP similarity, and LAION aesthetics"""
    global current_population, current_tensors, selected_indices, current_text_prompt
    
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    
    current_population = population
    current_tensors = tensors
    
    best_ind = 0
    best_fitness = 0
    
    for idx in range(len(population)):
        fitness = 0.01 * np.random.random()
        
        # User selection component
        user_selection_score = 0.0
        if idx in selected_indices:
            user_selection_score = 1.0
        
        # CLIP semantic similarity
        clip_similarity_score = 0.0
        if current_text_prompt and CLIP_AVAILABLE and text_features is not None:
            clip_similarity_score = calculate_clip_similarity(tensors[idx])
        elif current_text_prompt:
            clip_similarity_score = text_prompt_fitness(tensors[idx], current_text_prompt)
        
        # LAION aesthetic score
        aesthetic_score = 0.0
        if CLIP_AVAILABLE and LAION_AVAILABLE and aesthetic_model is not None:
            aesthetic_score = calculate_aesthetic_score(tensors[idx])
            aesthetic_score = max(0.0, min(1.0, (aesthetic_score - 2.0) / 8.0))
        
        # Combine fitness components
        if len(selected_indices) > 0:
            fitness = (user_selection_score * 1.0 +
                      clip_similarity_score * 0.8 +
                      aesthetic_score * 0.3)
        else:
            if current_text_prompt:
                fitness = (clip_similarity_score * 1.0 +
                          aesthetic_score * 0.2)
            else:
                fitness = aesthetic_score * 0.5 + 0.5 * np.random.random()
        
        population[idx]['fitness'] = fitness
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_ind = idx
    
    return population, best_ind

def get_generation_stats():
    """Calculate and return current generation statistics"""
    global current_engine, current_population, current_generation
    
    if not current_population:
        return "No population data"
    
    fitnesses = [ind['fitness'] for ind in current_population]
    depths = [ind['depth'] for ind in current_population]
    
    stats = {
        'generation': current_generation,
        'best_fitness': max(fitnesses),
        'avg_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'avg_depth': np.mean(depths),
        'population_size': len(current_population)
    }
    
    return f"""Generation: {stats['generation']}
Best Fitness: {stats['best_fitness']:.3f}
Avg Fitness: {stats['avg_fitness']:.3f} ± {stats['std_fitness']:.3f}
Avg Depth: {stats['avg_depth']:.1f}
Population: {stats['population_size']}
Model Status: {model_status}"""

def get_image_details(evt: gr.SelectData):
    """Show expression and details for selected image"""
    global current_population
    
    if current_population and evt.index < len(current_population):
        ind = current_population[evt.index]
        details = f"""Image {evt.index}
Expression: {ind['tree'].get_str()}
Fitness: {ind['fitness']:.4f}
Depth: {ind['depth']}
Nodes: {ind['nodes']}
Selected: {evt.index in selected_indices}"""
        return details
    return "No data available"

def select_top_n(n, prompt):
    """Auto-select top N individuals by fitness"""
    global current_population, current_text_prompt
    
    if not current_population:
        return [], "No population"
    
    # Update prompt if changed
    if prompt != current_text_prompt:
        current_text_prompt = prompt
        if prompt and CLIP_AVAILABLE:
            encode_text_prompt(prompt)
    
    # Sort by fitness and select top N
    sorted_pop = sorted(enumerate(current_population), 
                       key=lambda x: x[1]['fitness'], reverse=True)
    n = min(n, len(sorted_pop))
    top_indices = [idx for idx, _ in sorted_pop[:n]]
    
    return top_indices, f"Auto-selected top {n}: {top_indices}"

def load_generation_from_history(gen_num):
    """Load a previous generation"""
    global current_population, current_tensors, current_generation, selected_indices
    
    gen_num = int(gen_num)
    if gen_num in generation_history:
        data = generation_history[gen_num]
        current_population = copy.deepcopy(data['population'])
        current_tensors = [tf.constant(arr) for arr in data['tensors']]
        current_generation = gen_num
        selected_indices = data.get('selected_indices', [])
        
        images = get_current_images()
        stats = get_generation_stats()
        return images, f"Loaded generation {gen_num}", stats, selected_indices
    
    return [], f"Generation {gen_num} not found", "", []

def export_results(export_type):
    """Export evolution results in various formats"""
    global work_directory, current_engine, generation_history, current_population
    
    if not work_directory:
        return "No evolution data to export"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(work_directory, "exports")
    
    try:
        if export_type == "best_image":
            if current_engine and current_engine.best_overall:
                best_ind = current_engine.best_overall
                tensor = best_ind['tensor']
                img_array = tensor.numpy()
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                if len(img_array.shape) == 2:
                    img = Image.fromarray(img_array, mode='L')
                else:
                    img = Image.fromarray(img_array)
                
                img_path = os.path.join(export_dir, f"best_individual_{timestamp}.png")
                img.save(img_path)
                return f"✓ Best image saved to {img_path}"
            else:
                return "No best individual available"
        
        elif export_type == "expressions":
            expr_path = os.path.join(export_dir, f"expressions_{timestamp}.txt")
            with open(expr_path, 'w') as f:
                f.write(f"Evolution Expressions Export - {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                
                for gen, data in sorted(generation_history.items()):
                    f.write(f"Generation {gen}:\n")
                    f.write("-" * 20 + "\n")
                    for i, ind in enumerate(data['population']):
                        selected_mark = " [SELECTED]" if i in data.get('selected_indices', []) else ""
                        f.write(f"  Individual {i}{selected_mark}:\n")
                        f.write(f"    Expression: {ind['tree'].get_str()}\n")
                        f.write(f"    Fitness: {ind['fitness']:.4f}\n")
                        f.write(f"    Depth: {ind['depth']}, Nodes: {ind['nodes']}\n\n")
                    f.write("\n")
                
                if current_engine and current_engine.best_overall:
                    f.write("BEST OVERALL:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Expression: {current_engine.best_overall['tree'].get_str()}\n")
                    f.write(f"Fitness: {current_engine.best_overall['fitness']:.4f}\n")
            
            return f"✓ Expressions saved to {expr_path}"
        
        elif export_type == "evolution_log":
            log_path = os.path.join(export_dir, f"evolution_log_{timestamp}.json")
            
            # Prepare serializable data
            serializable_history = {}
            for gen, data in generation_history.items():
                serializable_history[str(gen)] = {
                    'timestamp': data['timestamp'],
                    'selected_indices': data.get('selected_indices', []),
                    'population_stats': {
                        'size': len(data['population']),
                        'expressions': [ind['tree'].get_str() for ind in data['population']],
                        'fitnesses': [ind['fitness'] for ind in data['population']],
                        'depths': [ind['depth'] for ind in data['population']],
                        'nodes': [ind['nodes'] for ind in data['population']]
                    }
                }
            
            log_data = {
                'export_timestamp': timestamp,
                'config': {
                    'model_status': model_status,
                    'clip_available': CLIP_AVAILABLE,
                    'laion_available': LAION_AVAILABLE,
                    'device': device,
                    'current_prompt': current_text_prompt
                },
                'evolution_history': serializable_history,
                'final_generation': current_generation,
                'best_overall': {
                    'expression': current_engine.best_overall['tree'].get_str(),
                    'fitness': current_engine.best_overall['fitness']
                } if current_engine and current_engine.best_overall else None
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return f"✓ Evolution log saved to {log_path}"
        
        elif export_type == "current_generation":
            if current_tensors and current_population:
                gen_export_dir = os.path.join(export_dir, f"generation_{current_generation}_{timestamp}")
                os.makedirs(gen_export_dir, exist_ok=True)
                
                # Save all images
                for idx, tensor in enumerate(current_tensors):
                    img_array = tensor.numpy()
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    
                    if len(img_array.shape) == 2:
                        img = Image.fromarray(img_array, mode='L')
                    else:
                        img = Image.fromarray(img_array)
                    
                    selected_mark = "_SELECTED" if idx in selected_indices else ""
                    img_path = os.path.join(gen_export_dir, f"individual_{idx:03d}{selected_mark}.png")
                    img.save(img_path)
                
                # Save metadata
                metadata = {
                    'generation': current_generation,
                    'prompt': current_text_prompt,
                    'selected_indices': selected_indices,
                    'individuals': [
                        {
                            'index': i,
                            'expression': ind['tree'].get_str(),
                            'fitness': ind['fitness'],
                            'depth': ind['depth'],
                            'nodes': ind['nodes']
                        }
                        for i, ind in enumerate(current_population)
                    ]
                }
                
                with open(os.path.join(gen_export_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return f"✓ Current generation exported to {gen_export_dir}"
            else:
                return "No current generation data available"
    
    except Exception as e:
        return f"✗ Export failed: {str(e)}"

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
        # Add visual indicator for selected images
        if i in selected_indices:
            # Add a border or overlay to show selection
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            width, height = img.size
            border_width = 5
            draw.rectangle([0, 0, width-1, height-1], outline="red", width=border_width)
        
        images.append((img, f"Image {i}"))
    
    return images

def on_image_select(evt: gr.SelectData, gallery_state):
    """Handle image selection in gallery"""
    global selected_indices
    
    if gallery_state is None:
        gallery_state = []
    
    # Toggle selection
    if evt.index in gallery_state:
        gallery_state.remove(evt.index)
        if evt.index in selected_indices:
            selected_indices.remove(evt.index)
    else:
        gallery_state.append(evt.index)
        if evt.index not in selected_indices:
            selected_indices.append(evt.index)
    
    return gallery_state, f"Selected: {sorted(gallery_state)}"

def submit_selection(gallery_state, text_prompt):
    """Submit user selection and proceed to next generation"""
    global selected_indices, current_generation, current_engine, work_directory, current_text_prompt
    
    if gallery_state is None:
        gallery_state = []
    
    selected_indices = gallery_state.copy()
    current_text_prompt = text_prompt
    
    # Save current generation to history
    save_generation_to_history()
    
    # Save current generation data to disk
    if current_tensors is not None and work_directory is not None:
        save_generation_images(current_tensors, current_generation, work_directory)
        save_generation_metadata(current_population, selected_indices, current_generation, work_directory, text_prompt)
    
    current_generation += 1
    
    # Run next generation if engine exists
    if current_engine is not None and current_generation <= current_engine.stop_value:
        status_msg = f"Generation {current_generation} starting..."
        new_gallery_state = []
        images = get_current_images()
        stats = get_generation_stats()
        return new_gallery_state, status_msg, images, stats
    else:
        return [], "Evolution complete!", [], get_generation_stats()

def initialize_evolution_with_status(pop_size, num_gens, resolution, seed, text_prompt):
    """Enhanced initialization with status updates"""
    global current_engine, current_generation, work_directory, current_text_prompt, model_status
    
    # Phase 1: Model initialization
    yield "Initializing models...", [], "", []
    
    current_text_prompt = text_prompt
    models_loaded = False
    
    if text_prompt:
        models_loaded = initialize_models()
        if models_loaded and CLIP_AVAILABLE:
            encode_text_prompt(text_prompt)
    
    yield f"Models: {model_status}", [], "", []
    
    # Phase 2: Setup
    yield "Setting up evolution environment...", [], "", []
    
    # Parse resolution
    res_parts = resolution.split('x')
    if len(res_parts) == 2:
        image_resolution = [int(res_parts[0]), int(res_parts[1]), 3]
    else:
        image_resolution = [256, 256, 3]
    
    # Create work directory
    work_directory = create_work_directory()
    
    # Save configuration
    config = {
        "population_size": int(pop_size),
        "generations": int(num_gens),
        "resolution": resolution,
        "seed": int(seed) if seed else None,
        "text_prompt": text_prompt,
        "clip_available": CLIP_AVAILABLE,
        "laion_available": LAION_AVAILABLE,
        "models_loaded": models_loaded,
        "model_status": model_status
    }
    
    with open(os.path.join(work_directory, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    yield "Creating TensorGP engine...", [], "", []
    
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
        device='/cpu:0',
        stop_criteria='generation',
        exp_prefix='interactive',
        save_image_pop=False,
        save_image_best=False,
        run_dir_path=work_directory
    )
    
    current_generation = 0
    
    yield "Generating initial population...", [], "", []
    
    # Initialize population
    current_engine.experiment.set_generation_directory(0, lambda: False)
    population, best = current_engine.initialize_population(
        max_depth=current_engine.max_init_depth,
        min_depth=current_engine.min_init_depth,
        individuals=current_engine.population_size,
        method=current_engine.method,
        max_nodes=current_engine.max_nodes
    )
    
    yield "Evaluating initial population...", [], "", []
    
    # Enhanced initial evaluation
    if text_prompt and models_loaded:
        population, best = current_engine.fitness_func_wrap(
            population=population,
            f_path=current_engine.experiment.cur_image_directory
        )
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]
    
    current_engine.population = population
    current_engine.best = best
    current_engine.best_overall = current_engine.deep_shallow_copy(best)
    
    # Save initial generation to history
    save_generation_to_history()
    
    status_msg = f"✓ Evolution initialized: {pop_size} individuals, {num_gens} generations"
    if text_prompt:
        status_msg += f"\n✓ Text guidance: '{text_prompt}'"
    
    images = get_current_images()
    stats = get_generation_stats()
    
    yield status_msg, images, stats, []

def run_single_generation():
    """Run a single generation of evolution"""
    global current_engine, selected_indices
    
    if current_engine is None or not current_engine.condition():
        return "Evolution complete or not initialized", [], get_generation_stats()
    
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
    
    return f"Generation {current_engine.current_generation} complete", get_current_images(), get_generation_stats()

# Create Gradio interface
with gr.Blocks(title="Enhanced Interactive Evolutionary Art", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced Interactive EML-Art System")
    gr.Markdown("Guide evolution using text descriptions and manual selection, enhanced with CLIP semantic understanding and LAION aesthetic assessment")
    
    with gr.Tab("Setup & Initialization"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Evolution Parameters")
                pop_size_input = gr.Number(value=16, label="Population Size", minimum=4, maximum=100)
                num_gens_input = gr.Number(value=20, label="Number of Generations", minimum=1, maximum=100)
                resolution_input = gr.Dropdown(
                    choices=["128x128", "256x256", "512x512"],
                    value="256x256",
                    label="Image Resolution"
                )
                seed_input = gr.Number(value=42, label="Random Seed (optional)")
                
            with gr.Column():
                gr.Markdown("### AI Guidance")
                text_prompt_input = gr.Textbox(
                    label="Text Prompt (CLIP + LAION Enhanced)",
                    placeholder="e.g., 'sunset with bright colors', 'abstract geometric patterns', 'serene landscape'",
                    value="",
                    lines=3,
                    info="Describe what you want to evolve. Uses CLIP for semantic understanding and LAION for aesthetic quality."
                )
                
                model_status_display = gr.Textbox(
                    label="AI Model Status", 
                    value="Not initialized", 
                    interactive=False
                )
        
        init_button = gr.Button("Initialize Evolution", variant="primary", size="lg")
        init_status = gr.Textbox(label="Initialization Status", interactive=False, lines=3)
    
    with gr.Tab("Evolution & Selection"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Current Generation")
                gallery = gr.Gallery(
                    label="Population",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    rows=4,
                    object_fit="contain",
                    height="600px",
                    interactive=True
                )
                
                with gr.Row():
                    submit_button = gr.Button("Submit Selection & Next Generation", variant="primary")
                    auto_run_button = gr.Button("Auto-select Top 25%", variant="secondary")
                
                with gr.Row():
                    select_top3_btn = gr.Button("Select Top 3", size="sm")
                    select_top5_btn = gr.Button("Select Top 5", size="sm")
                    clear_selection_btn = gr.Button("Clear Selection", size="sm")
                
            with gr.Column(scale=1):
                gr.Markdown("### Statistics & Info")
                stats_display = gr.Textbox(label="Generation Statistics", interactive=False, lines=8)
                
                selection_status = gr.Textbox(label="Selected Images", value="Selected: []", interactive=False)
                
                image_details = gr.Textbox(label="Selected Image Details", interactive=False, lines=6)
                
                gen_status = gr.Textbox(label="Evolution Status", value="Not started", interactive=False)
        
        gallery_state = gr.State([])
        
        gr.Markdown("### Instructions:")
        gr.Markdown("""
        1. **Enter a text description** to guide evolution with CLIP + LAION AI models
        2. **Click images** to select favorites (combines user preference + AI evaluation)  
        3. **System evaluates** using: User Selection + CLIP Semantic Similarity + LAION Aesthetics
        4. **First generation** evaluated primarily on text-image alignment + quality
        5. **Subsequent generations** balance all three components for optimal results
        """)
    
    with gr.Tab("History & Navigation"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generation Navigation")
                gen_slider = gr.Slider(
                    minimum=0, maximum=20, step=1, value=0,
                    label="Navigate to Generation",
                    info="Browse through evolution history"
                )
                
                with gr.Row():
                    prev_gen_btn = gr.Button("← Previous", size="sm")
                    next_gen_btn = gr.Button("Next →", size="sm")
                
                history_status = gr.Textbox(label="Navigation Status", interactive=False)
                
            with gr.Column():
                gr.Markdown("### Generation History")
                history_info = gr.Textbox(
                    label="Available Generations",
                    interactive=False,
                    lines=10,
                    info="List of saved generations"
                )
    
    with gr.Tab("Export & Save"):
        gr.Markdown("### Export Options")
        gr.Markdown("Save your evolution results in various formats")
        
        with gr.Row():
            with gr.Column():
                export_best_btn = gr.Button("Export Best Image", variant="primary")
                export_expressions_btn = gr.Button("Export All Expressions")
                
            with gr.Column():
                export_log_btn = gr.Button("Export Evolution Log")
                export_current_btn = gr.Button("Export Current Generation")
        
        export_status = gr.Textbox(label="Export Status", interactive=False, lines=3)
        
        gr.Markdown("### Export Descriptions:")
        gr.Markdown("""
        - **Best Image**: High-resolution PNG of the best evolved individual
        - **Expressions**: Text file with all mathematical expressions from evolution
        - **Evolution Log**: Complete JSON log with statistics and metadata
        - **Current Generation**: All images and data from current generation
        """)

    # Event handlers
    def update_gen_slider_max():
        """Update generation slider maximum based on history"""
        if generation_history:
            max_gen = max(generation_history.keys())
            return gr.Slider.update(maximum=max_gen, value=current_generation)
        return gr.Slider.update()
    
    def get_history_info():
        """Get formatted history information"""
        if not generation_history:
            return "No generation history available"
        
        info = "Generation History:\n" + "="*30 + "\n"
        for gen in sorted(generation_history.keys()):
            data = generation_history[gen]
            timestamp = time.strftime('%H:%M:%S', time.localtime(data['timestamp']))
            selected_count = len(data.get('selected_indices', []))
            info += f"Gen {gen:2d}: {timestamp} - {selected_count} selected\n"
        
        return info
    
    def clear_selection():
        """Clear current selection"""
        global selected_indices
        selected_indices = []
        return [], "Selected: []", get_current_images()
    
    def handle_prev_gen():
        """Navigate to previous generation"""
        current = int(gen_slider.value) if hasattr(gen_slider, 'value') else current_generation
        prev_gen = max(0, current - 1)
        return load_generation_from_history(prev_gen)
    
    def handle_next_gen():
        """Navigate to next generation"""
        current = int(gen_slider.value) if hasattr(gen_slider, 'value') else current_generation
        max_gen = max(generation_history.keys()) if generation_history else current_generation
        next_gen = min(max_gen, current + 1)
        return load_generation_from_history(next_gen)
    
    # Initialization
    init_button.click(
        fn=initialize_evolution_with_status,
        inputs=[pop_size_input, num_gens_input, resolution_input, seed_input, text_prompt_input],
        outputs=[init_status, gallery, stats_display, gallery_state]
    ).then(
        fn=lambda: (update_gen_slider_max(), get_history_info()),
        outputs=[gen_slider, history_info]
    )
    
    # Gallery selection
    gallery.select(
        fn=lambda evt, state: (
            on_image_select(evt, state)[0],
            on_image_select(evt, state)[1], 
            get_image_details(evt),
            get_current_images()
        ),
        inputs=[gallery_state],
        outputs=[gallery_state, selection_status, image_details, gallery]
    )
    
    # Submit and run generation
    def handle_submit(state, prompt):
        new_state, status, images, stats = submit_selection(state, prompt)
        gen_status_msg, new_images, new_stats = run_single_generation()
        return new_state, gen_status_msg, new_images, new_stats, get_history_info()
    
    submit_button.click(
        fn=handle_submit,
        inputs=[gallery_state, text_prompt_input],
        outputs=[gallery_state, gen_status, gallery, stats_display, history_info]
    ).then(
        fn=lambda: ([], "Selected: []", "", update_gen_slider_max()),
        outputs=[gallery_state, selection_status, image_details, gen_slider]
    )
    
    # Auto-selection buttons
    def handle_auto_run(prompt):
        auto_selected, msg = select_top_n(max(1, len(current_population) // 4), prompt)
        new_state, status, images, stats = submit_selection(auto_selected, prompt)
        gen_status_msg, new_images, new_stats = run_single_generation()
        return [], gen_status_msg, new_images, new_stats, "Selected: []"
    
    auto_run_button.click(
        fn=handle_auto_run,
        inputs=[text_prompt_input],
        outputs=[gallery_state, gen_status, gallery, stats_display, selection_status]
    )
    
    select_top3_btn.click(
        fn=lambda prompt: select_top_n(3, prompt),
        inputs=[text_prompt_input],
        outputs=[gallery_state, selection_status]
    ).then(
        fn=get_current_images,
        outputs=[gallery]
    )
    
    select_top5_btn.click(
        fn=lambda prompt: select_top_n(5, prompt),
        inputs=[text_prompt_input],
        outputs=[gallery_state, selection_status]
    ).then(
        fn=get_current_images,
        outputs=[gallery]
    )
    
    clear_selection_btn.click(
        fn=clear_selection,
        outputs=[gallery_state, selection_status, gallery]
    )
    
    # History navigation
    gen_slider.change(
        fn=load_generation_from_history,
        inputs=[gen_slider],
        outputs=[gallery, history_status, stats_display, gallery_state]
    )
    
    prev_gen_btn.click(
        fn=handle_prev_gen,
        outputs=[gallery, history_status, stats_display, gallery_state]
    )
    
    next_gen_btn.click(
        fn=handle_next_gen,
        outputs=[gallery, history_status, stats_display, gallery_state]
    )
    
    # Export functions
    export_best_btn.click(
        fn=lambda: export_results("best_image"),
        outputs=[export_status]
    )
    
    export_expressions_btn.click(
        fn=lambda: export_results("expressions"),
        outputs=[export_status]
    )
    
    export_log_btn.click(
        fn=lambda: export_results("evolution_log"),
        outputs=[export_status]
    )
    
    export_current_btn.click(
        fn=lambda: export_results("current_generation"),
        outputs=[export_status]
    )

# Command-line interface support
if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        generations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        prompt = sys.argv[4] if len(sys.argv) > 4 else ""
        
        print(f"Command line mode:")
        print(f"  Seed: {seed}")
        print(f"  Runs: {num_runs}")
        print(f"  Generations: {generations}")
        print(f"  Prompt: '{prompt}'")
        print("  Launching interactive GUI...")
    
    # Launch Gradio interface
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        quiet=False
    )