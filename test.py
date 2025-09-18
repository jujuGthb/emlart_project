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

# Model loading 
CLIP_AVAILABLE = False
AESTHETIC_MODEL_AVAILABLE = False
device = "cuda" if torch.cuda.is_available() else "cpu"  # Default to CUDA
vit_model = None
preprocess = None
aesthetic_model = None

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
    
    # Try to import LAION aesthetic model 
    try:
        from laion_aesthetics import MLP, normalizer, init_laion
        aesthetic_model, vit_model, preprocess = init_laion(device)  # Use detected device
        AESTHETIC_MODEL_AVAILABLE = True
        print(f"Full CLIP + LAION aesthetic model loaded successfully on {device}")
    except ImportError:
        # Fallback to just CLIP
        vit_model, preprocess = clip.load("ViT-L-14", device=device)
        print(f"CLIP loaded successfully on {device} (LAION aesthetic model not available)")
    except Exception as e:
        # If CUDA fails, fallback to CPU
        print(f"Failed to load models on {device}, falling back to CPU: {e}")
        device = "cpu"
        try:
            from laion_aesthetics import MLP, normalizer, init_laion
            aesthetic_model, vit_model, preprocess = init_laion(device)
            AESTHETIC_MODEL_AVAILABLE = True
            print(f"Models loaded successfully on CPU")
        except:
            vit_model, preprocess = clip.load("ViT-L-14", device=device)
            print(f"CLIP loaded successfully on CPU")
    
except ImportError:
    print("CLIP not available - running in basic mode")

# Global variables for evolution state
current_engine = None
current_population = None
current_tensors = None
current_generation = 0
selected_indices = []
work_directory = None
text_features = None
current_text_prompt = None
generation_history = {}
current_display_generation = 0
navigation_enabled = False
stats_data = []

def encode_text_prompt(prompt):
    """Encode text prompt using CLIP like in original"""
    global text_features
    if CLIP_AVAILABLE and vit_model is not None and prompt:
        text_inputs = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_features = vit_model.encode_text(text_inputs)
        return True
    return False

def create_work_directory(base_dir="enhanced_interactive_evolution"):
    """Create timestamped work directory"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "generations"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "selected"), exist_ok=True)
    return work_dir

def enhanced_fitness_evaluation(**kwargs):
    """
    Enhanced fitness evaluation that closely mirrors the original approach
    but allows for interactive selection weighting
    """
    global current_population, current_tensors, selected_indices, current_text_prompt
    
    population = kwargs.get('population')
    tensors = kwargs.get('tensors')
    generation = kwargs.get('generation', 0)
    
    current_population = population
    current_tensors = tensors
    
    fitnesses = []
    best_ind = 0
    best_fitness = float('-inf')
    
    for idx, tensor in enumerate(tensors):
        fitness = 0.0
        
        # Convert tensor to PIL like in original for model evaluation
        img_array = tensor.numpy()
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            pil_image = Image.fromarray(img_array, mode='L').convert('RGB')
        else:
            pil_image = Image.fromarray(img_array)
        
        # CLIP similarity evaluation (like original)
        if CLIP_AVAILABLE and current_text_prompt and text_features is not None:
            try:
                image = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = vit_model.encode_image(image)
                
                # Calculate similarity like in original
                similarity = torch.cosine_similarity(text_features, image_features, dim=-1).mean()
                clip_fitness = similarity.item()
                
                # Use LAION aesthetic model if available 
                if AESTHETIC_MODEL_AVAILABLE and aesthetic_model is not None:
                    im_emb_arr = normalizer(image_features.cpu().detach().numpy())
                    prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float))
                    aesthetic_fitness = prediction.item()
                    
                    # Combined fitness (with user selection weighting)
                    if idx in selected_indices:
                       
                        fitness = clip_fitness * 1.5 + aesthetic_fitness / 200.0 * 1.5 + 2.0
                    else:
                        
                        fitness = clip_fitness + aesthetic_fitness / 200.0
                else:
                    
                    if idx in selected_indices:
                        fitness = clip_fitness * 1.5 + 1.5
                    else:
                        fitness = clip_fitness
                        
            except Exception as e:
                print(f"Error in fitness evaluation: {e}")
                fitness = 1.0 if idx in selected_indices else 0.01
        else:
            # Fallback: user selection based fitness
            if idx in selected_indices:
                fitness = 1.0 + np.random.random() * 0.1
            else:
                fitness = np.random.random() * 0.1
        
        # Add small random component to avoid ties 
        fitness += 0.001 * np.random.random()
        
        population[idx]['fitness'] = fitness
        fitnesses.append(fitness)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_ind = idx
    
    return population, best_ind

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image with proper handling"""
    img_array = tensor.numpy()
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array, mode='L')
    else:
        return Image.fromarray(img_array)

def get_current_images(start_idx=0, grid_size=16):
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
    
    if evt.index in gallery_state:
        gallery_state.remove(evt.index)
    else:
        gallery_state.append(evt.index)
    
    return gallery_state, f"Selected: {gallery_state}"

def initialize_evolution(pop_size, num_gens, resolution, seed, text_prompt):
    """Initialize evolution using original's configuration"""
    global current_engine, current_generation, work_directory, current_text_prompt
    global current_display_generation, stats_data, generation_history
    
    stats_data = []
    generation_history = {}
    current_text_prompt = text_prompt
    
    # Initialize models 
    if text_prompt and CLIP_AVAILABLE:
        encode_text_prompt(text_prompt)
        print(f"Using enhanced guidance with prompt: {text_prompt}")
    
    # Parse resolution
    res_parts = resolution.split('x')
    if len(res_parts) == 2:
        image_resolution = [int(res_parts[0]), int(res_parts[1]), 3]
    else:
        image_resolution = [224, 224, 3]  # Default 
    
    work_directory = create_work_directory()
    
    # Use original's function set
    fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'log',
            'max', 'mdist', 'min', 'mult', 'neg', 'or', 'pow', 'sin', 'sqrt', 
            'sub', 'tan', 'warp', 'xor'}
    
    # Create engine with original's configuration
    current_engine = Engine(
        fitness_func=enhanced_fitness_evaluation,
        population_size=int(pop_size),
        tournament_size=5,  
        
        # EC parameters from original
        mutation_rate=0.9,
        crossover_rate=0.5,
        elitism=max(4, int(pop_size) // 3),
        
        # GP specific parameters from original
        terminal_prob=0.2,
        scalar_prob=0.50,
        uniform_scalar_prob=1,
        max_retries=20,
        
        # Mutations from original
        mutation_funcs=[Engine.point_mutation, Engine.subtree_mutation, 
                       Engine.insert_mutation, Engine.delete_mutation],
        mutation_probs=[0.25, 0.3, 0.2, 0.25],
        
        # Tree initialization from original
        method='ramped half-and-half',
        max_tree_depth=12,
        min_tree_depth=-1,
        min_init_depth=1,
        max_init_depth=6,
        
        # Bloat control from original
        bloat_control='weak',
        bloat_mode='depth',
        dynamic_limit=5,
        min_overall_size=1,
        max_overall_size=12,
        
        # Domain mapping from original
        domain=[-1, 1],
        codomain=[-1, 1],
        do_final_transform=True,
        final_transform=[0, 255],
        
        # Other parameters
        stop_value=int(num_gens),
        effective_dims=3,
        seed=int(seed) if seed else None,
        operators=fset,
        debug=0,
        save_to_file=1,
        target_dims=image_resolution,
        objective='maximizing',
        device='/gpu:0',
        stop_criteria='generation',
        exp_prefix='enhanced-interactive',
        save_image_pop=False,
        save_image_best=False,
        run_dir_path=work_directory
    )
    
    current_generation = 0
    current_display_generation = 0
    
    # Initialize population 
    current_engine.experiment.set_generation_directory(0, lambda: False)
    population, best = current_engine.initialize_population(
        max_depth=current_engine.max_init_depth,
        min_depth=current_engine.min_init_depth,
        individuals=current_engine.population_size,
        method=current_engine.method,
        max_nodes=current_engine.max_nodes
    )
    
    # Evaluate initial population with enhanced fitness
    population, best = current_engine.fitness_func_wrap(
        population=population,
        f_path=current_engine.experiment.cur_image_directory
    )
    
    # Sort by fitness like in original
    population.sort(key=lambda x: x['fitness'], reverse=True)
    best = population[0]
    
    current_engine.population = population
    current_engine.best = best
    current_engine.best_overall = current_engine.deep_shallow_copy(best)
    
    status_msg = f"Enhanced evolution initialized: {pop_size} individuals, {num_gens} generations"
    if text_prompt:
        status_msg += f"\nUsing semantic guidance: '{text_prompt}'"
    
    return status_msg, get_current_images()

def run_single_generation():
    """Run single generation using original's approach"""
    global current_engine, selected_indices, current_tensors, current_population
    
    if current_engine is None or not current_engine.condition():
        return "Evolution complete or not initialized", []
    
  
    # Get elite individuals
    new_population = current_engine.get_n_best_from_pop(
        population=current_engine.population, 
        n=current_engine.elitism
    )
    
    # Generate offspring 
    temp_population = []
    for _ in range(current_engine.population_size - current_engine.elitism):
       
        indiv_temp, parent, plist = current_engine.selection()
        member_depth, member_nodes = indiv_temp.get_depth()
        
        temp_population.append(
            new_individual(indiv_temp, fitness=0, depth=member_depth, 
                          nodes=member_nodes, valid=False, parents=plist)
        )
    
    # Evaluate all offspring
    if len(temp_population) > 0:
        temp_population, _ = current_engine.fitness_func_wrap(
            population=temp_population, 
            f_path=current_engine.experiment.current_directory
        )
    
    # Combine populations 
    new_population += temp_population
    current_engine.population = new_population
    current_population = new_population
    
    # Calculate tensors for display
    with tf.device(current_engine.device):
        current_tensors, _ = current_engine.calculate_tensors(current_engine.population)
    
    # Update best individuals
    current_engine.best = current_engine.get_n_best_from_pop(
        population=current_engine.population, n=1
    )[0]
    
    if current_engine.condition_overall(current_engine.best['fitness']):
        current_engine.best_overall = current_engine.deep_shallow_copy(current_engine.best)
    
    current_engine.current_generation += 1
    
    return f"Generation {current_engine.current_generation} complete", get_current_images()

def submit_selection(gallery_state, text_prompt):
    """Submit selection and proceed to next generation"""
    global selected_indices, current_generation, current_engine, work_directory
    global current_text_prompt, navigation_enabled, current_display_generation
    
    if navigation_enabled:
        return gallery_state, "Cannot submit while viewing historical generation", get_current_images()
    
    if gallery_state is None:
        gallery_state = []
    
    selected_indices = gallery_state.copy()
    current_text_prompt = text_prompt
    
    # Save to history
    save_generation_to_history()
    
    current_generation += 1
    current_display_generation = current_generation
    
    # Run next generation
    if current_engine is not None and current_generation <= current_engine.stop_value:
        status_msg, images = run_single_generation()
        return [], status_msg, images
    else:
        return [], "Evolution complete!", []

def save_generation_to_history():
    """Save current generation to history"""
    global generation_history, current_generation, current_population, current_tensors
    
    if current_population is not None and current_tensors is not None:
        fitness_values = [float(ind['fitness']) for ind in current_population]
        depth_values = [int(ind['depth']) for ind in current_population]
        nodes_values = [int(ind['nodes']) for ind in current_population]
        
        metadata = {
            'generation': current_generation,
            'selected_indices': selected_indices.copy(),
            'text_prompt': current_text_prompt,
            'fitness_stats': {
                'avg': float(np.mean(fitness_values)),
                'std': float(np.std(fitness_values)),
                'max': float(np.max(fitness_values)),
                'min': float(np.min(fitness_values))
            },
            'depth_stats': {
                'avg': float(np.mean(depth_values)),
                'max': int(np.max(depth_values)),
                'min': int(np.min(depth_values))
            },
            'nodes_stats': {
                'avg': float(np.mean(nodes_values)),
                'max': int(np.max(nodes_values)),
                'min': int(np.min(nodes_values))
            }
        }
        
        generation_history[current_generation] = {
            'population': copy.deepcopy(current_population),
            'tensors': [tensor.numpy().copy() for tensor in current_tensors],
            'metadata': metadata
        }
        
        stats_data.append([
            current_generation,
            metadata['fitness_stats']['avg'],
            metadata['fitness_stats']['std'],
            metadata['fitness_stats']['max'],
            metadata['depth_stats']['avg'],
            metadata['nodes_stats']['avg'],
            len(selected_indices)
        ])

def get_generation_history_table():
    """Get generation history table data"""
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

# Create enhanced Gradio interface
with gr.Blocks(title="Enhanced Interactive Evolutionary Art") as demo:
    
    gr.HTML("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🎨 Enhanced Interactive Evolutionary Art</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                Combining advanced AI models with interactive evolution
            </p>
        </div>
    """)
    
    with gr.Tab("🚀 Setup"):
        gr.HTML('<h3>Evolution Parameters</h3>')
        
        with gr.Row():
            pop_size_input = gr.Number(value=50, label="Population Size", minimum=10, maximum=200)
            num_gens_input = gr.Number(value=30, label="Generations", minimum=1, maximum=100)
        
        with gr.Row():
            resolution_input = gr.Dropdown(
                choices=["224x224", "256x256", "512x512"],
                value="224x224",
                label="Resolution"
            )
            seed_input = gr.Number(value=1, label="Random Seed")
        
        text_prompt_input = gr.Textbox(
            label="Text Prompt (Required for semantic guidance)",
            placeholder="e.g., 'sunset bright colors abstract art'",
            value="abstract colorful patterns",
            lines=2
        )
        
        init_button = gr.Button("Initialize Enhanced Evolution", variant="primary")
        init_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("🧬 Evolution"):
        with gr.Row():
            gen_status = gr.Textbox(label="Generation Status", interactive=False)
            current_gen_display = gr.Number(label="Current Generation", value=0, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Population",
                    columns=4,
                    rows=3,
                    height=400,
                    interactive=True
                )
                
                gallery_state = gr.State([])
                selection_status = gr.Textbox(label="Selected Images", interactive=False)
                
                with gr.Row():
                    submit_button = gr.Button("Submit Selection & Next Generation", variant="primary")
            
            with gr.Column(scale=2):
                history_table = gr.Dataframe(
                    headers=["Gen", "Avg Fit", "Max Fit", "Avg Depth", "Avg Nodes", "Selections"],
                    label="Evolution History",
                    interactive=False
                )
    
    # Event handlers
    init_button.click(
        fn=initialize_evolution,
        inputs=[pop_size_input, num_gens_input, resolution_input, seed_input, text_prompt_input],
        outputs=[init_status, gallery]
    ).then(
        fn=lambda: (0, get_generation_history_table()),
        outputs=[current_gen_display, history_table]
    )
    
    gallery.select(
        fn=on_image_select,
        inputs=[gallery_state],
        outputs=[gallery_state, selection_status]
    )
    
    submit_button.click(
        fn=submit_selection,
        inputs=[gallery_state, text_prompt_input],
        outputs=[gallery_state, gen_status, gallery]
    ).then(
        fn=lambda: (current_display_generation, get_generation_history_table()),
        outputs=[current_gen_display, history_table]
    ).then(
        fn=lambda: ([], "Selected: []"),
        outputs=[gallery_state, selection_status]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="localhost", server_port=7860)