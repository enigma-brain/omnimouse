# import debugpy
# try:
# 	debugpy.listen(('localhost', 5678))
# 	print('Listening for incoming debug connection on local port 5678...')
# 	debugpy.wait_for_client()
# except:
# 	print('WARNING: Failed to find incoming debug connection on local port 5678...')

"""
Visualize masking strategies for neural activity reconstruction training.

This script creates interactive visualizations of masking configurations specified
in Hydra config files, showing which regions are encoded, decoded, or ignored.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Mapping
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hydra
from omegaconf import DictConfig, ListConfig
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import the MaskingStrategy class
from omnimouse.masking.strategies import MaskingStrategy
from omnimouse.utils.utils import omegaconf_to_masking_strategy


def get_mask_regions(
    masking_strategy: MaskingStrategy,
    S: int,
    N: int,
    interpolation_buffer: int = 0,
    skip_n_samples: int = 0,
    kernel_size: int = 100,
    stride: int = 1,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Determine which regions are encoded, decoded, or ignored based on masking strategy.
    Uses the exact logic from _preprocess_input_responses in model.py.
    
    Returns a tuple of (masks_dict, error_messages).
    error_messages is a list of all errors encountered during processing.
    """
    # Initialize masks
    mask = np.zeros((S, N), dtype=int)  # 0: ignored, 1-2: encoded subblocks, 3-4: decoded subblocks
    error_messages = []
    
    # Get masking strategy parameters (matching variable names from model.py)
    psi = masking_strategy.prefix_start_idx
    pl = masking_strategy.prefix_len
    pe = psi + pl  # prefix end
    rcs = masking_strategy.response_context_start_idx
    mrc = masking_strategy.max_response_context_samples
    rce = rcs + mrc  # response context end
    fppl = masking_strategy.full_population_prefix_len
    fpe = psi + fppl  # full population prefix end
    ssi = masking_strategy.suffix_start_idx
    sl = masking_strategy.suffix_len
    se = ssi + sl  # suffix end
    nvn = min(masking_strategy.n_visible_neurons, N)
    mnrn = min(masking_strategy.max_n_reconstructed_neurons, N)

    # Check that encoding/decoding configuration is valid
    assert 0 <= rcs <= rce <= S, \
        f"Response context ({rcs} - {rce}) must be contained within full window ({0} - {S})!"
    assert rcs <= psi <= pe <= rce, \
        f"Prefix ({psi} - {pe}) must be contained within response context ({rcs} - {rce})!"
    assert fpe <= pe, \
        f"Full population prefix ({psi} - {fpe}) must be contained within prefix ({psi} - {pe})!"
    assert 0 <= ssi <= se <= S, \
        f"Suffix ({ssi} - {se}) must be contained within full window ({0} - {S})!"

    def update_mask(mask: np.ndarray, unmasked_sample_slice: slice, unmasked_neuron_slice: slice, val: int, phase: str) -> List[str]:
        _error_messages = []
        seq_s, n_neurons_s = mask[unmasked_sample_slice, unmasked_neuron_slice].shape
        if 0 in (seq_s, n_neurons_s):
            _error_messages.append(f"[{phase}] Empty slice (seq_s={seq_s}, n_neurons_s={n_neurons_s})")
        if seq_s % stride != 0:
            _error_messages.append(f"[{phase}] Sequence length is not divisible by stride")
        if 'encoder' in phase and seq_s < kernel_size:
            _error_messages.append(f"[{phase}] Insufficient samples (n_samples={seq_s} < kernel={kernel_size})")
        if 'encoder' in phase and (seq_s - kernel_size) % stride != 0:
            _error_messages.append(f"[{phase}] Seq, stride, and kernel size not compatible, responses will be dropped!")
        mask[unmasked_sample_slice, unmasked_neuron_slice] = val
        return _error_messages
    
    # ========== ENCODED REGIONS (from _preprocess_input_responses) ==========
    # First encoded subblock: sections A and C (full population prefix)
    if fppl > 0:
        unmasked_sample_slice = slice(psi, fpe)
        unmasked_neuron_slice = slice(0, None)
        error_messages.extend(update_mask(mask, unmasked_sample_slice, unmasked_neuron_slice, 1, 'encoded_1'))
    
    # Second encoded subblock: section B (remainder of prefix)
    if pl > 0 and nvn > 0 and fpe < pe:
        unmasked_sample_slice = slice(fpe, pe)
        unmasked_neuron_slice = slice(0, nvn)
        error_messages.extend(update_mask(mask, unmasked_sample_slice, unmasked_neuron_slice, 2, 'encoded_2'))
    
    # ========== DECODED REGIONS (from _preprocess_input_responses) ==========
    if sl > 0 and mnrn > 0:
        # Check for skip_n_samples buffer violation
        if ssi < skip_n_samples:
            error_messages.append("Suffix start index is less than skip_n_samples")
        
        n_start = N - mnrn
        # Calculate interpolation buffer ends
        fpe_plus_ib = min(fpe + interpolation_buffer, S) if fppl > 0 else 0
        pe_plus_ib = min(pe + interpolation_buffer, S) if pl > 0 else 0
        
        # if the visible population subset doesn't overlap with the reconstructed population,
        # or if the suffix doesn't overlap with the prefix range, we can reconstruct the
        # continuous block of samples (for `mnrn` neurons) from suffix start (or the 
        # interpolation buffer of the full population prefix, if it exists) to the suffix end.
        if n_start >= nvn or ssi >= pe_plus_ib:
            masked_sample_slice = slice(max(ssi, fpe_plus_ib), se)
            masked_neuron_slice = slice(n_start, None)
            error_messages.extend(update_mask(mask, masked_sample_slice, masked_neuron_slice, 3, 'decoded_1'))
        else:
            # if the visible population subset does overlap with the reconstructed population,
            # we first reconstruct the non-visible neurons within the prefix range.
            masked_sample_slice = slice(max(ssi, fpe_plus_ib), min(pe_plus_ib, se))
            masked_neuron_slice = slice(nvn, None)
            error_messages.extend(update_mask(mask, masked_sample_slice, masked_neuron_slice, 3, 'decoded_1'))
            
            # if the suffix range extends beyond the prefix range, we reconstruct all `mnrn`
            # neurons for the remaining suffix samples.
            if se > pe_plus_ib:
                masked_sample_slice = slice(pe_plus_ib, se)
                masked_neuron_slice = slice(n_start, None)
                error_messages.extend(update_mask(mask, masked_sample_slice, masked_neuron_slice, 4, 'decoded_2'))
    
    masks = {
        'encoded_1': mask == 1,
        'encoded_2': mask == 2,
        'decoded_1': mask == 3,
        'decoded_2': mask == 4,
        'ignored': mask == 0,
    }
    
    return masks, error_messages


def create_masking_figure(
    strategy: MaskingStrategy,
    S: int,
    N: int,
    V: int,
    title: str,
    interpolation_buffer: int = 0,
    skip_n_samples: int = 0,
    kernel_size: int = 100,
    stride: int = 1,
) -> Tuple[go.Figure, bool]:
    """Create an interactive Plotly figure showing the masking configuration.
    
    Returns:
        Tuple of (figure, has_error) where has_error is True if there was an error.
    """

    # Re-instantiate the strategy with concrete values
    strategy = strategy.instantiate(
        neural_population_size=N,
        num_samples_per_block=S,
        num_frames_per_block=V,
    )

    # Add some metadata to the title
    title = f"{title}<br># Response Context: {strategy.max_response_context_samples} | # Frames: {strategy.n_visible_frames} | Behavior in: {strategy.behavior_encoded} | Behavior out: {strategy.behavior_reconstructed}"
    
    # Get mask regions using exact logic from model.py
    masks, error_messages = get_mask_regions(strategy, S, N, interpolation_buffer, skip_n_samples, kernel_size, stride)
    has_error = len(error_messages) > 0
    
    # Create subplots: main plot + 3 bars
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.75, 0.08, 0.08, 0.09],  # Main plot gets most space, bars get smaller space with tight spacing
        subplot_titles=["Neural Activity Masking", "", "", ""],  # Only title for main plot
        vertical_spacing=0.05,  # Reduced spacing between all subplots
        shared_xaxes=True,
    )
    
    # ========== MAIN PLOT (Row 1) ==========
    # First, add a white background rectangle to main plot
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=S, y1=N,
        fillcolor="white",
        line=dict(width=0),
        layer="below",
        row=1, col=1
    )
    
    # Color mapping for main plot
    color_map = {
        'encoded_1': ('lightgreen', 'Encoded (full population prefix)'),
        'encoded_2': ('darkgreen', 'Encoded (visible neurons)'),
        'decoded_1': ('lightsalmon', 'Decoded (subblock 1)'),
        'decoded_2': ('darkorange', 'Decoded (subblock 2)')
    }
    
    # For large tensors, we'll draw rectangles more efficiently
    # by finding continuous blocks
    for mask_type, (color, description) in color_map.items():
        mask = masks[mask_type]
        if not mask.any():
            continue
            
        # Find rows and columns where mask is True
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
            
        # Group into continuous rectangles (simplified approach)
        # For efficiency with large N, we'll draw one rectangle per continuous sample range
        unique_samples = np.unique(rows)
        for sample in unique_samples:
            neuron_indices = cols[rows == sample]
            if len(neuron_indices) > 0:
                n_min = neuron_indices.min()
                n_max = neuron_indices.max() + 1
                
                # Add rectangle (transpose coordinates: sample->x, neuron->y)
                fig.add_shape(
                    type="rect",
                    x0=sample, y0=n_min, x1=sample + 1, y1=n_max,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                    row=1, col=1
                )
    
    # ========== BAR PLOTS ==========
    # Bar 1: Visible Frames (60 frames long, green bar for n_visible_frames)
    visible_frames_length = 60
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=visible_frames_length, y1=1,
        fillcolor="white",
        line=dict(width=1, color="black"),
        row=2, col=1
    )
    
    if strategy.n_visible_frames > 0:
        # Video blocks start at n_visible_frames_start_idx
        visible_start = strategy.n_visible_frames_start_idx
        visible_end = min(visible_start + strategy.n_visible_frames, visible_frames_length)
        fig.add_shape(
            type="rect",
            x0=visible_start, y0=0, x1=visible_end, y1=1,
            fillcolor="green",
            line=dict(width=0),
            row=2, col=1
        )
    
    # Bar 2: Behavior Encoded (60 samples long, green if behavior_encoded is True)
    behavior_bar_length = 60
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=behavior_bar_length, y1=1,
        fillcolor="white",
        line=dict(width=1, color="black"),
        row=3, col=1
    )
    
    if strategy.behavior_encoded:
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=behavior_bar_length, y1=1,
            fillcolor="green",
            line=dict(width=0),
            row=3, col=1
        )
    
    # Bar 3: Behavior Reconstructed (60 samples long, orange if behavior_reconstructed is True)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=behavior_bar_length, y1=1,
        fillcolor="white",
        line=dict(width=1, color="black"),
        row=4, col=1
    )
    
    if strategy.behavior_reconstructed:
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=behavior_bar_length, y1=1,
            fillcolor="orange",
            line=dict(width=0),
            row=4, col=1
        )
    
    # Add a dummy trace for the legend and basic interactivity
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Get strategy parameters for lines
    psi = strategy.prefix_start_idx
    pl = strategy.prefix_len
    pe = psi + pl # prefix end
    rcs = strategy.response_context_start_idx
    mrc = strategy.max_response_context_samples
    rce = rcs + mrc # response context end
    fppl = strategy.full_population_prefix_len
    fpe = psi + fppl # full population prefix end
    ssi = strategy.suffix_start_idx
    sl = strategy.suffix_len
    se = ssi + sl # suffix end
    nvn = min(strategy.n_visible_neurons, N)
    mnrn = min(strategy.max_n_reconstructed_neurons, N)
    n_start = N - mnrn
    fpe_plus_ib = min(fpe + interpolation_buffer, S) if fppl > 0 else 0
    pe_plus_ib = min(pe + interpolation_buffer, S) if pl > 0 else 0
    
    # Vertical lines (sample boundaries - now on x-axis) - only on main plot
    sample_lines = [
        ('full_population_prefix_len', fpe, 'blue'),
        ('prefix_start', psi, 'green'),
        ('prefix_end', pe, 'green'),
        ('response_context_start', rcs, 'purple'),
        ('response_context_end', rce, 'purple'),
        ('suffix_start', ssi, 'orange'),
        ('suffix_end', se, 'red'),
        ('skip_n_samples', skip_n_samples, 'brown'),
    ]
    
    if 0 < fppl < S:
        sample_lines.append(('fppl_buffer_end', fpe_plus_ib, 'gray'))
    if 0 < pl < S:
        sample_lines.append(('pl_buffer_end', pe_plus_ib, 'gray'))
    
    # Horizontal lines (neuron boundaries - now on y-axis) - only on main plot
    neuron_lines = [
        ('n_visible_neurons', nvn, 'green'),
        ('n_reconstructed_start', n_start, 'orange'),
    ]
    
    # Add vertical lines (sample boundaries) to main plot
    for label, pos, color in sample_lines:
        if 0 <= pos <= S:
            # Special styling for response context lines
            if 'response_context' in label:
                line_style = dict(color=color, width=4, dash='solid')  # Thicker and solid
            else:
                line_style = dict(color=color, width=2, dash='dot')    # Normal style
                
            fig.add_shape(
                type='line',
                x0=pos, x1=pos,
                y0=0, y1=N,
                line=line_style,
                row=1, col=1
            )
            # Position annotations to avoid overlap
            y_pos = N * (0.95 - 0.05 * sample_lines.index((label, pos, color)))
            
            # Special text color for response context lines
            text_color = 'purple' if 'response_context' in label else 'black'
            
            fig.add_annotation(
                x=pos, y=y_pos,
                text=f'{label}={pos}',
                showarrow=False,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                opacity=0.9,
                font=dict(size=9, color=text_color),
                textangle=-45,
                row=1, col=1
            )
    
    # Add horizontal lines (neuron boundaries) to main plot
    for label, pos, color in neuron_lines:
        if 0 <= pos <= N:
            fig.add_shape(
                type='line',
                x0=0, x1=S,
                y0=pos, y1=pos,
                line=dict(color=color, width=2, dash='dot'),
                row=1, col=1
            )
            fig.add_annotation(
                x=S*0.02, y=pos,
                text=f'{label}={pos}',
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.8)',
                opacity=0.9,
                font=dict(size=9),
                row=1, col=1
            )
    
    # Add text annotations to the bars
    fig.add_annotation(
        x=visible_frames_length/2, y=0.5,
        text=f'n_visible_frames = {strategy.n_visible_frames} (start: {strategy.n_visible_frames_start_idx})',
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=10),
        row=2, col=1
    )
    
    fig.add_annotation(
        x=behavior_bar_length/2, y=0.5,
        text=f'behavior_encoded = {strategy.behavior_encoded}',
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=10),
        row=3, col=1
    )
    
    fig.add_annotation(
        x=behavior_bar_length/2, y=0.5,
        text=f'behavior_reconstructed = {strategy.behavior_reconstructed}',
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=10),
        row=4, col=1
    )
    
    # Add error annotation if there's an error (to main plot)
    if has_error:
        # Add multiple red error annotations for each error
        for i, error_msg in enumerate(error_messages):
            # Position errors vertically spaced
            y_offset = (i - (len(error_messages) - 1) / 2) * N * 0.15  # Space errors vertically
            error_y = N/2 + y_offset
            
            fig.add_annotation(
                x=S/2, y=error_y,
                text=f"<b>ERROR {i+1}:</b><br>{error_msg}",
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=3,
                arrowcolor="red",
                ax=0, ay=-30,
                bgcolor='rgba(255,0,0,0.9)',
                bordercolor='darkred',
                borderwidth=2,
                font=dict(size=11, color='white'),
                align='center',
                row=1, col=1
            )
        
        # Also add title modification to indicate error
        title = f"⚠️ {len(error_messages)} ERROR(S): {title}"
    
    # Calculate appropriate figure dimensions
    # Since N >> S, we want to compress the y-axis relative to x-axis
    aspect_ratio = 0.05  # Adjust this to control the compression
    fig_height = max(800, min(1300, 600 + N * aspect_ratio))  # Increased height for better subplot spacing
    fig_width = max(800, min(1600, 400 + S * 10))
    
    # Update layout for each subplot
    fig.update_xaxes(
        range=[0, S],
        constrain='domain',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showticklabels=True,  # Explicitly show tick labels on main plot
        row=1, col=1
    )
    
    fig.update_yaxes(
        title='Neurons',
        range=[0, N],
        constrain='domain',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    # # Add "Samples" label to the left of the x-axis
    # fig.add_annotation(
    #     x=-0.08, y=0,
    #     text='Samples',
    #     showarrow=False,
    #     xanchor='center',
    #     yanchor='middle',
    #     font=dict(size=10),
    #     textangle=0,
    #     row=1, col=1
    # )
    
    # Configure bar subplot axes
    fig.update_xaxes(range=[0, visible_frames_length], showticklabels=False, row=2, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=2, col=1)
    
    fig.update_xaxes(range=[0, behavior_bar_length], showticklabels=False, row=3, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=3, col=1)
    
    fig.update_xaxes(range=[0, behavior_bar_length], showticklabels=False, row=4, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=4, col=1)
    
    # Update overall layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=14, color='red' if has_error else 'black'),
        },
        width=fig_width,
        height=fig_height,
        template='plotly_white',
        margin=dict(l=80, r=450, t=150, b=80),  # Increased right margin for more legend spacing
        showlegend=False,
    )
    
    # Add legend box with parameters
    legend_text = (
        f"<b>Configuration Parameters:</b><br>"
        f"n_visible_neurons: {strategy.n_visible_neurons}<br>"
        f"prefix_start_idx: {strategy.prefix_start_idx}<br>"
        f"prefix_len: {strategy.prefix_len}<br>"
        f"full_population_prefix_len: {strategy.full_population_prefix_len}<br>"
        f"response_context_start_idx: {strategy.response_context_start_idx}<br>"
        f"max_response_context_samples: {strategy.max_response_context_samples}<br>"
        f"max_n_reconstructed_neurons: {strategy.max_n_reconstructed_neurons}<br>"
        f"suffix_start_idx: {strategy.suffix_start_idx}<br>"
        f"suffix_len: {strategy.suffix_len}<br>"
        f"n_visible_frames_start_idx: {strategy.n_visible_frames_start_idx}<br>"
        f"n_visible_frames: {strategy.n_visible_frames}<br>"
        f"behavior_encoded: {strategy.behavior_encoded}<br>"
        f"behavior_reconstructed: {strategy.behavior_reconstructed}<br>"
        f"weight: {strategy.weight}<br>"
        f"<br><b>Color Legend:</b><br>"
        f"<span style='color:lightgreen'>■</span> Light Green: Encoded (full population prefix)<br>"
        f"<span style='color:darkgreen'>■</span> Dark Green: Encoded (visible neurons)<br>"
        f"<span style='color:lightsalmon'>■</span> Light Orange: Decoded (subblock 1)<br>"
        f"<span style='color:darkorange'>■</span> Dark Orange: Decoded (subblock 2)<br>"
        f"<span style='color:gray'>■</span> White: Ignored<br>"
        f"<span style='color:green'>■</span> Green: Visible frames & behavior encoding<br>"
        f"<span style='color:orange'>■</span> Orange: Behavior decoding"
    )
    
    # Add error info to legend if there's an error
    if has_error:
        error_list = "<br>".join([f"{i+1}. {msg}" for i, msg in enumerate(error_messages)])
        legend_text = (
            f"<b style='color:red'>{len(error_messages)} ERROR(S) FOUND:</b><br>{error_list}<br><br>"
            + legend_text
        )
    
    fig.add_annotation(
        text=legend_text,
        showarrow=False,
        xref="paper", yref="paper",
        x=1.02, y=0.98,
        xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='red' if has_error else 'black',
        borderwidth=2 if has_error else 1,
        font=dict(size=10),
        align='left',
    )
    
    return fig, has_error


def create_interactive_html(
    strategies: List[Tuple[str, MaskingStrategy]],
    S: int,
    N: int,
    V: int,
    main_title: str,
    output_file: Optional[Path] = None,
    interpolation_buffer: int = 0,
    skip_n_samples: int = 0,
    kernel_size: int = 100,
    stride: int = 1,
) -> str:
    """Create an interactive HTML page with navigation between strategies."""
    
    # Generate all figures and track which ones have errors
    figures_html = []
    error_strategies = []
    for i, (name, strategy) in enumerate(strategies):
        fig, has_error = create_masking_figure(
            strategy, S, N, V,
            f"{main_title}: {name}",
            interpolation_buffer,
            skip_n_samples,
            kernel_size,
            stride
        )
        # Convert to HTML div
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"figure_{i}")
        figures_html.append(fig_html)
        error_strategies.append(has_error)
    
    # Create dropdown options with red color for error strategies
    dropdown_options = []
    for i, (name, _) in enumerate(strategies):
        style = 'color: red; font-weight: bold;' if error_strategies[i] else ''
        dropdown_options.append(f'<option value="{i}" style="{style}">{name}{"⚠️" if error_strategies[i] else ""}</option>')
    
    # Create the HTML template
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{main_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .navigation {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            gap: 20px;
        }}
        .nav-button {{
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }}
        .nav-button:hover {{
            background-color: #45a049;
        }}
        .nav-button:disabled {{
            background-color: #cccccc;
            cursor: not-allowed;
        }}
        .nav-button.error {{
            background-color: #f44336;
        }}
        .nav-button.error:hover {{
            background-color: #da190b;
        }}
        .counter {{
            font-size: 18px;
            font-weight: bold;
        }}
        .counter.error {{
            color: red;
        }}
        .figure-container {{
            position: relative;
            width: 100%;
            min-height: 600px;
        }}
        .figure {{
            display: none;
        }}
        .figure.active {{
            display: block;
        }}
        .strategy-selector {{
            margin: 0 20px;
        }}
        select {{
            padding: 8px;
            font-size: 16px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .error-option {{
            color: red !important;
            font-weight: bold !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center;">{main_title}</h1>
        <div class="navigation">
            <button class="nav-button" id="prevBtn" onclick="changeStrategy(-1)">← Previous</button>
            <span class="counter" id="counter">1 / {len(strategies)}</span>
            <button class="nav-button" id="nextBtn" onclick="changeStrategy(1)">Next →</button>
            <div class="strategy-selector">
                <label for="strategySelect">Jump to: </label>
                <select id="strategySelect" onchange="jumpToStrategy(this.value)">
                    {"".join(dropdown_options)}
                </select>
            </div>
        </div>
        <div class="figure-container">
            {"".join([f'<div class="figure{" active" if i == 0 else ""}" id="figure_{i}">{fig}</div>' for i, fig in enumerate(figures_html)])}
        </div>
    </div>
    
    <script>
        let currentIndex = 0;
        const totalStrategies = {len(strategies)};
        const errorStrategies = {str(error_strategies).lower()};  // Convert to JavaScript boolean array
        
        function updateButtonStyles(index) {{
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const counter = document.getElementById('counter');
            
            // Update counter color based on error status
            if (errorStrategies[index]) {{
                counter.classList.add('error');
            }} else {{
                counter.classList.remove('error');
            }}
            
            // Update button colors for current strategy error status
            if (errorStrategies[index]) {{
                prevBtn.classList.add('error');
                nextBtn.classList.add('error');
            }} else {{
                prevBtn.classList.remove('error');
                nextBtn.classList.remove('error');
            }}
        }}
        
        function showStrategy(index) {{
            // Hide all figures
            document.querySelectorAll('.figure').forEach(fig => {{
                fig.classList.remove('active');
            }});
            
            // Show selected figure
            document.getElementById('figure_' + index).classList.add('active');
            
            // Update counter
            const errorSymbol = errorStrategies[index] ? ' ⚠️' : '';
            document.getElementById('counter').textContent = (index + 1) + ' / ' + totalStrategies + errorSymbol;
            
            // Update select dropdown
            document.getElementById('strategySelect').value = index;
            
            // Update button states
            document.getElementById('prevBtn').disabled = (index === 0);
            document.getElementById('nextBtn').disabled = (index === totalStrategies - 1);
            
            // Update button styles based on error status
            updateButtonStyles(index);
            
            currentIndex = index;
        }}
        
        function changeStrategy(direction) {{
            const newIndex = currentIndex + direction;
            if (newIndex >= 0 && newIndex < totalStrategies) {{
                showStrategy(newIndex);
            }}
        }}
        
        function jumpToStrategy(index) {{
            showStrategy(parseInt(index));
        }}
        
        // Initialize first strategy
        showStrategy(0);
        
        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowLeft') {{
                changeStrategy(-1);
            }} else if (event.key === 'ArrowRight') {{
                changeStrategy(1);
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    if output_file is not None:
        with open(output_file, 'w+') as f:
            f.write(html_template)
        print(f"Saved interactive visualization: {output_file}")
    return html_template


def visualize_masking_strategies(
    strategies: Sequence[DictConfig | MaskingStrategy] | Mapping[str, DictConfig | MaskingStrategy],
    num_samples_per_block: int,
    num_frames_per_block: int,
    max_population_size: Optional[int] = None,
    n_neurons_multiple_of: Optional[int] = None,
    interpolation_buffer: Optional[int] = None,
    skip_n_samples: Optional[int] = None,
    response_stride_samples: Optional[int] = None,
    response_window_size_samples: Optional[int] = None,
    max_num_strategies: Optional[int] = None,
    output_file: Optional[Path] = None,
    title: str = "Masking Strategies",
) -> None:
    """Main function to visualize masking strategies."""
    
    # Get output directory from config or use default
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get max num strategies per config
    max_num_strategies = max_num_strategies
    
    # Get model parameters
    S = num_samples_per_block
    N = max_population_size
    if N is None:
        N = 8000
    nm = n_neurons_multiple_of
    if nm is not None:
        N = nm * (N // nm)
    V = num_frames_per_block
    interpolation_buffer = interpolation_buffer
    if interpolation_buffer is None:
        interpolation_buffer = 0
    skip_n_samples = skip_n_samples
    if skip_n_samples is None:
        skip_n_samples = 0
    stride = response_stride_samples
    if stride is None:
        stride = 1
    kernel_size = response_window_size_samples
    if kernel_size is None:
        kernel_size = 100
    
    print(f"Samples per block (S): {S}")
    print(f"Max population size (N): {N}")
    print(f"Interpolation buffer: {interpolation_buffer}")
    print(f"Skip n samples: {skip_n_samples}")
        
    # Convert to list of MaskingStrategy objects
    strategy_list = []
    if isinstance(strategies, Mapping):
        strategies = list(strategies.items())
    elif isinstance(strategies, Sequence):
        strategies = [(f'Strategy {i+1}', strategy) for i, strategy in enumerate(strategies)]
    else:
        raise ValueError(f"Invalid strategies type: {type(strategies)}")
    for idx, (name, strategy_cfg) in enumerate(strategies):
        if max_num_strategies is not None and idx >= max_num_strategies:
            break
        if not isinstance(strategy_cfg, MaskingStrategy):
            strategy = omegaconf_to_masking_strategy(strategy_cfg)
        else:
            strategy = strategy_cfg
        strategy_list.append((name, strategy))
    
    print(f"Found {len(strategy_list)} training strategies")
    
    # Create interactive HTML
    html_template = create_interactive_html(
        strategy_list,
        S, N, V,
        title,
        output_file,
        interpolation_buffer,
        skip_n_samples,
        kernel_size,
        stride
    )
    
    return html_template