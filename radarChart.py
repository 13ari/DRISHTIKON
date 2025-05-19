#!/usr/bin/env python3
"""
radarChart.py - Generate radar charts comparing model performance across attributes and states.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import glob
import re

# Directory structure
BASE_DIR = "/home/nemilai/Desktop/IITP/Outputs"
OUTPUT_DIRS = [
    "outputs_gemma",
    "outputs_janus",
    "outputs_chitrarth",
    "outputs_gpt4o_mini_all",
    "outputs_intern_llm",
    "outputs_intern_slm",
    "outputs_kimivl",
    "outputs_llama_llm",
    "outputs_llava_llm",
    "outputs_maya",
    "outputs_qwenomni",
    "outputs_qwenvl",
    "outputs_smol",
]

# Define radar chart function
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)
            
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            
        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return plt.Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return mpatches.RegularPolygon((0.5, 0.5), num_vars,
                                          radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
                
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon returns a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    
    register_projection(RadarAxes)
    return theta

def get_model_name_from_dir(dir_name):
    """Extract model name from directory name."""
    # Remove 'outputs_' prefix if it exists
    if dir_name.startswith('outputs_'):
        return dir_name[len('outputs_'):]
    return dir_name

# List of valid attributes as specified by the user
VALID_ATTRIBUTES = [
    "Art", "Costume", "Cuisine", "Cultural Common Sense", "Dance and Music",
    "Festivals", "History", "Language", "Medicine", "Nightlife", "Personalities",
    "Religion", "Rituals and Ceremonies", "Sports", "Tourism", "Transport"
]

def clean_category_name(name):
    """Clean category or state names by replacing underscores with spaces"""
    return name.replace('_', ' ') if name else name

def load_and_process_data():
    """Load all CSV files and process data for visualization."""
    model_data = {}
    all_attributes = set()
    all_states = set()
    
    # Loop through each model output directory
    for output_dir in OUTPUT_DIRS:
        full_dir_path = os.path.join(BASE_DIR, output_dir)
        if not os.path.exists(full_dir_path):
            print(f"Directory {full_dir_path} does not exist, skipping.")
            continue
            
        model_name = get_model_name_from_dir(output_dir)
        model_data[model_name] = {
            'by_attribute': {},
            'by_state': {}
        }
        
        # Find all CSV files in this directory
        csv_files = glob.glob(os.path.join(full_dir_path, "*.csv"))
        
        # Process each CSV file
        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        if not all_dfs:
            print(f"No valid CSV files found in {full_dir_path}, skipping.")
            continue
            
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter for only valid attributes
        print(f"Found attributes: {combined_df['attribute'].unique()}")
        
        # Calculate accuracy by attribute, but only for valid attributes
        attribute_groups = combined_df.groupby('attribute')
        for attribute, group in attribute_groups:
            clean_attr = clean_category_name(attribute)
            if clean_attr in VALID_ATTRIBUTES or attribute in VALID_ATTRIBUTES:
                accuracy = group['predicted_correctly'].mean()
                # Use clean attribute name for consistency
                model_data[model_name]['by_attribute'][clean_attr] = accuracy
                all_attributes.add(clean_attr)
        
        # Calculate accuracy by state
        state_groups = combined_df.groupby('state')
        for state, group in state_groups:
            clean_state = clean_category_name(state)
            accuracy = group['predicted_correctly'].mean()
            model_data[model_name]['by_state'][clean_state] = accuracy
            all_states.add(clean_state)
    
    # Use only the valid attributes, even if some weren't found in the data
    valid_attrs_found = [attr for attr in VALID_ATTRIBUTES if attr in all_attributes]
    print(f"Including {len(valid_attrs_found)} valid attributes in chart")
    
    return model_data, valid_attrs_found, sorted(list(all_states))

def create_unique_initials(categories):
    """Create unique initials for each category, even when there are duplicates."""
    initials = {}
    used_initials = set()
    
    for category in categories:
        words = category.split()
        if len(words) > 1:
            # For multi-word categories, use first letter of each word
            initial = ''.join(word[0].upper() for word in words)
        else:
            # For single-word categories, use first letter
            initial = category[0].upper()
        
        # If initial is already used, try next letter(s)
        original_initial = initial
        i = 1
        while initial in used_initials and i < len(category):
            # Try next letter in the category name
            initial = original_initial + category[i].upper()
            i += 1
        
        # If we still have a duplicate, add a number
        j = 1
        while initial in used_initials:
            initial = original_initial + str(j)
            j += 1
        
        initials[category] = initial
        used_initials.add(initial)
    
    return initials

def create_radar_chart(model_data, categories, chart_type):
    """Create a radar chart comparing models across categories (attributes or states)."""
    n_cats = len(categories)
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create larger figure for research paper quality
    # Create larger figure with extra padding for labels
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))
    
    # Remove frame border which might clip labels
    ax.spines['polar'].set_visible(False)
    
    # Increase distance between the plot edge and the figure edge
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create unique initials for categories
    initials_map = create_unique_initials(categories)
    wrapped_labels = []
    
    # Process ALL category labels - replace underscores and wrap properly
    for category in categories:
        # First replace any underscores with spaces
        clean_category = category.replace('_', ' ')
        
        # Split into words for better line breaks
        words = clean_category.split()
        
        if len(words) == 1 and len(clean_category) > 10:
            # For single long words, just use the word as is - don't break within words
            wrapped_labels.append(clean_category)
        elif len(clean_category) > 10:
            # For multi-word categories, aim for balanced lines
            max_chars_per_line = 10  # Reduced to ensure labels stay outside
            current_line = []
            wrapped_label = []
            
            for word in words:
                # Check if adding the next word exceeds the limit
                current_line_len = sum(len(w) for w in current_line) + len(current_line) - 1 if current_line else 0
                
                if current_line_len + len(word) + 1 <= max_chars_per_line:
                    current_line.append(word)
                else:
                    # Start a new line if the limit is exceeded
                    if current_line:  # Only append if there are words in the current line
                        wrapped_label.append(" ".join(current_line))
                    current_line = [word]
            
            # Append the last line
            if current_line:
                wrapped_label.append(" ".join(current_line))
            
            wrapped_labels.append("\n".join(wrapped_label))
        else:
            wrapped_labels.append(clean_category)
    
    # Custom colors for better differentiation between models
    custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    # Prepare data for plotting
    for i, (model_name, data) in enumerate(model_data.items()):
        if chart_type == 'attribute':
            category_data = data['by_attribute']
        else:  # state
            category_data = data['by_state']
        
        # Prepare values for plotting with loop closing
        values = [category_data.get(cat, 0) for cat in categories]
        values_closed = values + [values[0]]  # Close the loop
        
        # Plot with custom color
        ax.plot(angles, values_closed, 
                color=custom_colors[i % len(custom_colors)], 
                linewidth=2, 
                label=model_name)
        ax.fill(angles, values_closed, 
                facecolor=custom_colors[i % len(custom_colors)], 
                alpha=0.25)
    
    # Set up the chart ticks and labels
    ax.set_xticks(angles[:-1])  # Set ticks at each category
    ax.set_xticklabels(wrapped_labels, fontsize=13, wrap=False)
    
    # Create a background circle slightly larger than the data circle to mask grid lines
    # that might extend beyond the data circle
    bg_circle = plt.Circle((0, 0), 1.05, transform=ax.transData._b, fill=True, 
                          color='white', alpha=0.8, zorder=0)
    ax.add_artist(bg_circle)
    
    # Position labels around the radar chart with proper alignment
    # Get existing tick positions
    for i, label in enumerate(ax.get_xticklabels()):
        # Get angle in radians
        angle_rad = angles[i] 
        
        # Determine horizontal alignment based on position
        if 0 <= angle_rad <= np.pi/2 or angle_rad >= 3*np.pi/2:
            # Right half - left align text
            ha = 'left'
        else:
            # Left half - right align text
            ha = 'right'
        
        # Set the horizontal alignment
        label.set_ha(ha)
    
    # Increase padding to push labels further out
    ax.tick_params(axis='x', pad=30, labelsize=13)
    
    # Configure y-axis grid and labels (accuracy percentage)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)
    ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
    ax.yaxis.grid(True, color="gray", linestyle="dotted", linewidth=0.8)
    
    # Add title
    if chart_type == 'attribute':
        title = "Model Accuracies Across Different Attributes"
    else:
        title = "Model Accuracies Across Different States of India"
    ax.set_title(title, size=16, y=1.1, fontweight='bold')
    
    # Add legend at the bottom center with multiple columns
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), 
              fontsize=15, 
              ncol=min(4, len(model_data)))
    
    # Save figure
    plt.tight_layout()
    output_filename = f"radar_chart_by_{chart_type}.png"
    plt.savefig(os.path.join(BASE_DIR, output_filename), dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    
    # Also save PDF for publication-quality graphics
    pdf_filename = f"radar_chart_by_{chart_type}.pdf"
    plt.savefig(os.path.join(BASE_DIR, pdf_filename), format='pdf', bbox_inches='tight')
    print(f"Saved {pdf_filename}")
    
    plt.close()

def main():
    print("Loading and processing data...")
    model_data, all_attributes, all_states = load_and_process_data()
    
    if not model_data:
        print("No data was processed. Check if the directories exist and contain valid CSV files.")
        return
    
    print(f"Creating radar charts for {len(model_data)} models...")
    
    # Create radar chart by attribute
    create_radar_chart(model_data, all_attributes, 'attribute')
    
    # Create radar chart by state
    create_radar_chart(model_data, all_states, 'state')
    
    print("Done!")

if __name__ == "__main__":
    main()
