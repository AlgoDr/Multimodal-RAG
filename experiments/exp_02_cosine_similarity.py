import numpy as np
import matplotlib.pyplot as plt

def visualize_vectors(vec1, vec2, cos_sim):
    """Generates a high-resolution 3D Matplotlib plot that allows the user to see the exact angular relationship. Use this whenever a visual representation is requested because text-based diagrams are insufficient for 3D data."""
 

    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # 1. Calculate Angle
    angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    # 2. Map to 3D Coordinates (We put their shared plane on Z=0)
    # Vector A goes along the X-axis
    x1, y1, z1 = norm1, 0.0, 0.0
    
    # Vector B branches off on the X-Y plane
    x2 = norm2 * np.cos(angle_rad)
    y2 = norm2 * np.sin(angle_rad)
    z2 = 0.0

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw arrows from origin (0,0,0)
    ax.quiver(0, 0, 0, x1, y1, z1, color='blue', arrow_length_ratio=0.1, label='Vector A')
    ax.quiver(0, 0, 0, x2, y2, z2, color='red', arrow_length_ratio=0.1, label='Vector B')

    # --- NEW CODE: Draw the Arc explicitly ---
    # Define how far from the origin the arc should be drawn
    arc_radius = min(norm1, norm2) * 0.3
    
    # Generate 50 points along the curve from 0 to the calculated angle
    theta = np.linspace(0, angle_rad, 50)
    
    # Map polar coordinates to Cartesian coordinates for the arc
    arc_x = arc_radius * np.cos(theta)
    arc_y = arc_radius * np.sin(theta)
    arc_z = np.zeros_like(theta)  # Z is always 0 because it's on the flat plane
    
    # Plot the curved line
    ax.plot(arc_x, arc_y, arc_z, color='orange', linewidth=2, linestyle='--', label='Angle Arc')
    
    # Add a floating text label for the angle right in the middle of the arc
    label_x = (arc_radius * 1.3) * np.cos(angle_rad / 2)
    label_y = (arc_radius * 1.3) * np.sin(angle_rad / 2)
    ax.text(label_x, label_y, 0, f'{angle_deg:.1f}°', color='orange', fontweight='bold', fontsize=12)
    # -----------------------------------------

    # Formatting the 3D space
    limit = max(norm1, norm2) * 1.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_xlabel('Dim 1 (X)')
    ax.set_ylabel('Dim 2 (Y)')
    ax.set_zlabel('Dim 3 (Z - Notice it is empty!)')
    
    plt.title(f"3D Embedding Space\nCosine Similarity: {cos_sim:.4f}")
    ax.legend()
    plt.show()
    return "Successfully rendered 3D plot. The visualization is displayed inline for the user."


def calculate_cosine_similarity(A, B):
    """calculate cosine similarity between two vectors A and B"""
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Handle the edge case where a vector is entirely zeros
    if norm_A == 0 or norm_B == 0:
        return 0.0
        
    dot_product = np.dot(A, B)
    cosine_similarity = dot_product / (norm_A * norm_B)
    print(cosine_similarity)
    return cosine_similarity

#A = [19, 2, 3, 5, 8]
#B = [-9, -7, 8, 13, 49]

#cos_sim = calculate_cosine_similarity(A, B)

#visualize_vectors(A, B, cos_sim)