import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import ndimage
from skimage import measure
import plotly.graph_objects as go

from dataset.dataset import BraTSDataset
from models.hybrid_model import HybridUNetTransformer


# ---------------------------------------------------------------------------
# 2D helpers (unchanged behavior, kept from original)
# ---------------------------------------------------------------------------

def normalize_mri_for_display(image):
    """Map MRI intensity to a cleaner 0-1 display range."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image - low) / (high - low + 1e-8), 0, 1)
    return image


def pick_center_slice(mask):
    """Select a slice that contains the segmentation instead of a random middle slice."""
    if mask.ndim != 3:
        return mask.shape[0] // 2

    non_zero = np.argwhere(mask > 0)
    if len(non_zero) == 0:
        return mask.shape[0] // 2

    return int(np.median(non_zero[:, 0]))


def overlay_segmentation(mri_slice, seg_slice, alpha=0.45, color=(1.0, 0.4, 0.4)):
    """Overlay segmentation mask on MRI slice for a cleaner visual result."""
    h, w = mri_slice.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    seg_mask = seg_slice > 0
    if not np.any(seg_mask):
        return mri_slice

    overlay[..., 0] = np.where(seg_mask, color[0], 0.0)
    overlay[..., 1] = np.where(seg_mask, color[1], 0.0)
    overlay[..., 2] = np.where(seg_mask, color[2], 0.0)

    rgb_image = np.stack([mri_slice] * 3, axis=-1)
    return rgb_image * (1 - alpha) + overlay * alpha


# ---------------------------------------------------------------------------
# 3D helpers — replaced slow pure-python flood fill with scipy (instant),
# and voxel cubes with a smooth marching-cubes mesh you can rotate freely.
# ---------------------------------------------------------------------------

def keep_largest_connected_component(mask):
    """Keep only the largest connected tumor component and remove scattered noise.

    Uses scipy.ndimage.label — this is a fast C implementation, unlike the
    original pure-Python flood fill which could take minutes on real volumes.
    """
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask

    # 26-connectivity (matches the original's 3x3x3 neighborhood)
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1

    return labeled == largest_label


def build_tumor_mesh(segmentation_volume, smooth_sigma=0.6, keep_largest_only=False):
    """Convert a binary/label tumor volume into a smooth triangle mesh via
    marching cubes. Optionally restrict to the largest connected region.

    Returns (verts, faces, values) or None if there's nothing to show.
    """
    tumor_mask = (segmentation_volume > 0).astype(np.float32)

    if keep_largest_only:
        tumor_mask = keep_largest_connected_component(tumor_mask > 0).astype(np.float32)

    if not np.any(tumor_mask):
        return None

    # Light gaussian smoothing on the binary mask before marching cubes gives
    # a much cleaner, less "blocky" surface than raw voxel edges.
    smoothed = ndimage.gaussian_filter(tumor_mask, sigma=smooth_sigma)

    # Isosurface at 0.5 — the boundary between tumor / not-tumor after smoothing
    level = 0.5 if smoothed.max() > 0.5 else smoothed.max() * 0.5
    verts, faces, normals, values = measure.marching_cubes(smoothed, level=level)

    return verts, faces, normals, values


def plot_3d_tumor_interactive(
    segmentation_volume,
    title="3D Tumor Shape",
    output_html="tumor_3d_interactive.html",
    keep_largest_only=False,
    color="crimson",
):
    """Render a smooth, freely-rotatable (360°, zoomable, pannable) 3D tumor
    mesh as a standalone interactive HTML file using Plotly.

    Open the saved .html in any browser and drag to rotate, scroll to zoom.
    """
    mesh = build_tumor_mesh(segmentation_volume, keep_largest_only=keep_largest_only)
    if mesh is None:
        print("⚠️ No tumor voxels found; skipping 3D visualization.")
        return None

    verts, faces, normals, values = mesh
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=color,
                opacity=1.0,
                lighting=dict(
                    ambient=0.35,
                    diffuse=0.9,
                    specular=0.6,
                    roughness=0.35,
                    fresnel=0.2,
                ),
                lightposition=dict(x=200, y=200, z=200),
                flatshading=False,
                name="Tumor",
                showscale=False,
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title="Depth", visible=True, backgroundcolor="white"),
            yaxis=dict(title="Height", visible=True, backgroundcolor="white"),
            zaxis=dict(title="Width", visible=True, backgroundcolor="white"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=60, b=0),
        height=800,
    )

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"✅ Interactive 3D tumor (rotate/zoom freely) saved as '{output_html}'")

    # Also save a static high-res snapshot for quick preview / reports
    try:
        static_path = output_html.replace(".html", "_preview.png")
        fig.write_image(static_path, width=1200, height=1000, scale=2)
        print(f"✅ Static preview image saved as '{static_path}'")
    except Exception as e:
        # kaleido might not be installed; the HTML is the primary deliverable anyway
        print(f"ℹ️ Skipped static PNG export ({e}). The interactive HTML still works.")

    return fig


def plot_3d_tumor_multiclass(
    segmentation_volume,
    class_colors=None,
    class_names=None,
    title="3D Tumor Sub-Regions",
    output_html="tumor_3d_multiclass.html",
):
    """Render each tumor sub-region (e.g. edema, enhancing, necrotic core) as
    its own colored, independently togglable mesh in the same interactive
    scene. Useful for BraTS-style multi-class masks (labels 0-4).
    """
    labels_present = [int(l) for l in np.unique(segmentation_volume) if l != 0]
    if not labels_present:
        print("⚠️ No tumor voxels found; skipping 3D visualization.")
        return None

    default_colors = {1: "gold", 2: "royalblue", 3: "crimson", 4: "limegreen"}
    default_names = {1: "Necrotic Core", 2: "Edema", 3: "Enhancing Tumor", 4: "Other"}
    class_colors = class_colors or default_colors
    class_names = class_names or default_names

    traces = []
    for label in labels_present:
        region = (segmentation_volume == label).astype(np.float32)
        if not np.any(region):
            continue
        smoothed = ndimage.gaussian_filter(region, sigma=0.6)
        if smoothed.max() < 0.5:
            continue
        verts, faces, normals, values = measure.marching_cubes(smoothed, level=0.5)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        traces.append(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                color=class_colors.get(label, "gray"),
                opacity=0.92,
                name=class_names.get(label, f"Class {label}"),
                lighting=dict(ambient=0.35, diffuse=0.9, specular=0.6, roughness=0.35, fresnel=0.2),
                lightposition=dict(x=200, y=200, z=200),
                flatshading=False,
                showlegend=True,
            )
        )

    if not traces:
        print("⚠️ No renderable tumor sub-regions found.")
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis_title="Depth",
            yaxis_title="Height",
            zaxis_title="Width",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        legend=dict(itemsizing="constant"),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=60, b=0),
        height=800,
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"✅ Interactive multi-class 3D tumor saved as '{output_html}'")
    return fig


# Keep the original matplotlib static function too, in case you still want
# a quick non-interactive PNG snapshot without opening a browser.
def plot_3d_tumor_shape_static(segmentation_volume, title="3D Tumor Shape"):
    """Static matplotlib fallback (fixed viewing angle, not rotatable)."""
    tumor_mask = keep_largest_connected_component(segmentation_volume > 0)
    if not np.any(tumor_mask):
        print("⚠️ No tumor voxels found; skipping 3D visualization.")
        return

    coords = np.argwhere(tumor_mask)
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0)
    cropped = tumor_mask[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.voxels(
        cropped,
        facecolors=[1.0, 0.08, 0.08, 0.9],
        edgecolors="maroon",
        linewidth=0.08,
        alpha=0.85,
        shade=False,
    )
    ax.set_title(title, pad=22, fontsize=16)
    ax.set_xlabel("Depth", labelpad=12)
    ax.set_ylabel("Height", labelpad=12)
    ax.set_zlabel("Width", labelpad=12)
    ax.view_init(elev=22, azim=38)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_xlim(0, cropped.shape[0])
    ax.set_ylim(0, cropped.shape[1])
    ax.set_zlim(0, cropped.shape[2])

    output_path = "tumor_3d_shape.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Static 3D tumor shape saved as '{output_path}'")


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HybridUNetTransformer(in_channels=4, num_classes=5)
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded!")

    dataset = BraTSDataset("data", image_size=(96, 96, 96))
    print(f"\n📊 Total patients: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ No patients found!")
        return

    image, mask = dataset[0]
    print(f"📷 Image shape: {image.shape}")
    print(f"🎯 Mask shape: {mask.shape}")

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)

    prediction = prediction.cpu().numpy().squeeze()
    mask = mask.numpy().astype(np.int64)

    print(f"\n✅ Prediction shape: {prediction.shape}")
    print(f"🎯 Real mask shape: {mask.shape}")

    slice_idx = pick_center_slice(mask)
    print(f"📌 Selected slice: {slice_idx}")

    mri_volume = image[0].cpu().numpy()
    mri_slice = mri_volume[:, slice_idx, :, :]
    mri_slice = mri_slice.mean(axis=0)
    mri_slice = normalize_mri_for_display(mri_slice)

    pred_slice = prediction[slice_idx]
    target_slice = mask[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].set_title("MRI Input")
    axes[0].axis("off")

    axes[1].imshow(mri_slice, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(pred_slice == 0, pred_slice > 0),
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    axes[2].imshow(mri_slice, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(target_slice == 0, target_slice > 0),
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    output_path = "prediction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Clean prediction image saved as '{output_path}'")
    plt.close(fig)

    # --- New: smooth, freely-rotatable (360°) interactive 3D tumor view ---
    # Single solid-color mesh of the whole predicted tumor region:
    plot_3d_tumor_interactive(
        prediction,
        title="Predicted Tumor — 3D (drag to rotate, scroll to zoom)",
        output_html="tumor_3d_interactive.html",
        keep_largest_only=False,   # set True to strip small noisy islands
        color="crimson",
    )

    # Multi-class view: each BraTS sub-region (necrotic core / edema /
    # enhancing tumor) as its own colored, toggle-able layer:
    plot_3d_tumor_multiclass(
        prediction,
        title="Predicted Tumor — Sub-Regions (3D)",
        output_html="tumor_3d_multiclass.html",
    )

    correct = (prediction == mask).sum()
    total = prediction.size
    accuracy = correct / total * 100
    print(f"\n📈 Pixel-wise Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    predict()