import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import viser

from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs
from gsplat_viewer_2dgs_brics import GsplatViewerBrics
from nerfview import CameraState, RenderTabState, apply_float_colormap


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    # Mutable container to hold the currently loaded scene
    data = {
        "means": None,
        "quats": None,
        "scales": None,
        "opacities": None,
        "colors": None,
        "sh_degree": None,
    }

    def load_from_dir(dir_path: Path) -> bool:
        ckpt_path = Path(dir_path) / "ckpts" / "ckpt_6999.pt"
        if not ckpt_path.exists():
            print(f"[viewer] Checkpoint not found: {ckpt_path}")
            return False
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means = ckpt["means"]
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)
        scales = torch.exp(ckpt["scales"])
        opacities = torch.sigmoid(ckpt["opacities"])
        sh0 = ckpt["sh0"]
        shN = ckpt["shN"]
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        data["means"] = means
        data["quats"] = quats
        data["scales"] = scales
        data["opacities"] = opacities
        data["colors"] = colors
        data["sh_degree"] = sh_degree
        print("[viewer] Loaded:", ckpt_path)
        return True

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=data["means"],
            quats=data["quats"],
            scales=data["scales"],
            opacities=data["opacities"],
            colors=data["colors"],
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, data["sh_degree"])
                if data["sh_degree"] is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
        )
        render_tab_state.total_gs_count = len(data["means"]) if data["means"] is not None else 0
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            depth = render_median
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals * 0.5 + 0.5
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders

    # Load the initial directory before starting the server
    initial_dir = Path(args.output_dir[0] if isinstance(args.output_dir, list) else args.output_dir)
    if not load_from_dir(initial_dir):
        raise FileNotFoundError(f"Initial checkpoint not found under: {initial_dir}/ckpts/ckpt_6999.pt")

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = GsplatViewerBrics(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=initial_dir,
        mode="rendering",
        selectable_output_dirs=args.output_dir,
        on_select_dir=lambda p: (load_from_dir(p), getattr(viewer, "rerender", lambda *_: None)(None))[0],
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="+",
        default=["results/"],
        help="One or more directories where viewer can write outputs; selectable at runtime.",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    args = parser.parse_args()

    cli(main, args, verbose=True)
