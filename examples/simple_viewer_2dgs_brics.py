import argparse
import math
import time
import threading
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import viser
import numpy as np

from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs
from gsplat_viewer_2dgs_brics import GsplatViewerBrics
from nerfview import CameraState, RenderTabState, apply_float_colormap

REFRESH_INTERVAL = 600

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
        "ckpt_num": None,
        "dir": None,
    }

    # Lightweight LRU cache of loaded scenes to avoid reloading when toggling
    CACHE_SIZE = 2
    cache = OrderedDict()  # key: str(path), value: dict like data
    load_lock = threading.Lock()
    load_token = {"id": 0}  # increments for each new request to cancel stale loads

    def _find_best_ckpt(dir_path: Path):
        import re
        ckpts_dir = Path(dir_path) / "ckpts"
        if not ckpts_dir.exists():
            return None, -1
        best_num = -1
        best_path = None
        for p in ckpts_dir.glob("ckpt_*.pt"):
            m = re.match(r"^ckpt_(\d+)(?:_rank\d+)?\.pt$", p.name)
            if not m:
                continue
            n = int(m.group(1))
            if n > best_num:
                best_num = n
                best_path = p
        return best_path, best_num

    def _do_load(dir_path: Path, token_id: int):
        # Resolve and check best ckpt
        best_path, best_num = _find_best_ckpt(dir_path)
        if best_path is None:
            print(f"[viewer] No matching ckpt_*.pt found under: {dir_path}/ckpts")
            if viewer is not None:
                viewer.set_loading(False, f"No ckpt_*.pt in {dir_path}/ckpts")
            return

        # Short-circuit if already up-to-date for this dir
        if data.get("dir") == str(dir_path) and data.get("ckpt_num") == best_num:
            if viewer is not None:
                viewer.set_loading(False, f"Already loaded ckpt_{best_num}.pt")
                try:
                    viewer.set_checkpoint_number(best_num)
                except Exception:
                    pass
            return

        key = str(Path(dir_path).resolve())
        # Try cache
        if key in cache:
            cached = cache[key]
            if cached.get("ckpt_num") == best_num:
                with load_lock:
                    if token_id != load_token["id"]:
                        return
                    data.update(cached)
                if viewer is not None:
                    viewer.set_loading(False, f"Cached ckpt_{best_num}.pt")
                    try:
                        viewer.set_checkpoint_number(best_num)
                        viewer.rerender(None)
                    except Exception:
                        pass
                return

        # Load on CPU then move to GPU non-blocking
        try:
            ckpt_cpu = torch.load(best_path, map_location="cpu")["splats"]
            # Move and process on GPU
            means = ckpt_cpu["means"].to(device, non_blocking=True)
            quats = F.normalize(ckpt_cpu["quats"].to(device, non_blocking=True), p=2, dim=-1)
            scales = torch.exp(ckpt_cpu["scales"].to(device, non_blocking=True))
            opacities = torch.sigmoid(ckpt_cpu["opacities"].to(device, non_blocking=True))
            sh0 = ckpt_cpu["sh0"].to(device, non_blocking=True)
            shN = ckpt_cpu["shN"].to(device, non_blocking=True)
            colors = torch.cat([sh0, shN], dim=-2)
            sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        except Exception as e:
            print(f"[viewer] Load failed: {e}")
            if viewer is not None:
                viewer.set_loading(False, f"Error: {e}")
            return

        with load_lock:
            if token_id != load_token["id"]:
                return
            data.update(
                {
                    "means": means,
                    "quats": quats,
                    "scales": scales,
                    "opacities": opacities,
                    "colors": colors,
                    "sh_degree": sh_degree,
                    "ckpt_num": best_num,
                    "dir": key,
                }
            )
            cache[key] = {k: data[k] for k in ("means","quats","scales","opacities","colors","sh_degree","ckpt_num","dir")}
            # Enforce LRU size
            while len(cache) > CACHE_SIZE:
                cache.popitem(last=False)

        print("[viewer] Loaded:", best_path)
        if viewer is not None:
            viewer.set_loading(False, f"Loaded {best_path.name}")
            try:
                viewer.set_checkpoint_number(best_num)
                viewer.rerender(None)
            except Exception:
                pass

    def request_load(dir_path: Path):
        # Announce loading and kick a background thread; cancel older loads via token
        if viewer is not None:
            viewer.set_loading(True, f"Loading from {dir_path}...")
        with load_lock:
            load_token["id"] += 1
            token_id = load_token["id"]
        th = threading.Thread(target=_do_load, args=(dir_path, token_id), daemon=True)
        th.start()

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        # If no data loaded yet, return a solid background to avoid crashes
        if data["means"] is None:
            bg = np.array(render_tab_state.backgrounds, dtype=np.float32) / 255.0
            bg = bg.reshape(1, 1, 3)
            return np.tile(bg, (height, width, 1))
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

    # Resolve an initial gsplat_2dgs directory from base_dir/defaults
    def _resolve_initial_dir(base_dir: Path, default_date: str | None, default_multiseq: str | None) -> Path | None:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return None
        # Collect dates
        dates = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
        def _is_date_dir(p: Path) -> bool:
            n = p.name
            return len(n) == 10 and n[4] == '-' and n[7] == '-' and n.replace('-', '').isdigit()
        date_dirs = [d for d in dates if _is_date_dir(d)]
        if not date_dirs:
            return None
        # Choose date
        chosen_date = None
        if default_date is not None:
            for d in date_dirs:
                if d.name == default_date:
                    chosen_date = d
                    break
        if chosen_date is None:
            chosen_date = date_dirs[-1]
        # Collect multisequences with gsplat_2dgs
        multis = [m for m in sorted(chosen_date.iterdir()) if m.is_dir() and m.name.startswith('multisequence') and (m / 'gsplat_2dgs').exists()]
        if not multis:
            return None
        chosen_multi = None
        if default_multiseq is not None:
            for m in multis:
                if m.name == default_multiseq:
                    chosen_multi = m
                    break
        if chosen_multi is None:
            chosen_multi = multis[-1]
        return (chosen_multi / 'gsplat_2dgs')

    initial_dir = _resolve_initial_dir(Path(args.base_dir), args.default_date, args.default_multiseq)

    server = viser.ViserServer(port=args.port, verbose=False)

    viewer = None  # will be set after construction

    def _on_select_dir(p: Path):
        request_load(p)

    viewer = GsplatViewerBrics(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=(initial_dir if initial_dir is not None else Path(args.base_dir)),
        mode="rendering",
        base_dir=Path(args.base_dir),
        default_date=args.default_date,
    default_multiseq=args.default_multiseq,
    max_dates=args.max_dates,
    max_multis=args.max_multis,
        on_select_dir=_on_select_dir,
    )
    # Periodically refresh base_dir every 10 minutes in the background
    def _periodic_refresh():
        while True:
            try:
                time.sleep(REFRESH_INTERVAL)
                if hasattr(viewer, "refresh_base_dir"):
                    viewer.refresh_base_dir()
                    print("[viewer] base_dir refreshed")
            except Exception:
                # Keep the thread alive despite transient issues
                pass

    t = threading.Thread(target=_periodic_refresh, daemon=True)
    t.start()
    # Kick initial load asynchronously
    if initial_dir is not None:
        request_load(initial_dir)
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing <YYYY-MM-DD>/multisequence*/gsplat_2dgs",
    )
    parser.add_argument(
        "--default_date",
        type=str,
        default=None,
        help="Optional default date (YYYY-MM-DD) to preselect.",
    )
    parser.add_argument(
        "--default_multiseq",
        type=str,
        default=None,
        help="Optional default multisequence name to preselect (e.g., multisequence3).",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--max_dates",
        type=int,
        default=None,
        help="Optional cap on number of date folders shown (keep the latest N).",
    )
    parser.add_argument(
        "--max_multis",
        type=int,
        default=None,
        help="Optional cap on number of multisequences shown per date (keep latest N).",
    )
    args = parser.parse_args()

    cli(main, args, verbose=True)
