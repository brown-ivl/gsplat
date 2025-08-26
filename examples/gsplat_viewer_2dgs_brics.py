from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Tuple, Union

import viser

from gsplat_viewer_2dgs import GsplatRenderTabState, GsplatViewer as _BaseGsplatViewer


PathLike = Union[str, Path]


class GsplatViewerBrics(_BaseGsplatViewer):
    """
    Gsplat 2DGS Viewer with selectable directories.

    This subclass adds a small GUI section that lets users switch the
    viewer's directory at runtime among a pre-approved list supplied when
    constructing the viewer. The selection controls where inputs (e.g.,
    ckpts) are read from and where screenshots/exports are saved by the
    base viewer.

    Usage:
        server = viser.ViserServer(port=8080)
        viewer = GsplatViewerBrics(
            server=server,
            render_fn=render_fn,
            output_dir=Path("results/default"),
            selectable_output_dirs=[
                "results/default",
                "results/alt1",
                "/abs/path/elsewhere",
            ],
        )
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        selectable_output_dirs: Iterable[PathLike] | None = None,
        section_label: str = "Input",
    ) -> None:
        # Normalize and prepare selectable directories
        choices: List[Path] = []
        if selectable_output_dirs is not None:
            for p in selectable_output_dirs:
                try:
                    pp = Path(p).expanduser().resolve()
                except Exception:
                    # Fallback to Path without resolve if path doesn't exist yet
                    pp = Path(p).expanduser()
                if pp not in choices:
                    choices.append(pp)

        # Always include the initially provided output_dir as a choice (first)
        try:
            initial_dir = Path(output_dir).expanduser().resolve()
        except Exception:
            initial_dir = Path(output_dir).expanduser()
        if initial_dir not in choices:
            choices.insert(0, initial_dir)
        else:
            # Ensure the initial choice is the first
            choices = [initial_dir] + [c for c in choices if c != initial_dir]

        # Store choices and label map early so our helper can use them
        self._output_dir_choices = list(choices)
        self._label_to_path: Dict[str, Path] = {}
        used_labels: set[str] = set()
        for idx, p in enumerate(self._output_dir_choices):
            base = p.name or str(p)
            label = base
            if label in used_labels:
                label = f"{base} ({idx})"
            used_labels.add(label)
            self._label_to_path[label] = p

        # Build the small GUI to switch directory BEFORE initializing base UI,
        # so it appears at the top of the panel.
        input_folder = server.gui.add_folder(section_label)
        with input_folder:
            cur_path_text = server.gui.add_text(
                "Current Directory",
                initial_value=str(initial_dir),
                disabled=True,
                hint="Dir used by viewer. Loads ckpt from <dir>/ckpts/ckpt_6999.pt.",
            )
            dropdown = server.gui.add_dropdown(
                "Select Directory",
                tuple(self._label_to_path.keys()),
                initial_value=self._label_for_path(initial_dir),
                hint="Choose where to load inputs/save outputs.",
            )

            @dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                label = dropdown.value
                new_dir = self._label_to_path.get(label)
                if new_dir is None:
                    return
                try:
                    Path(new_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                self.output_dir = Path(new_dir)
                cur_path_text.value = str(self.output_dir)

        # Now initialize the base viewer so our Input section stays on top
        super().__init__(server, render_fn, output_dir, mode)

        # Keep a reference for potential future updates
        self._input_folder = input_folder
        self._output_dir_handles = {"dropdown": dropdown, "cur_path_text": cur_path_text}

    # Utility to find a label given a path value
    def _label_for_path(self, p: PathLike) -> str:
        p = Path(p)
        for label, path in self._label_to_path.items():
            # Resolve may raise; compare as-is first
            if path == p:
                return label
            try:
                if path.resolve() == p.resolve():
                    return label
            except Exception:
                # If resolve fails for either, skip
                pass
        # Fallback to first label
        return next(iter(self._label_to_path.keys()))
