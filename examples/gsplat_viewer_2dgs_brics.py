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
        *,
        base_dir: PathLike,
        section_label: str = "Input",
        on_select_dir: Callable[[Path], None] | None = None,
        default_date: str | None = None,
        default_multiseq: str | None = None,
    ) -> None:
        self._on_select_dir = on_select_dir
        self._base_dir = Path(base_dir).expanduser()

        # Scan dates and multisequences
        self._date_to_multis: Dict[str, List[str]] = {}
        date_labels = self._scan_base_dir()

        if not date_labels:
            # Fallback to provided output_dir only
            date_labels = ["unknown"]
            self._date_to_multis["unknown"] = ["multisequence0"]

        # Choose defaults
        cur_date = default_date if default_date in date_labels else date_labels[-1]
        multis_for_date = self._date_to_multis.get(cur_date, [])
        if not multis_for_date:
            multis_for_date = ["multisequence0"]
        cur_multi = (
            default_multiseq if default_multiseq in multis_for_date else multis_for_date[-1]
        )

        # Resolve current gsplat directory
        cur_gsplat_dir = self._resolve_gsplat_dir(cur_date, cur_multi)
        if cur_gsplat_dir is None:
            cur_gsplat_dir = output_dir

        # Add a top-level Status section for prominent visibility
        status_folder = server.gui.add_folder("Status")
        with status_folder:
            status_banner = server.gui.add_text(
                "STATUS",
                initial_value="🟢 Idle",
                disabled=False,
                hint="Overall viewer status.",
            )

        # Build the Input section BEFORE initializing base UI (keeps it near top)
        input_folder = server.gui.add_folder(section_label)
        with input_folder:
            base_dir_text = server.gui.add_text(
                "Base Directory",
                initial_value=str(self._base_dir),
                disabled=True,
                hint="Root folder scanned for dates/multisequences.",
            )
            date_dropdown = server.gui.add_dropdown(
                "Date",
                tuple(date_labels),
                initial_value=cur_date,
                hint="Date folder (YYYY-MM-DD)",
            )
            multi_dropdown = server.gui.add_dropdown(
                "Multisequence",
                tuple(multis_for_date),
                initial_value=cur_multi,
                hint="Select multisequence under the chosen date.",
            )
            cur_path_text = server.gui.add_text(
                "Current Directory",
                initial_value=str(cur_gsplat_dir),
                disabled=True,
                hint="Selected <date>/<multisequence>/gsplat_2dgs path.",
            )
            ckpt_number = server.gui.add_number(
                "Checkpoint",
                initial_value=0,
                disabled=True,
                hint="Highest ckpt_<number>.pt loaded from the directory.",
            )

            @date_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                # Update multisequence choices for the new date
                new_date = date_dropdown.value
                new_multis = tuple(self._date_to_multis.get(new_date, []))
                try:
                    # Update dropdown choices and reset value
                    multi_dropdown.choices = new_multis  # type: ignore[attr-defined]
                except Exception:
                    try:
                        multi_dropdown.options = new_multis  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if new_multis:
                    try:
                        multi_dropdown.value = new_multis[-1]  # type: ignore[attr-defined]
                    except Exception:
                        pass
                sel_multi = getattr(multi_dropdown, "value", None) or (
                    new_multis[-1] if new_multis else None
                )
                new_dir = self._resolve_gsplat_dir(new_date, sel_multi)
                if new_dir is not None:
                    self.output_dir = new_dir
                    cur_path_text.value = str(self.output_dir)
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass

            @multi_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                new_date = date_dropdown.value
                new_multi = multi_dropdown.value
                new_dir = self._resolve_gsplat_dir(new_date, new_multi)
                if new_dir is not None:
                    self.output_dir = new_dir
                    cur_path_text.value = str(self.output_dir)
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass

        # Initialize the base viewer so Input section stays on top
        super().__init__(server, render_fn, cur_gsplat_dir, mode)

        # Keep references
        self._input_folder = input_folder
        self._output_dir_handles = {
            "status_banner": status_banner,
            "base_dir_text": base_dir_text,
            "date_dropdown": date_dropdown,
            "multi_dropdown": multi_dropdown,
            "cur_path_text": cur_path_text,
            "ckpt_number": ckpt_number,
        }

    def set_checkpoint_number(self, number: int | None) -> None:
        try:
            h = self._output_dir_handles.get("ckpt_number")  # type: ignore[attr-defined]
            if h is None:
                return
            h.value = int(number) if number is not None else 0
        except Exception:
            # UI update failures should not break the viewer
            pass

    def set_loading(self, loading: bool, msg: str | None = None) -> None:
        try:
            base_text = msg if msg is not None else ("Loading..." if loading else "Idle")
            # Simple heuristic for an emoji prefix
            if loading:
                prefix = "🟡"
            else:
                low = (base_text or "").lower()
                if any(s in low for s in ("error", "fail", "no ckpt", "no ckpts", "not found", "missing")):
                    prefix = "🔴"
                else:
                    prefix = "🟢"
            text = f"{prefix} {base_text}"
            sb = self._output_dir_handles.get("status_banner")  # type: ignore[attr-defined]
            if sb is not None:
                sb.value = text
            # Optionally disable selectors while loading
            dd = self._output_dir_handles.get("date_dropdown")
            md = self._output_dir_handles.get("multi_dropdown")
            for w in (dd, md):
                if w is None:
                    continue
                try:
                    w.disabled = bool(loading)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    # Utility to find a label given a path value
    def _label_for_path(self, p: PathLike) -> str:
        # Legacy helper retained for compatibility; unused in new base-dir mode.
        return str(Path(p))

    def _resolve_gsplat_dir(self, date_label: str | None, multi_label: str | None) -> Path | None:
        if date_label is None or multi_label is None:
            return None
        d = self._base_dir / date_label / multi_label / "gsplat_2dgs"
        try:
            return d.resolve()
        except Exception:
            return d

    def _scan_base_dir(self) -> List[str]:
        self._date_to_multis.clear()
        date_labels: List[str] = []
        try:
            it = self._base_dir.iterdir() if self._base_dir.exists() else []
            for d in sorted(it):
                if not d.is_dir():
                    continue
                name = d.name
                if (
                    len(name) == 10
                    and name[4] == "-"
                    and name[7] == "-"
                    and name.replace("-", "").isdigit()
                ):
                    multis: List[str] = []
                    for m in sorted(d.iterdir()):
                        if not m.is_dir():
                            continue
                        if m.name.startswith("multisequence") and (m / "gsplat_2dgs").exists():
                            multis.append(m.name)
                    if multis:
                        date_labels.append(name)
                        self._date_to_multis[name] = multis
        except Exception:
            pass
        return date_labels

    def refresh_base_dir(self) -> None:
        """Rescan base_dir and update dropdowns. Keep current selection if possible.

        If current selection disappears, pick the last available. If selection
        changes, update current directory and invoke on_select_dir.
        """
        try:
            prev_date = getattr(self._output_dir_handles.get("date_dropdown"), "value", None)
            prev_multi = getattr(self._output_dir_handles.get("multi_dropdown"), "value", None)
        except Exception:
            prev_date = None
            prev_multi = None

        date_labels = self._scan_base_dir()
        if not date_labels:
            return

        dd = self._output_dir_handles.get("date_dropdown")
        md = self._output_dir_handles.get("multi_dropdown")
        cur_path_text = self._output_dir_handles.get("cur_path_text")

        # Update date dropdown choices
        new_dates = tuple(date_labels)
        try:
            dd.choices = new_dates  # type: ignore[attr-defined]
        except Exception:
            try:
                dd.options = new_dates  # type: ignore[attr-defined]
            except Exception:
                pass

        # Choose date
        cur_date = prev_date if prev_date in date_labels else date_labels[-1]
        try:
            dd.value = cur_date  # type: ignore[attr-defined]
        except Exception:
            pass

        # Update multisequence choices
        multis = self._date_to_multis.get(cur_date, [])
        new_multis = tuple(multis)
        try:
            md.choices = new_multis  # type: ignore[attr-defined]
        except Exception:
            try:
                md.options = new_multis  # type: ignore[attr-defined]
            except Exception:
                pass
        cur_multi = prev_multi if prev_multi in multis else (multis[-1] if multis else None)
        if cur_multi is None:
            return
        try:
            md.value = cur_multi  # type: ignore[attr-defined]
        except Exception:
            pass

        # Resolve and update current directory
        new_dir = self._resolve_gsplat_dir(cur_date, cur_multi)
        if new_dir is None:
            return
        if Path(self.output_dir) != new_dir:
            self.output_dir = new_dir
            try:
                cur_path_text.value = str(self.output_dir)
            except Exception:
                pass
            if self._on_select_dir is not None:
                try:
                    self._on_select_dir(self.output_dir)
                except Exception:
                    pass
