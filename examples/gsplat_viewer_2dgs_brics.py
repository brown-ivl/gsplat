from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Tuple, Union

import viser

from gsplat_viewer_2dgs import GsplatRenderTabState, GsplatViewer as _BaseGsplatViewer


PathLike = Union[str, Path]


class GsplatViewerBrics(_BaseGsplatViewer):
    """
    Gsplat 2DGS Viewer with a fast, lazy-scanning Input panel for BRiCS folders.
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
        on_select_ckpt: Callable[[Path], None] | None = None,
        default_date: str | None = None,
        default_multiseq: str | None = None,
        max_dates: int | None = None,
        max_multis: int | None = None,
    ) -> None:
        self._on_select_dir = on_select_dir
        self._on_select_ckpt = on_select_ckpt
        self._base_dir = Path(base_dir).expanduser()

        # Caches for fast scanning
        self._date_to_multis = {}
        self._date_mtimes = {}
        self._base_mtime = None
        self._date_labels = []
        self._max_dates = max_dates
        self._max_multis = max_multis
        # For checkpoint dropdown label->path mapping
        self._ckpt_label_to_path = {}

        # Initial scan
        date_labels = self._scan_date_labels()
        if not date_labels:
            date_labels = ["unknown"]
            self._date_to_multis["unknown"] = []

        # Choose defaults
        cur_date = default_date if (default_date in date_labels) else date_labels[-1]
        multis_for_date = self._scan_multis_for_date(cur_date)
        cur_multi = (
            default_multiseq
            if (default_multiseq in multis_for_date)
            else (multis_for_date[-1] if multis_for_date else None)
        )

        # Resolve current gsplat directory
        cur_gsplat_dir = self._resolve_gsplat_dir(cur_date, cur_multi)
        if cur_gsplat_dir is None:
            cur_gsplat_dir = output_dir

        # Top-level Status section
        status_folder = server.gui.add_folder("Status")
        with status_folder:
            status_banner = server.gui.add_text(
                "STATUS",
                initial_value="🟢 Idle",
                disabled=False,
                hint="Overall viewer status.",
            )

        # Input section (kept near top)
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
            if multis_for_date:
                multi_dropdown = server.gui.add_dropdown(
                    "Multisequence",
                    tuple(multis_for_date),
                    initial_value=cur_multi,
                    hint="Select multisequence under the chosen date.",
                )
            else:
                multi_dropdown = server.gui.add_dropdown(
                    "Multisequence",
                    tuple(["<none>"]),
                    initial_value="<none>",
                    hint="No multisequence found under the chosen date.",
                )
                try:
                    multi_dropdown.disabled = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            ckpt_dropdown = server.gui.add_dropdown(
                "Checkpoint",
                tuple(["<none>"]),
                initial_value="<none>",
                hint="Select a checkpoint (ckpts/ckpt_*.pt or .pth) to load.",
            )
            # Ensure it's interactable initially
            try:
                ckpt_dropdown.disabled = False  # type: ignore[attr-defined]
            except Exception:
                pass

            @date_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                new_date = date_dropdown.value
                new_multis = tuple(self._scan_multis_for_date(new_date))
                if new_multis:
                    try:
                        multi_dropdown.choices = new_multis  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            multi_dropdown.options = new_multis  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    try:
                        multi_dropdown.value = new_multis[-1]  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        multi_dropdown.disabled = False  # type: ignore[attr-defined]
                    except Exception:
                        pass
                else:
                    try:
                        multi_dropdown.choices = tuple(["<none>"])  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            multi_dropdown.options = tuple(["<none>"])  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    try:
                        multi_dropdown.value = "<none>"  # type: ignore[attr-defined]
                        multi_dropdown.disabled = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                sel_multi = getattr(multi_dropdown, "value", None) or (
                    new_multis[-1] if new_multis else None
                )
                new_dir = self._resolve_gsplat_dir(new_date, sel_multi)
                if new_dir is not None:
                    self.output_dir = new_dir
                    # Update checkpoints for new dir
                    self._update_ckpt_dropdown(self.output_dir)
                    # Callbacks
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass
                    sel_label = getattr(ckpt_dropdown, "value", None)
                    if sel_label and sel_label != "<none>" and self._on_select_ckpt is not None:
                        try:
                            sel_path = self._ckpt_label_to_path.get(sel_label, sel_label)
                            self._on_select_ckpt(Path(sel_path))
                        except Exception:
                            pass

            @multi_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                new_date = date_dropdown.value
                new_multi = multi_dropdown.value
                new_dir = self._resolve_gsplat_dir(new_date, new_multi)
                if new_dir is not None:
                    self.output_dir = new_dir
                    # Update checkpoints for new dir
                    self._update_ckpt_dropdown(self.output_dir)
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass
                    sel_label = getattr(ckpt_dropdown, "value", None)
                    if sel_label and sel_label != "<none>" and self._on_select_ckpt is not None:
                        try:
                            sel_path = self._ckpt_label_to_path.get(sel_label, sel_label)
                            self._on_select_ckpt(Path(sel_path))
                        except Exception:
                            pass

            @ckpt_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                label = getattr(ckpt_dropdown, "value", None)
                if not label or label == "<none>":
                    return
                if self._on_select_ckpt is not None:
                    try:
                        path = self._ckpt_label_to_path.get(label, label)
                        self._on_select_ckpt(Path(path))
                    except Exception:
                        pass

        # Initialize base viewer
        super().__init__(server, render_fn, cur_gsplat_dir, mode)

        # Keep references
        self._input_folder = input_folder
        self._output_dir_handles = {
            "status_banner": status_banner,
            "base_dir_text": base_dir_text,
            "date_dropdown": date_dropdown,
            "multi_dropdown": multi_dropdown,
            "ckpt_dropdown": ckpt_dropdown,
        }

        # Populate checkpoint list initially
        try:
            self._update_ckpt_dropdown(cur_gsplat_dir)
        except Exception:
            pass

    def set_checkpoint_number(self, number: int | None) -> None:
        try:
            h = self._output_dir_handles.get("ckpt_number")  # type: ignore[attr-defined]
            if h is None:
                return
            h.value = int(number) if number is not None else 0
        except Exception:
            pass

    def set_loading(self, loading: bool, msg: str | None = None) -> None:
        try:
            base_text = msg if msg is not None else ("Loading..." if loading else "Idle")
            if loading:
                prefix = "🟡"
            else:
                low = (base_text or "").lower()
                prefix = "🔴" if any(s in low for s in ("error", "fail", "no ckpt", "no ckpts", "not found", "missing")) else "🟢"
            text = f"{prefix} {base_text}"
            sb = self._output_dir_handles.get("status_banner")  # type: ignore[attr-defined]
            if sb is not None:
                sb.value = text
            for w in (
                self._output_dir_handles.get("date_dropdown"),
                self._output_dir_handles.get("multi_dropdown"),
                self._output_dir_handles.get("ckpt_dropdown"),
            ):
                if w is None:
                    continue
                try:
                    w.disabled = bool(loading)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    def _scan_ckpt_files(self, gsplat_dir: Path | None) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        if gsplat_dir is None:
            return out
        roots: List[Path] = []
        ck = Path(gsplat_dir) / "ckpts"
        if ck.exists() and ck.is_dir():
            roots.append(ck)
        roots.append(Path(gsplat_dir))  # fallback: sometimes ckpts are placed at root
        import re
        seen: set[str] = set()
        for root in roots:
            for ext in ("pt", "pth"):
                for p in root.glob(f"ckpt_*.{ext}"):
                    m = re.match(r"^ckpt_(\d+)(?:_rank\d+)?\.(?:pt|pth)$", p.name)
                    if not m:
                        continue
                    full = str(p.resolve())
                    if full in seen:
                        continue
                    seen.add(full)
                    out.append((full, int(m.group(1))))
        if not out:
            # Fallback: recursive search under gsplat dir (handles non-standard layouts)
            for ext in ("pt", "pth"):
                for p in Path(gsplat_dir).rglob(f"ckpt_*.{ext}"):
                    m = re.match(r"^ckpt_(\d+)(?:_rank\d+)?\.(?:pt|pth)$", p.name)
                    if not m:
                        continue
                    full = str(p.resolve())
                    if full in seen:
                        continue
                    seen.add(full)
                    out.append((full, int(m.group(1))))
        out.sort(key=lambda t: t[1])
        return out

    def _update_ckpt_dropdown(self, gsplat_dir: Path | None) -> None:
        dd = self._output_dir_handles.get("ckpt_dropdown")
        if dd is None:
            return
        # Ensure dropdown is enabled when updating choices
        try:
            dd.disabled = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            print(f"[viewer] scanning ckpts under: {gsplat_dir}")
        except Exception:
            pass
        items = self._scan_ckpt_files(gsplat_dir)
        try:
            print(f"[viewer] found {len(items)} ckpt(s)")
        except Exception:
            print("[viewer] _scan_ckpt_files failed to return any")
        # Update UI debug fields
        try:
            dtext = self._output_dir_handles.get("ckpt_dir_text")
            ctext = self._output_dir_handles.get("ckpt_count_text")
            ltext = self._output_dir_handles.get("ckpt_list_text")
            if dtext is not None:
                # Prefer gsplat_dir/ckpts if exists
                p = (Path(gsplat_dir) / "ckpts") if gsplat_dir is not None else None
                if p is not None:
                    dtext.value = str(p.resolve())
            if ctext is not None:
                ctext.value = str(len(items))
            if ltext is not None:
                try:
                    import os
                    ltext.value = ", ".join([os.path.basename(p) for p, _n in items])
                except Exception:
                    ltext.value = ""
        except Exception:
            pass
        if not items:
            try:
                dd.choices = tuple(["<none>"])  # type: ignore[attr-defined]
                dd.value = "<none>"  # type: ignore[attr-defined]
            except Exception:
                try:
                    dd.options = tuple(["<none>"])  # type: ignore[attr-defined]
                    dd.value = "<none>"  # type: ignore[attr-defined]
                except Exception:
                    print("[viewer] failed to set ckpt dropdown to <none>")
            return
        # Map labels (basenames) to full paths for stable, readable dropdown
        import os
        labels = [os.path.basename(p) for p, _n in items]
        self._ckpt_label_to_path = {lab: p for lab, (p, _n) in zip(labels, items)}
        label_choices = tuple(labels)
        try:
            dd.choices = label_choices  # type: ignore[attr-defined]
        except Exception:
            try:
                dd.options = label_choices  # type: ignore[attr-defined]
            except Exception:
                print("[viewer] failed to set ckpt dropdown options")
        last_label = labels[-1]
        try:
            dd.value = last_label  # type: ignore[attr-defined]
        except Exception:
            print("[viewer] failed to set ckpt dropdown value")
        # Re-ensure enabled after assigning value
        try:
            dd.disabled = False  # type: ignore[attr-defined]
        except Exception:
            print("[viewer] failed to enable ckpt dropdown")
        # Reflect number in numeric display
        try:
            nums = [n for _p, n in items]
            self.set_checkpoint_number(nums[-1] if nums else 0)
        except Exception:
            print("[viewer] failed to set checkpoint number")

    def _resolve_gsplat_dir(self, date_label: str | None, multi_label: str | None) -> Path | None:
        if date_label is None or multi_label is None:
            return None
        d = self._base_dir / date_label / multi_label / "gsplat_2dgs"
        try:
            return d.resolve()
        except Exception:
            return d

    def _scan_date_labels(self) -> List[str]:
        try:
            if not self._base_dir.exists():
                self._date_labels = []
                self._base_mtime = None
                return []
            try:
                base_mtime = self._base_dir.stat().st_mtime_ns
            except Exception:
                base_mtime = None
            if self._date_labels and self._base_mtime is not None and base_mtime == self._base_mtime:
                return list(self._date_labels)
            labels: List[str] = []
            for d in self._base_dir.iterdir():
                if not d.is_dir():
                    continue
                n = d.name
                if len(n) == 10 and n[4] == "-" and n[7] == "-" and n.replace("-", "").isdigit():
                    labels.append(n)
            labels.sort()
            if self._max_dates is not None and self._max_dates > 0:
                labels = labels[-self._max_dates:]
            self._date_labels = labels
            self._base_mtime = base_mtime
            return labels
        except Exception:
            return list(self._date_labels)

    def _scan_multis_for_date(self, date_label: str | None) -> List[str]:
        if date_label is None:
            return []
        d = self._base_dir / date_label
        if not d.exists() or not d.is_dir():
            self._date_to_multis.pop(date_label, None)
            self._date_mtimes.pop(date_label, None)
            return []
        try:
            mtime = d.stat().st_mtime_ns
        except Exception:
            mtime = 0
        if self._date_mtimes.get(date_label) == mtime and date_label in self._date_to_multis:
            return list(self._date_to_multis.get(date_label, []))
        multis: List[str] = []
        try:
            for p in d.glob("multisequence*/gsplat_2dgs"):
                if p.is_dir():
                    multis.append(p.parent.name)
        except Exception:
            pass
        multis.sort()
        if self._max_multis is not None and self._max_multis > 0:
            multis = multis[-self._max_multis:]
        self._date_to_multis[date_label] = multis
        self._date_mtimes[date_label] = mtime
        return multis

    def refresh_base_dir(self) -> None:
        try:
            prev_date = getattr(self._output_dir_handles.get("date_dropdown"), "value", None)
            prev_multi = getattr(self._output_dir_handles.get("multi_dropdown"), "value", None)
        except Exception:
            prev_date = None
            prev_multi = None

        date_labels = self._scan_date_labels()
        if not date_labels:
            return

        dd = self._output_dir_handles.get("date_dropdown")
        md = self._output_dir_handles.get("multi_dropdown")
        cur_path_text = self._output_dir_handles.get("cur_path_text")

        new_dates = tuple(date_labels)
        try:
            dd.choices = new_dates  # type: ignore[attr-defined]
        except Exception:
            try:
                dd.options = new_dates  # type: ignore[attr-defined]
            except Exception:
                pass

        cur_date = prev_date if prev_date in date_labels else date_labels[-1]
        try:
            dd.value = cur_date  # type: ignore[attr-defined]
        except Exception:
            pass

        multis = self._scan_multis_for_date(cur_date)
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

        new_dir = self._resolve_gsplat_dir(cur_date, cur_multi)
        if new_dir is None:
            return
        if Path(self.output_dir) != new_dir:
            self.output_dir = new_dir
            try:
                cur_path_text.value = str(self.output_dir)
            except Exception:
                pass
            # Update ckpt dropdown for the new dir
            self._update_ckpt_dropdown(self.output_dir)
            if self._on_select_dir is not None:
                try:
                    self._on_select_dir(self.output_dir)
                except Exception:
                    pass
            # Trigger load of selected ckpt if any
            try:
                sel_label = getattr(self._output_dir_handles.get("ckpt_dropdown"), "value", None)
            except Exception:
                sel_label = None
            if sel_label and sel_label != "<none>" and self._on_select_ckpt is not None:
                try:
                    path = self._ckpt_label_to_path.get(sel_label, sel_label)
                    self._on_select_ckpt(Path(path))
                except Exception:
                    pass
