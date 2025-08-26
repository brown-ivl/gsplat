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
        self._date_to_multis: Dict[str, List[str]] = {}
        self._date_mtimes: Dict[str, int] = {}
        self._base_mtime: int | None = None
        self._date_labels: List[str] = []
        self._max_dates: int | None = max_dates
        self._max_multis: int | None = max_multis

        # Initial scan
        date_labels = self._scan_date_labels()
        if not date_labels:
            date_labels = ["unknown"]
            self._date_to_multis["unknown"] = ["multisequence0"]

        # Choose defaults
        cur_date = default_date if (default_date in date_labels) else date_labels[-1]
        multis_for_date = self._scan_multis_for_date(cur_date)
        if not multis_for_date:
            multis_for_date = ["multisequence0"]
        cur_multi = (
            default_multiseq if (default_multiseq in multis_for_date) else multis_for_date[-1]
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
                "Checkpoint Number",
                initial_value=0,
                disabled=True,
                hint="Highest ckpt_<number>.pt loaded from the directory.",
            )
            ckpt_dropdown = server.gui.add_dropdown(
                "Checkpoint",
                tuple(["<none>"]),
                initial_value="<none>",
                hint="Select a checkpoint (ckpts/ckpt_*.pt) to load.",
            )

            @date_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                new_date = date_dropdown.value
                new_multis = tuple(self._scan_multis_for_date(new_date))
                try:
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
                    # Update checkpoints for new dir
                    self._update_ckpt_dropdown(self.output_dir)
                    # Callbacks
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass
                    sel_ck = getattr(ckpt_dropdown, "value", None)
                    if sel_ck and sel_ck != "<none>" and self._on_select_ckpt is not None:
                        try:
                            self._on_select_ckpt(Path(sel_ck))
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
                    # Update checkpoints for new dir
                    self._update_ckpt_dropdown(self.output_dir)
                    if self._on_select_dir is not None:
                        try:
                            self._on_select_dir(self.output_dir)
                        except Exception:
                            pass
                    sel_ck = getattr(ckpt_dropdown, "value", None)
                    if sel_ck and sel_ck != "<none>" and self._on_select_ckpt is not None:
                        try:
                            self._on_select_ckpt(Path(sel_ck))
                        except Exception:
                            pass

            @ckpt_dropdown.on_update
            def _(_evt) -> None:  # noqa: ANN001
                val = getattr(ckpt_dropdown, "value", None)
                if not val or val == "<none>":
                    return
                if self._on_select_ckpt is not None:
                    try:
                        self._on_select_ckpt(Path(val))
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
            "cur_path_text": cur_path_text,
            "ckpt_number": ckpt_number,
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
        ck = Path(gsplat_dir) / "ckpts"
        if not ck.exists() or not ck.is_dir():
            return out
        import re
        for p in ck.glob("ckpt_*.pt"):
            m = re.match(r"^ckpt_(\d+)(?:_rank\d+)?\.pt$", p.name)
            if not m:
                continue
            out.append((str(p.resolve()), int(m.group(1))))
        out.sort(key=lambda t: t[1])
        return out

    def _update_ckpt_dropdown(self, gsplat_dir: Path | None) -> None:
        dd = self._output_dir_handles.get("ckpt_dropdown")
        if dd is None:
            return
        items = self._scan_ckpt_files(gsplat_dir)
        if not items:
            try:
                dd.choices = tuple(["<none>"])  # type: ignore[attr-defined]
                dd.value = "<none>"  # type: ignore[attr-defined]
            except Exception:
                try:
                    dd.options = tuple(["<none>"])  # type: ignore[attr-defined]
                    dd.value = "<none>"  # type: ignore[attr-defined]
                except Exception:
                    pass
            return
        choices = tuple([p for p, _n in items])
        try:
            dd.choices = choices  # type: ignore[attr-defined]
        except Exception:
            try:
                dd.options = choices  # type: ignore[attr-defined]
            except Exception:
                pass
        last = choices[-1]
        try:
            dd.value = last  # type: ignore[attr-defined]
        except Exception:
            pass
        # Reflect number in numeric display
        try:
            nums = [n for _p, n in items]
            self.set_checkpoint_number(nums[-1] if nums else 0)
        except Exception:
            pass

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
                sel_ck = getattr(self._output_dir_handles.get("ckpt_dropdown"), "value", None)
            except Exception:
                sel_ck = None
            if sel_ck and sel_ck != "<none>" and self._on_select_ckpt is not None:
                try:
                    self._on_select_ckpt(Path(sel_ck))
                except Exception:
                    pass
