import sys
import importlib.util
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    from pyqtgraph.parametertree import Parameter, ParameterTree
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing dependency: {missing}. Install pyqtgraph and a Qt binding, "
        "for example: python3 -m pip install pyqtgraph PyQt6"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NLSE_PACKAGE_DIR = PROJECT_ROOT / "NLSE"
sys.path.insert(0, str(PROJECT_ROOT))


def _load_coupled_rings_class():
    package_name = "NLSE"
    module_name = f"{package_name}.coupled_ring_simulation"
    module_path = NLSE_PACKAGE_DIR / "coupled_ring_simulation.py"

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(NLSE_PACKAGE_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CoupledRings


CoupledRings = _load_coupled_rings_class()


QT_SLOT = getattr(QtCore, "Slot", None) or getattr(QtCore, "pyqtSlot", None)
if QT_SLOT is None:
    def QT_SLOT(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

QT_VERTICAL = (
    QtCore.Qt.Orientation.Vertical
    if hasattr(QtCore.Qt, "Orientation")
    else QtCore.Qt.Vertical
)
QT_HORIZONTAL = (
    QtCore.Qt.Orientation.Horizontal
    if hasattr(QtCore.Qt, "Orientation")
    else QtCore.Qt.Horizontal
)
QT_APP_SHORTCUT = (
    QtCore.Qt.ShortcutContext.ApplicationShortcut
    if hasattr(QtCore.Qt, "ShortcutContext")
    else QtCore.Qt.ApplicationShortcut
)
C = 299792458
TRACES_DIR = Path(__file__).resolve().parent / "traces"
SPECTRUM_BAR_WIDTH_SCALE = 0.18
RESONANCE_DETUNING_POINTS = 10000
FIELD_LABELS = {
    "main": "A_main",
    "aux": "A_aux",
    "out": "A_out",
    "drop": "A_drop",
}
SPECTRUM_FIELDS = ("main", "aux", "out")
TIME_FIELDS = ("main", "aux", "out", "drop")
FIELD_COLORS = {
    "main": ("#4c78a8", "#2f4f6f"),
    "aux": ("#59a14f", "#2f6f39"),
    "out": ("#b07aa1", "#7a4f72"),
    "drop": ("#d95f02", "#8f3f00"),
}
TRACE_COLORS = {
    "main": "#f28e2b",
    "aux": "#e15759",
    "out": "#edc948",
}


DEFAULT_PARAMS = {
    "wl_center_nm": 1560.0,
    "FSR_main": 200e9,
    "FSR_aux": 206e9,
    "coupling_ring_bus": 0.01074,
    "coupling_ring_ring": 0.006816,
    "coupling_ring_drop": 0.0,
    "gamma": 1.229,
    "alpha_int_main": 8.643,
    "alpha_int_aux": 4.321,
    "beta_2": 40e-27,
    "intensity_in": 0.22,
    "detuning_main": 0.05340,
    "detuning_aux": 0.1891,
    "detuning_step": 1e-4,
    "nrt": 10000,
    "n_per_step": 5,
    "n_small_time": 512,
}


FLOAT_FIELDS = {
    "wl_center_nm",
    "FSR_main",
    "FSR_aux",
    "coupling_ring_bus",
    "coupling_ring_ring",
    "coupling_ring_drop",
    "gamma",
    "alpha_int_main",
    "alpha_int_aux",
    "beta_2",
    "intensity_in",
    "detuning_main",
    "detuning_aux",
    "detuning_step",
}


INT_FIELDS = {"nrt", "n_per_step", "n_small_time"}
SIMULATION_FIELDS = {
    "wl_center_nm",
    "FSR_main",
    "FSR_aux",
    "coupling_ring_bus",
    "coupling_ring_ring",
    "coupling_ring_drop",
    "gamma",
    "alpha_int_main",
    "alpha_int_aux",
    "beta_2",
    "n_small_time",
}

PARAM_GROUPS = {
    "Ring": [
        "wl_center_nm",
        "FSR_main",
        "FSR_aux",
        "coupling_ring_bus",
        "coupling_ring_ring",
        "coupling_ring_drop",
    ],
    "Propagation": ["gamma", "alpha_int_main", "alpha_int_aux", "beta_2"],
    "Drive": ["intensity_in", "detuning_main", "detuning_aux", "detuning_step"],
    "Numerics": ["nrt", "n_per_step", "n_small_time"],
}


def dBm(arr):
    return 10 * np.log10(np.maximum(arr, np.finfo(float).tiny)) + 30


class ParameterPanel(QtWidgets.QWidget):
    paramsApplied = QtCore.Signal(dict)
    paramsChanged = QtCore.Signal(dict)

    def __init__(self, params):
        super().__init__()
        self._updating = False
        self.setWindowTitle("Coupled Ring Parameters")
        self.setMinimumWidth(240)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.tree = ParameterTree(showHeader=False)
        self.params = self._make_tree(params)
        self.params.sigTreeStateChanged.connect(self._on_tree_changed)
        self.tree.setParameters(self.params, showTop=False)

        apply_button = QtWidgets.QPushButton("Apply and rebuild")
        reset_button = QtWidgets.QPushButton("Reset defaults")
        apply_button.clicked.connect(self.apply)
        reset_button.clicked.connect(self.reset_defaults)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(apply_button)
        buttons.addWidget(reset_button)
        buttons.addStretch(1)

        layout.addWidget(self.tree)
        layout.addLayout(buttons)
        self.resize(340, 560)

    def _make_tree(self, params):
        children = []
        for group_name, keys in PARAM_GROUPS.items():
            group_children = []
            for key in keys:
                group_children.append(
                    {
                        "name": key,
                        "type": "int" if key in INT_FIELDS else "float",
                        "value": params[key],
                        "step": self._step_for(key),
                    }
                )
            children.append({"name": group_name, "type": "group", "children": group_children})
        return Parameter.create(name="params", type="group", children=children)

    def _step_for(self, key):
        steps = {
            "wl_center_nm": 0.1,
            "FSR_main": 1e9,
            "FSR_aux": 1e9,
            "coupling_ring_bus": 1e-4,
            "coupling_ring_ring": 1e-4,
            "coupling_ring_drop": 1e-4,
            "gamma": 1e-3,
            "alpha_int_main": 1e-3,
            "alpha_int_aux": 1e-3,
            "beta_2": 1e-27,
            "intensity_in": 0.01,
            "detuning_main": 1e-4,
            "detuning_aux": 1e-4,
            "detuning_step": 1e-5,
            "nrt": 100,
            "n_per_step": 1,
            "n_small_time": 64,
        }
        return steps.get(key, 1)

    def _param_for(self, key):
        for group_name, keys in PARAM_GROUPS.items():
            if key in keys:
                return self.params.param(group_name, key)
        raise KeyError(key)

    def values(self):
        values = {}
        for key in DEFAULT_PARAMS:
            value = self._param_for(key).value()
            values[key] = int(value) if key in INT_FIELDS else float(value)
        return values

    def set_values(self, params):
        self._updating = True
        try:
            for key, value in params.items():
                if key in DEFAULT_PARAMS:
                    self._param_for(key).setValue(value)
        finally:
            self._updating = False

    def _on_tree_changed(self, *_args):
        if not self._updating:
            self.paramsChanged.emit(self.values())

    def apply(self):
        self.paramsApplied.emit(self.values())

    def reset_defaults(self):
        self.set_values(DEFAULT_PARAMS)
        self.paramsChanged.emit(self.values())


class SimulationWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, object, object)
    failed = QtCore.Signal(str)

    def __init__(self, simulation, run_sim, params, a_main_init, a_aux_init):
        super().__init__()
        self.simulation = simulation
        self.run_sim = run_sim
        self.params = params
        self.a_main_init = a_main_init
        self.a_aux_init = a_aux_init

    @QT_SLOT()
    def run(self):
        try:
            intensity = self.params["intensity_in"]
            a_in = np.sqrt(intensity) * np.ones(np.asarray(self.simulation.time).shape)
            result = self.run_sim(
                self.a_main_init,
                self.a_aux_init,
                a_in,
                detuning_main=self.params["detuning_main"],
                detuning_aux=self.params["detuning_aux"],
                nrt=self.params["nrt"],
                n_per_step=self.params["n_per_step"],
            )
            if len(result) == 2:
                a_main, a_aux = result
                a_out = np.zeros_like(a_main)
                a_drop = np.zeros_like(a_aux)
            else:
                a_main, a_aux, a_out, a_drop = result
            fields = {
                "main": np.asarray(a_main),
                "aux": np.asarray(a_aux),
                "out": np.asarray(a_out),
                "drop": np.asarray(a_drop),
            }
            time = np.asarray(self.simulation.time)
            wavelength_nm = 1e9 * np.asarray(self.simulation.wavelength)
            self.finished.emit(fields, time, wavelength_nm, self.params)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coupled Ring Explorer")
        self.params = dict(DEFAULT_PARAMS)
        self.simulation = None
        self.run_sim = None
        self.a_main = None
        self.a_aux = None
        self.a_out = None
        self.a_drop = None
        self._thread = None
        self._worker = None
        self.loaded_trace = None
        self.pending_trace_seed = None
        self.trace_overlays = {}

        self.param_panel = ParameterPanel(self.params)
        self.param_panel.paramsApplied.connect(self.apply_parameters)
        self.param_panel.paramsChanged.connect(self.sync_quick_controls)

        self._build_ui()
        self.rebuild_simulation(reset_fields=True)
        self.param_panel.show()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.spectrum_tabs = QtWidgets.QTabWidget()
        self.spectrum_plots = {}
        self.spectrum_bars = {}
        for field in SPECTRUM_FIELDS:
            plot = self._make_spectrum_plot(f"{FIELD_LABELS[field]} spectrum")
            self.spectrum_tabs.addTab(plot, FIELD_LABELS[field])
            self.spectrum_plots[field] = plot
            self.spectrum_bars[field] = None

        self.time_tabs = QtWidgets.QTabWidget()
        self.time_plots = {}
        self.time_curves = {}
        for field in TIME_FIELDS:
            plot = self._make_time_plot(field)
            brush, _ = FIELD_COLORS[field]
            curve = plot.plot(pen=pg.mkPen(brush, width=2))
            self.time_tabs.addTab(plot, FIELD_LABELS[field])
            self.time_plots[field] = plot
            self.time_curves[field] = curve

        main_plots = QtWidgets.QSplitter(QT_VERTICAL)
        main_plots.addWidget(self.spectrum_tabs)
        main_plots.addWidget(self.time_tabs)
        main_plots.setStretchFactor(0, 2)
        main_plots.setStretchFactor(1, 1)

        self.diagnostic_plots = self._build_diagnostic_plots()
        plot_area = QtWidgets.QSplitter(QT_HORIZONTAL)
        plot_area.addWidget(main_plots)
        plot_area.addWidget(self.diagnostic_plots)
        plot_area.setStretchFactor(0, 4)
        plot_area.setStretchFactor(1, 1)
        plot_area.setSizes([780, 280])

        controls_panel = QtWidgets.QVBoxLayout()
        controls_panel.setSpacing(4)
        inputs = QtWidgets.QHBoxLayout()
        inputs.setSpacing(6)
        action_controls = QtWidgets.QHBoxLayout()
        action_controls.setSpacing(6)
        self.main_detuning = QtWidgets.QDoubleSpinBox()
        self.main_detuning.setDecimals(8)
        self.main_detuning.setRange(-1000, 1000)
        self.main_detuning.setSingleStep(DEFAULT_PARAMS["detuning_step"])
        self.main_detuning.setMaximumWidth(115)
        self.aux_detuning = QtWidgets.QDoubleSpinBox()
        self.aux_detuning.setDecimals(8)
        self.aux_detuning.setRange(-1000, 1000)
        self.aux_detuning.setSingleStep(DEFAULT_PARAMS["detuning_step"])
        self.aux_detuning.setMaximumWidth(115)
        self.step_size = QtWidgets.QDoubleSpinBox()
        self.step_size.setDecimals(8)
        self.step_size.setRange(1e-9, 1000)
        self.step_size.setSingleStep(1e-4)
        self.step_size.setValue(DEFAULT_PARAMS["detuning_step"])
        self.step_size.setMaximumWidth(115)
        self.intensity_in = QtWidgets.QDoubleSpinBox()
        self.intensity_in.setDecimals(6)
        self.intensity_in.setRange(0, 1e6)
        self.intensity_in.setSingleStep(0.01)
        self.intensity_in.setValue(DEFAULT_PARAMS["intensity_in"])
        self.intensity_in.setMaximumWidth(100)
        self.nrt_control = QtWidgets.QSpinBox()
        self.nrt_control.setRange(1, 10_000_000)
        self.nrt_control.setSingleStep(100)
        self.nrt_control.setValue(DEFAULT_PARAMS["nrt"])
        self.nrt_control.setMaximumWidth(95)

        self.use_latest = QtWidgets.QCheckBox("use latest fields")
        self.use_latest.setChecked(True)

        for label, widget in (
            ("main", self.main_detuning),
            ("aux", self.aux_detuning),
            ("step", self.step_size),
            ("Iin", self.intensity_in),
            ("nrt", self.nrt_control),
        ):
            inputs.addWidget(QtWidgets.QLabel(label))
            inputs.addWidget(widget)
        inputs.addWidget(self.use_latest)
        inputs.addStretch(1)

        for text, slot in (
            ("- main", lambda: self.step_detuning("detuning_main", -1)),
            ("+ main", lambda: self.step_detuning("detuning_main", 1)),
            ("- aux", lambda: self.step_detuning("detuning_aux", -1)),
            ("+ aux", lambda: self.step_detuning("detuning_aux", 1)),
        ):
            button = QtWidgets.QPushButton(text)
            button.setMaximumWidth(68)
            button.clicked.connect(slot)
            action_controls.addWidget(button)

        self.run_button = QtWidgets.QPushButton("rerun")
        self.reset_button = QtWidgets.QPushButton("reset fields")
        self.params_button = QtWidgets.QPushButton("parameters")
        self.run_button.clicked.connect(self.run_simulation)
        self.reset_button.clicked.connect(self.reset_fields)
        self.params_button.clicked.connect(self.show_parameter_panel)
        action_controls.addWidget(self.run_button)
        action_controls.addWidget(self.reset_button)
        action_controls.addWidget(self.params_button)
        action_controls.addStretch(1)

        trace_controls = QtWidgets.QHBoxLayout()
        trace_controls.setSpacing(6)
        self.save_trace_button = QtWidgets.QPushButton("save trace")
        self.recall_trace_button = QtWidgets.QPushButton("recall trace")
        self.clear_trace_button = QtWidgets.QPushButton("clear trace")
        self.use_trace_seed_button = QtWidgets.QPushButton("use trace seed")
        self.load_trace_params = QtWidgets.QCheckBox("load trace parameters")
        self.clear_trace_button.setEnabled(False)
        self.use_trace_seed_button.setEnabled(False)
        self.save_trace_button.clicked.connect(self.save_trace)
        self.recall_trace_button.clicked.connect(self.recall_trace)
        self.clear_trace_button.clicked.connect(self.clear_trace_overlay)
        self.use_trace_seed_button.clicked.connect(self.arm_trace_seed)
        for button in (
            self.save_trace_button,
            self.recall_trace_button,
            self.clear_trace_button,
            self.use_trace_seed_button,
        ):
            button.setMaximumWidth(110)
            trace_controls.addWidget(button)
        trace_controls.addWidget(self.load_trace_params)
        trace_controls.addStretch(1)
        controls_panel.addLayout(inputs)
        controls_panel.addLayout(action_controls)
        controls_panel.addLayout(trace_controls)

        self.status = QtWidgets.QLabel("Ready")
        self.conversion_efficiency_label = QtWidgets.QLabel("Conversion efficiency: --")
        self.conversion_efficiency_label.setMinimumWidth(220)
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.status, stretch=1)
        status_row.addWidget(self.conversion_efficiency_label)
        layout.addLayout(controls_panel)
        layout.addWidget(plot_area)
        layout.addLayout(status_row)
        self.setCentralWidget(central)
        self._install_shortcuts()
        self.resize(980, 760)
        self.setMinimumSize(760, 520)

    def _build_diagnostic_plots(self):
        diagnostics = QtWidgets.QSplitter(QT_VERTICAL)
        diagnostics.setMinimumWidth(220)

        self.resonance_plot = pg.PlotWidget(title="Resonance transmission")
        self.resonance_plot.setLabel("bottom", "Detuning frequency (GHz)")
        self.resonance_plot.setLabel("left", "Transmission")
        self.resonance_plot.showGrid(x=True, y=True, alpha=0.2)
        self.resonance_plot.setMinimumSize(220, 220)
        self._disable_axis_si_prefix(self.resonance_plot, "bottom")
        self._disable_axis_si_prefix(self.resonance_plot, "left")
        self.resonance_plot.addLegend(offset=(8, 8))
        self.resonance_linear_curve = self.resonance_plot.plot(
            pen=pg.mkPen("#4c78a8", width=2),
            name="Linear",
        )
        self.resonance_nonlinear_curve = self.resonance_plot.plot(
            pen=pg.mkPen("#e15759", width=2),
            name="With conversion loss",
        )

        diagnostics.addWidget(self.resonance_plot)
        diagnostics.addWidget(QtWidgets.QWidget())
        diagnostics.setStretchFactor(0, 0)
        diagnostics.setStretchFactor(1, 1)
        diagnostics.setSizes([280, 420])
        return diagnostics

    def show_parameter_panel(self):
        self.param_panel.show()
        self.param_panel.raise_()
        self.param_panel.activateWindow()

    def _make_spectrum_plot(self, title):
        plot = pg.PlotWidget(title=title)
        plot.setLabel("bottom", "Wavelength (nm)")
        plot.setLabel("left", "Intensity (dBm)")
        self._disable_axis_si_prefix(plot, "bottom")
        self._disable_axis_si_prefix(plot, "left")
        plot.setXRange(1400, 1700)
        plot.setYRange(-10, 60)
        return plot

    def _make_time_plot(self, field):
        plot = pg.PlotWidget(title=f"{FIELD_LABELS[field]} time-domain intensity")
        plot.setLabel("bottom", "Time (ps)")
        plot.setLabel("left", "Intensity (W)")
        self._disable_axis_si_prefix(plot, "bottom")
        self._disable_axis_si_prefix(plot, "left")
        return plot

    def _disable_axis_si_prefix(self, plot, axis_name):
        axis = plot.getAxis(axis_name)
        if hasattr(axis, "enableAutoSIPrefix"):
            axis.enableAutoSIPrefix(False)

    def _install_shortcuts(self):
        shortcuts = (
            ("Up", lambda: self.step_detuning("detuning_main", 1)),
            ("Down", lambda: self.step_detuning("detuning_main", -1)),
            ("Right", lambda: self.step_detuning("detuning_aux", 1)),
            ("Left", lambda: self.step_detuning("detuning_aux", -1)),
            ("R", self.run_simulation),
            ("Ctrl+K", self.clear_trace_overlay),
        )
        for key, slot in shortcuts:
            if not hasattr(self, "_shortcuts"):
                self._shortcuts = []
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QT_APP_SHORTCUT)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def apply_parameters(self, params):
        self.sync_quick_controls(params)
        self.params.update(params)
        self.rebuild_simulation(reset_fields=True)
        self.run_simulation()

    def sync_quick_controls(self, params):
        self.main_detuning.setValue(params["detuning_main"])
        self.aux_detuning.setValue(params["detuning_aux"])
        self.step_size.setValue(params["detuning_step"])
        self.intensity_in.setValue(params["intensity_in"])
        self.nrt_control.setValue(params["nrt"])
        self.main_detuning.setSingleStep(params["detuning_step"])
        self.aux_detuning.setSingleStep(params["detuning_step"])

    def rebuild_simulation(self, reset_fields):
        wl_center = self.params["wl_center_nm"] * 1e-9
        omega_0 = 2 * np.pi * C / wl_center
        neff = C / (699e-6 * self.params["FSR_main"])

        self.simulation = CoupledRings(
            neff=neff,
            FSR_main=self.params["FSR_main"],
            FSR_aux=self.params["FSR_aux"],
            coupling_ring_bus=self.params["coupling_ring_bus"],
            coupling_ring_ring=self.params["coupling_ring_ring"],
            coupling_ring_drop=self.params["coupling_ring_drop"],
            gamma=self.params["gamma"],
            alpha_int_main=self.params["alpha_int_main"],
            alpha_int_aux=self.params["alpha_int_aux"],
            beta_2=self.params["beta_2"],
            omega_0=omega_0,
            n_small_time=self.params["n_small_time"],
        )
        self.run_sim = self.simulation.get_ikeda_map()
        self.main_detuning.setValue(self.params["detuning_main"])
        self.aux_detuning.setValue(self.params["detuning_aux"])
        self.step_size.setValue(self.params["detuning_step"])
        self.intensity_in.setValue(self.params["intensity_in"])
        self.nrt_control.setValue(self.params["nrt"])
        self.main_detuning.setSingleStep(self.params["detuning_step"])
        self.aux_detuning.setSingleStep(self.params["detuning_step"])
        if reset_fields:
            self.reset_fields()

    def reset_fields(self):
        shape = np.asarray(self.simulation.time).shape
        self.a_main = np.zeros((1, *shape), dtype=np.complex64)
        self.a_aux = np.zeros((1, *shape), dtype=np.complex64)
        self.a_out = np.zeros((1, *shape), dtype=np.complex64)
        self.a_drop = np.zeros((1, *shape), dtype=np.complex64)
        self.update_plots()
        self.update_resonance_transmission_plot(self.collect_ui_params())
        self.update_conversion_efficiency(self.params.get("intensity_in", self.intensity_in.value()))
        self.status.setText("Input fields reset to zero")

    def collect_ui_params(self):
        panel_params = self.param_panel.values()
        panel_params["detuning_main"] = self.main_detuning.value()
        panel_params["detuning_aux"] = self.aux_detuning.value()
        panel_params["detuning_step"] = self.step_size.value()
        panel_params["intensity_in"] = self.intensity_in.value()
        panel_params["nrt"] = self.nrt_control.value()
        return panel_params

    def current_run_params(self):
        panel_params = self.collect_ui_params()
        needs_rebuild = any(
            panel_params[key] != self.params[key]
            for key in SIMULATION_FIELDS
        )

        self.params.update(panel_params)
        self.param_panel.set_values(self.params)

        if needs_rebuild:
            self.rebuild_simulation(reset_fields=True)

        return dict(self.params)

    def _ensure_traces_dir(self):
        TRACES_DIR.mkdir(exist_ok=True)
        return TRACES_DIR

    def _spectrum_db(self, field):
        cross_section = np.abs(np.fft.fft(field) / field.size) ** 2
        return dBm(cross_section)

    def conversion_efficiency(self, intensity_in):
        if self.a_main is None or self.a_out is None or intensity_in <= 0:
            return np.nan
        n_samples = self.a_main[0].size
        out_spectrum_power = np.abs(np.fft.fft(self.a_out[-1]) / n_samples) ** 2
        return np.sum(out_spectrum_power[1:]) / intensity_in

    def update_conversion_efficiency(self, intensity_in):
        efficiency = self.conversion_efficiency(intensity_in)
        if np.isfinite(efficiency):
            self.conversion_efficiency_label.setText(
                f"Conversion efficiency: {efficiency:.4g} ({100 * efficiency:.3g}%)"
            )
        else:
            self.conversion_efficiency_label.setText("Conversion efficiency: --")

    def _trace_dataset(self):
        if any(arr is None for arr in self.current_fields().values()):
            raise RuntimeError("No fields are available to save yet.")

        params = self.collect_ui_params()
        self.params.update(params)
        self.param_panel.set_values(self.params)
        time = getattr(self, "time", np.asarray(self.simulation.time))
        wavelength_nm = getattr(self, "wavelength_nm", 1e9 * np.asarray(self.simulation.wavelength))
        fields = {name: arr[-1] for name, arr in self.current_fields().items()}

        ds = xr.Dataset(
            data_vars={},
            coords={
                "sample": np.arange(fields["main"].size),
                "time_s": ("sample", time),
                "wavelength_nm": ("sample", wavelength_nm),
            },
            attrs={
                "trace_schema": "coupled_ring_trace_v1",
                "description": "Final coupled-ring fields and spectra saved from coupled_ring_pyqtgraph_app.py",
            },
        )
        for field, arr in fields.items():
            label = FIELD_LABELS[field]
            ds[f"{label}_real"] = ("sample", np.real(arr))
            ds[f"{label}_imag"] = ("sample", np.imag(arr))
            ds[f"{field}_spectrum_dBm"] = ("sample", self._spectrum_db(arr))
        for key, value in params.items():
            ds.attrs[f"param_{key}"] = int(value) if key in INT_FIELDS else float(value)
        return ds

    def save_trace(self):
        try:
            ds = self._trace_dataset()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "No trace to save", str(exc))
            return

        traces_dir = self._ensure_traces_dir()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save trace",
            str(traces_dir / "trace.nc"),
            "NetCDF files (*.nc)",
        )
        if not path:
            return
        trace_path = Path(path)
        if trace_path.suffix.lower() != ".nc":
            trace_path = trace_path.with_suffix(".nc")
        ds.to_netcdf(trace_path, engine="scipy")
        self.status.setText(f"Saved trace: {trace_path.name}")

    def recall_trace(self):
        traces_dir = self._ensure_traces_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Recall trace",
            str(traces_dir),
            "Trace files (*.nc *.csv);;NetCDF files (*.nc);;CSV files (*.csv)",
        )
        if not path:
            return

        try:
            trace = self._load_trace(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load trace", str(exc))
            return

        self.loaded_trace = trace
        self.pending_trace_seed = None
        if self.load_trace_params.isChecked():
            self.apply_trace_parameters(trace)
        self._show_trace_overlay(trace)
        self.clear_trace_button.setEnabled(True)
        self.use_trace_seed_button.setEnabled(True)
        self.status.setText(f"Loaded trace overlay: {trace['path'].name}")

    def apply_trace_parameters(self, trace):
        trace_params = trace.get("params", {})
        if not trace_params:
            QtWidgets.QMessageBox.information(
                self,
                "No trace parameters",
                "This trace does not contain saved simulation parameters.",
            )
            return

        updated_params = dict(self.params)
        updated_params.update(trace_params)
        needs_rebuild = any(
            updated_params[key] != self.params[key]
            for key in SIMULATION_FIELDS
        )
        self.params.update(updated_params)
        self.param_panel.set_values(self.params)
        self.sync_quick_controls(self.params)
        if needs_rebuild:
            self.rebuild_simulation(reset_fields=True)

    def _load_trace(self, path):
        if path.suffix.lower() == ".nc":
            return self._load_netcdf_trace(path)
        return self._load_csv_trace(path)

    def _load_csv_trace(self, path):
        df = pd.read_csv(path)
        required = {
            "wavelength_nm",
            "A_main_real",
            "A_main_imag",
            "A_aux_real",
            "A_aux_imag",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Trace is missing columns: {', '.join(missing)}")

        main = df["A_main_real"].to_numpy() + 1j * df["A_main_imag"].to_numpy()
        aux = df["A_aux_real"].to_numpy() + 1j * df["A_aux_imag"].to_numpy()
        out = self._complex_from_dataframe(df, "A_out")
        drop = self._complex_from_dataframe(df, "A_drop")
        main_spectrum = (
            df["main_spectrum_dBm"].to_numpy()
            if "main_spectrum_dBm" in df
            else
            df["main_spectrum_db"].to_numpy()
            if "main_spectrum_db" in df
            else self._spectrum_db(main)
        )
        aux_spectrum = (
            df["aux_spectrum_dBm"].to_numpy()
            if "aux_spectrum_dBm" in df
            else
            df["aux_spectrum_db"].to_numpy()
            if "aux_spectrum_db" in df
            else self._spectrum_db(aux)
        )
        out_spectrum = (
            df["out_spectrum_dBm"].to_numpy()
            if "out_spectrum_dBm" in df
            else
            df["out_spectrum_db"].to_numpy()
            if "out_spectrum_db" in df
            else self._spectrum_db(out) if out is not None else None
        )
        return {
            "path": path,
            "wavelength_nm": df["wavelength_nm"].to_numpy(),
            "a_main": main.astype(np.complex64),
            "a_aux": aux.astype(np.complex64),
            "a_out": None if out is None else out.astype(np.complex64),
            "a_drop": None if drop is None else drop.astype(np.complex64),
            "main_spectrum_db": main_spectrum,
            "aux_spectrum_db": aux_spectrum,
            "out_spectrum_db": out_spectrum,
        }

    def _complex_from_dataframe(self, df, label):
        real_col = f"{label}_real"
        imag_col = f"{label}_imag"
        if real_col not in df or imag_col not in df:
            return None
        return df[real_col].to_numpy() + 1j * df[imag_col].to_numpy()

    def _load_netcdf_trace(self, path):
        required = {
            "A_main_real",
            "A_main_imag",
            "A_aux_real",
            "A_aux_imag",
        }
        with xr.open_dataset(path) as ds:
            missing = sorted(required - set(ds.variables))
            if missing:
                raise ValueError(f"Trace is missing variables: {', '.join(missing)}")
            if "wavelength_nm" not in ds.coords and "wavelength_nm" not in ds.variables:
                raise ValueError("Trace is missing wavelength_nm coordinate.")
            trace_params = self._params_from_attrs(ds.attrs)

            main = ds["A_main_real"].values + 1j * ds["A_main_imag"].values
            aux = ds["A_aux_real"].values + 1j * ds["A_aux_imag"].values
            out = self._complex_from_dataset(ds, "A_out")
            drop = self._complex_from_dataset(ds, "A_drop")
            wavelength_nm = ds["wavelength_nm"].values
            main_spectrum = (
                ds["main_spectrum_dBm"].values
                if "main_spectrum_dBm" in ds.variables
                else ds["main_spectrum_db"].values
                if "main_spectrum_db" in ds.variables
                else self._spectrum_db(main)
            )
            aux_spectrum = (
                ds["aux_spectrum_dBm"].values
                if "aux_spectrum_dBm" in ds.variables
                else ds["aux_spectrum_db"].values
                if "aux_spectrum_db" in ds.variables
                else self._spectrum_db(aux)
            )
            out_spectrum = (
                ds["out_spectrum_dBm"].values
                if "out_spectrum_dBm" in ds.variables
                else ds["out_spectrum_db"].values
                if "out_spectrum_db" in ds.variables
                else self._spectrum_db(out) if out is not None else None
            )

        return {
            "path": path,
            "wavelength_nm": wavelength_nm,
            "a_main": main.astype(np.complex64),
            "a_aux": aux.astype(np.complex64),
            "a_out": None if out is None else out.astype(np.complex64),
            "a_drop": None if drop is None else drop.astype(np.complex64),
            "main_spectrum_db": main_spectrum,
            "aux_spectrum_db": aux_spectrum,
            "out_spectrum_db": out_spectrum,
            "params": trace_params,
        }

    def _params_from_attrs(self, attrs):
        params = {}
        for key in DEFAULT_PARAMS:
            attr = f"param_{key}"
            if attr in attrs:
                params[key] = self._coerce_param_value(key, attrs[attr])
        return params

    def _coerce_param_value(self, key, value):
        if isinstance(value, np.ndarray):
            value = value.item()
        return int(value) if key in INT_FIELDS else float(value)

    def _complex_from_dataset(self, ds, label):
        real_var = f"{label}_real"
        imag_var = f"{label}_imag"
        if real_var not in ds.variables or imag_var not in ds.variables:
            return None
        return ds[real_var].values + 1j * ds[imag_var].values

    def _show_trace_overlay(self, trace):
        for field in SPECTRUM_FIELDS:
            spectrum = trace.get(f"{field}_spectrum_db")
            if spectrum is None:
                continue
            self.trace_overlays[field] = self._set_trace_overlay(
                self.spectrum_plots[field],
                self.trace_overlays.get(field),
                trace["wavelength_nm"],
                spectrum,
                TRACE_COLORS[field],
            )

    def _set_trace_overlay(self, plot, existing_overlay, wavelength_nm, spectrum_db, color):
        order = np.argsort(wavelength_nm)
        if existing_overlay is None:
            existing_overlay = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
            existing_overlay.setZValue(20)
            plot.addItem(existing_overlay)
        existing_overlay.setData(wavelength_nm[order], spectrum_db[order])
        return existing_overlay

    def clear_trace_overlay(self):
        for field, overlay in list(self.trace_overlays.items()):
            if overlay is not None:
                self.spectrum_plots[field].removeItem(overlay)
        self.loaded_trace = None
        self.pending_trace_seed = None
        self.trace_overlays = {}
        self.clear_trace_button.setEnabled(False)
        self.use_trace_seed_button.setEnabled(False)
        self.status.setText("Cleared trace overlay")

    def arm_trace_seed(self):
        if self.loaded_trace is None:
            QtWidgets.QMessageBox.warning(self, "No trace loaded", "Recall a trace before using it as a seed.")
            return

        shape = np.asarray(self.simulation.time).shape
        if self.loaded_trace["a_main"].shape != shape or self.loaded_trace["a_aux"].shape != shape:
            QtWidgets.QMessageBox.warning(
                self,
                "Trace shape mismatch",
                "This trace cannot seed the current simulation because its field length "
                f"is {self.loaded_trace['a_main'].size}, but the current simulation length is {shape[0]}.",
            )
            return

        self.pending_trace_seed = (
            self.loaded_trace["a_main"].copy(),
            self.loaded_trace["a_aux"].copy(),
        )
        self.status.setText(f"Trace seed armed for next run: {self.loaded_trace['path'].name}")

    def step_detuning(self, key, direction):
        step = self.step_size.value()
        if key == "detuning_main":
            self.main_detuning.setValue(self.main_detuning.value() + direction * step)
        else:
            self.aux_detuning.setValue(self.aux_detuning.value() + direction * step)
        self.run_simulation()

    def run_simulation(self):
        if self._thread is not None:
            self.status.setText("Simulation already running")
            return

        params = self.current_run_params()
        shape = np.asarray(self.simulation.time).shape
        seed_source = "latest fields"
        if self.pending_trace_seed is not None:
            if self.pending_trace_seed[0].shape != shape or self.pending_trace_seed[1].shape != shape:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Trace shape mismatch",
                    "The armed trace seed no longer matches the current simulation size. "
                    "Recall or arm a compatible trace before running.",
                )
                self.pending_trace_seed = None
                return
            a_main_init, a_aux_init = self.pending_trace_seed
            self.pending_trace_seed = None
            seed_source = "trace seed"
        elif self.use_latest.isChecked() and self.a_main is not None and self.a_aux is not None:
            a_main_init = self.a_main[-1].copy()
            a_aux_init = self.a_aux[-1].copy()
        else:
            a_main_init = np.zeros(shape, dtype=np.complex64)
            a_aux_init = np.zeros(shape, dtype=np.complex64)
            seed_source = "zero fields"

        self.run_button.setEnabled(False)
        self.status.setText(f"Running from {seed_source}...")
        self._thread = QtCore.QThread()
        self._worker = SimulationWorker(self.simulation, self.run_sim, params, a_main_init, a_aux_init)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.simulation_finished)
        self._worker.failed.connect(self.simulation_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self.thread_finished)
        self._thread.start()

    @QT_SLOT(object, object, object, object)
    def simulation_finished(self, fields, time, wavelength_nm, params):
        self.a_main = fields["main"]
        self.a_aux = fields["aux"]
        self.a_out = fields["out"]
        self.a_drop = fields["drop"]
        self.time = time
        self.wavelength_nm = wavelength_nm
        self.update_plots()
        self.update_resonance_transmission_plot(params)
        self.update_conversion_efficiency(params["intensity_in"])
        self.status.setText(
            "Complete: "
            f"main={params['detuning_main']:.6g}, "
            f"aux={params['detuning_aux']:.6g}, "
            f"intensity={params['intensity_in']:.6g}"
        )

    @QT_SLOT(str)
    def simulation_failed(self, message):
        QtWidgets.QMessageBox.critical(self, "Simulation failed", message)
        self.status.setText("Simulation failed")

    def thread_finished(self):
        self._thread = None
        self._worker = None
        self.run_button.setEnabled(True)

    def update_resonance_transmission_plot(self, params=None):
        if self.simulation is None:
            return

        params = self.collect_ui_params() if params is None else params
        detuning_main = float(params["detuning_main"])
        detuning_aux = float(params["detuning_aux"])
        detuning = np.linspace(
            min((detuning_main, detuning_aux)) - 0.2,
            max((detuning_main, detuning_aux)) + 0.15,
            RESONANCE_DETUNING_POINTS,
        )
        detuning_frequency_ghz = self.simulation.FSR_main * detuning / (2 * np.pi) / 1e9

        try:
            linear = self.simulation.resonance_transmission_linear(
                detuning,
                detuning_main,
                detuning_aux,
            )
            self._set_curve_data(self.resonance_linear_curve, detuning_frequency_ghz, linear)
        except Exception:
            self.resonance_linear_curve.setData([], [])

        if self.a_main is None or self.a_aux is None:
            self.resonance_nonlinear_curve.setData([], [])
            return

        a_main = np.asarray(self.a_main[-1])
        a_aux = np.asarray(self.a_aux[-1])
        if not self._has_finite_dc_component(a_main) or not self._has_finite_dc_component(a_aux):
            self.resonance_nonlinear_curve.setData([], [])
            return

        try:
            nonlinear = self.simulation.resonance_transmission_nonlinear(
                detuning,
                detuning_main,
                detuning_aux,
                a_main,
                a_aux,
            )
            self._set_curve_data(self.resonance_nonlinear_curve, detuning_frequency_ghz, nonlinear)
        except Exception:
            self.resonance_nonlinear_curve.setData([], [])

    def _has_finite_dc_component(self, field):
        dc_component = np.fft.fft(field)[0]
        return np.isfinite(dc_component.real) and np.isfinite(dc_component.imag) and abs(dc_component) > 0

    def _set_curve_data(self, curve, x, y):
        x = np.asarray(x, dtype=float)
        y = self._as_real_array(y)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            curve.setData(x[finite], y[finite])
        else:
            curve.setData([], [])

    def _as_real_array(self, values):
        values = np.asarray(values)
        if np.iscomplexobj(values):
            values = np.abs(values) ** 2
        else:
            values = np.real(values)
        return np.asarray(values, dtype=float)

    def update_plots(self):
        if self.a_main is None:
            return

        time = getattr(self, "time", np.asarray(self.simulation.time))
        time_ps = 1e12 * np.asarray(time)
        wavelength_nm = getattr(self, "wavelength_nm", 1e9 * np.asarray(self.simulation.wavelength))
        fields = self.current_fields()

        for field in SPECTRUM_FIELDS:
            brush, pen = FIELD_COLORS[field]
            self.spectrum_bars[field] = self._set_spectrum_bars(
                self.spectrum_plots[field],
                self.spectrum_bars[field],
                wavelength_nm,
                fields[field][-1],
                brush=brush,
                pen=pen,
            )

        for field in TIME_FIELDS:
            self.time_curves[field].setData(time_ps, np.abs(fields[field][-1]) ** 2)

    def current_fields(self):
        return {
            "main": self.a_main,
            "aux": self.a_aux,
            "out": self.a_out,
            "drop": self.a_drop,
        }

    def _set_spectrum_bars(self, plot, existing_bars, wavelength_nm, field, *, brush, pen):
        spectrum_db = self._spectrum_db(field)
        order = np.argsort(wavelength_nm)
        sorted_wavelength = wavelength_nm[order]
        sorted_bar_height = spectrum_db[order] + 100
        bar_width = (
            SPECTRUM_BAR_WIDTH_SCALE * abs(np.mean(np.diff(np.fft.fftshift(wavelength_nm))))
            if sorted_wavelength.size > 1
            else 0.5
        )

        if existing_bars is not None:
            plot.removeItem(existing_bars)
        bars = pg.BarGraphItem(
            x=sorted_wavelength,
            height=sorted_bar_height,
            width=bar_width,
            y0=-100,
            brush=pg.mkBrush(brush),
            pen=pg.mkPen(pen, width=0.5),
        )
        plot.addItem(bars)
        return bars

    def closeEvent(self, event):
        self.param_panel.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    exec_app = getattr(app, "exec", None) or app.exec_
    sys.exit(exec_app())


if __name__ == "__main__":
    main()
