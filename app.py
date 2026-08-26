"""AI-Based Smart Production Scheduling and Real-Time Rescheduling System.
Self-contained Streamlit application with explicit form dropdown selectors for production orders and machines.
"""

from dataclasses import dataclass
from typing import Any, Literal
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="AI Smart Production Scheduler & Rescheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. DATA MODELS
# ==========================================


@dataclass
class ProductionOrder:
    job_id: str
    part: str
    machine_id: str
    quantity: int
    material: str
    operation: str
    processing_time: float
    setup_time: float
    priority: int
    due_date: float


@dataclass
class MachineInfo:
    machine_id: str
    machine_type: str
    status: Literal["AVAILABLE", "DOWN"]
    current_job: str = "None"
    speed_factor: float = 1.0


@dataclass
class HistoricalDataRecord:
    part: str
    material: str
    operation: str
    cutting_speed: float
    feed_rate: float
    depth_of_cut: float
    tool_condition: float
    actual_processing_time: float


@dataclass
class ScheduledJob:
    job_id: str
    part: str
    machine_id: str
    quantity: int
    material: str
    operation: str
    setup_time: float
    base_processing_time: float
    predicted_processing_time: float
    start_time: float
    finish_time: float
    due_date: float
    priority: int
    is_late: bool
    lateness: float
    changeover_time: float


@dataclass
class ScheduleResult:
    scheduled_jobs: list[ScheduledJob]
    makespan: float
    total_processing_time: float
    total_changeover_time: float
    total_idle_time: float
    machine_utilization: dict[str, float]
    machine_workload: dict[str, float]
    late_jobs_count: int
    bottleneck_machine: str
    optimization_objective: str


@dataclass
class ReschedulingComparison:
    before_result: ScheduleResult
    after_result: ScheduleResult
    downed_machines: list[str]
    rescheduled_at_time: float
    impact_summary: str


# ==========================================
# 2. SAMPLE DATA
# ==========================================


def get_sample_production_orders() -> list[ProductionOrder]:
    return [
        ProductionOrder("JOB-101", "Gear Shaft A", "CNC-01", 50, "Steel 4140", "Turning & Milling", 2.5, 0.5, 4, 24.0),
        ProductionOrder("JOB-102", "Housing Cover B", "CNC-02", 30, "Aluminum 6061", "Boring", 1.8, 0.4, 5, 16.0),
        ProductionOrder("JOB-103", "Drive Axle C", "CNC-01", 40, "Steel 4340", "Lathe Cut", 3.0, 0.6, 3, 36.0),
        ProductionOrder("JOB-104", "Flange Plate D", "CNC-03", 60, "Stainless Steel", "Surface Grinding", 1.2, 0.3, 2, 30.0),
        ProductionOrder("JOB-105", "Bracket E", "CNC-02", 25, "Aluminum 7075", "Milling", 2.0, 0.5, 4, 20.0),
        ProductionOrder("JOB-106", "Spindle Shaft F", "CNC-03", 35, "Steel 4140", "Turning", 2.7, 0.4, 3, 40.0),
    ]


def get_sample_machines() -> list[MachineInfo]:
    return [
        MachineInfo("CNC-01", "5-Axis CNC Lathe", "AVAILABLE", "Idle", 1.0),
        MachineInfo("CNC-02", "Vertical Machining Center", "AVAILABLE", "Idle", 1.0),
        MachineInfo("CNC-03", "Horizontal Mill", "AVAILABLE", "Idle", 1.0),
    ]


def get_sample_historical_data() -> list[HistoricalDataRecord]:
    return [
        HistoricalDataRecord("Gear Shaft A", "Steel 4140", "Turning & Milling", 250.0, 0.25, 2.0, 0.85, 2.45),
        HistoricalDataRecord("Housing Cover B", "Aluminum 6061", "Boring", 400.0, 0.35, 1.5, 0.90, 1.75),
        HistoricalDataRecord("Drive Axle C", "Steel 4340", "Lathe Cut", 220.0, 0.20, 2.5, 0.75, 3.10),
        HistoricalDataRecord("Flange Plate D", "Stainless Steel", "Surface Grinding", 180.0, 0.15, 1.0, 0.95, 1.20),
        HistoricalDataRecord("Bracket E", "Aluminum 7075", "Milling", 380.0, 0.30, 1.8, 0.80, 1.95),
        HistoricalDataRecord("Spindle Shaft F", "Steel 4140", "Turning", 240.0, 0.22, 2.2, 0.88, 2.65),
        HistoricalDataRecord("Gear Shaft A", "Steel 4140", "Turning & Milling", 260.0, 0.26, 2.0, 0.60, 2.70),
        HistoricalDataRecord("Housing Cover B", "Aluminum 6061", "Boring", 410.0, 0.36, 1.5, 0.50, 1.90),
    ]


# ==========================================
# 3. MACHINE LEARNING PREDICTOR
# ==========================================


class ProcessingTimePredictor:
    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None
        self.part_encoder = LabelEncoder()
        self.material_encoder = LabelEncoder()
        self.operation_encoder = LabelEncoder()
        self.feature_names: list[str] = [
            "part_encoded",
            "material_encoded",
            "operation_encoded",
            "cutting_speed",
            "feed_rate",
            "depth_of_cut",
            "tool_condition",
        ]
        self.mae: float = 0.0
        self.r2: float = 0.0
        self.feature_importances: dict[str, float] = {}
        self.is_trained: bool = False

    def train(self, records: list[HistoricalDataRecord]) -> dict[str, Any]:
        if not records or len(records) < 2:
            self.is_trained = False
            return {"status": "insufficient_data", "mae": 0.0, "r2": 0.0}

        df = pd.DataFrame(
            [
                {
                    "part": r.part,
                    "material": r.material,
                    "operation": r.operation,
                    "cutting_speed": r.cutting_speed,
                    "feed_rate": r.feed_rate,
                    "depth_of_cut": r.depth_of_cut,
                    "tool_condition": r.tool_condition,
                    "target": r.actual_processing_time,
                }
                for r in records
            ]
        )

        df["part_encoded"] = self.part_encoder.fit_transform(df["part"])
        df["material_encoded"] = self.material_encoder.fit_transform(df["material"])
        df["operation_encoded"] = self.operation_encoder.fit_transform(df["operation"])

        X = df[self.feature_names]
        y = df["target"]

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        self.mae = float(mean_absolute_error(y, y_pred))
        self.r2 = float(r2_score(y, y_pred))

        importances = self.model.feature_importances_
        self.feature_importances = {
            name: float(imp) for name, imp in zip(self.feature_names, importances, strict=True)
        }
        self.is_trained = True

        return {
            "status": "success",
            "mae": self.mae,
            "r2": self.r2,
            "feature_importances": self.feature_importances,
        }

    def predict_processing_time(
        self,
        part: str,
        material: str,
        operation: str,
        fallback_time: float,
        cutting_speed: float = 250.0,
        feed_rate: float = 0.25,
        depth_of_cut: float = 2.0,
        tool_condition: float = 0.9,
    ) -> float:
        if not self.is_trained or self.model is None:
            return fallback_time

        try:
            part_val = (
                self.part_encoder.transform([part])[0]
                if part in self.part_encoder.classes_
                else 0
            )
            mat_val = (
                self.material_encoder.transform([material])[0]
                if material in self.material_encoder.classes_
                else 0
            )
            op_val = (
                self.operation_encoder.transform([operation])[0]
                if operation in self.operation_encoder.classes_
                else 0
            )

            features = pd.DataFrame(
                [
                    [
                        part_val,
                        mat_val,
                        op_val,
                        cutting_speed,
                        feed_rate,
                        depth_of_cut,
                        tool_condition,
                    ]
                ],
                columns=self.feature_names,
            )

            pred = float(self.model.predict(features)[0])
            return max(0.1, pred)
        except Exception:
            return fallback_time


# ==========================================
# 4. OPTIMIZATION & SCHEDULING ENGINE
# ==========================================

OptimizationObjective = Literal[
    "Minimum production time",
    "Minimum machine idle time",
    "Minimum late jobs",
    "Minimum changeover time",
    "Balanced optimization",
]


class ProductionOptimizer:
    def __init__(
        self,
        orders: list[ProductionOrder],
        machines: list[MachineInfo],
        predictor: ProcessingTimePredictor | None = None,
    ) -> None:
        self.orders = orders
        self.machines = machines
        self.predictor = predictor

    def schedule(
        self,
        objective: OptimizationObjective = "Balanced optimization",
        excluded_machines: list[str] | None = None,
    ) -> ScheduleResult:
        excluded = set(excluded_machines or [])
        active_machines = [
            m for m in self.machines if m.status == "AVAILABLE" and m.machine_id not in excluded
        ]

        if not active_machines:
            active_machines = [
                MachineInfo(machine_id="EMERGENCY-01", machine_type="Backup", status="AVAILABLE")
            ]

        sorted_orders = self._sort_orders_by_objective(self.orders, objective)

        machine_ready_time: dict[str, float] = {m.machine_id: 0.0 for m in active_machines}
        machine_last_part: dict[str, str] = {m.machine_id: "" for m in active_machines}

        scheduled_jobs: list[ScheduledJob] = []

        for order in sorted_orders:
            target_machine = order.machine_id
            if target_machine not in machine_ready_time:
                target_machine = min(machine_ready_time, key=lambda m: machine_ready_time[m])

            base_time = order.processing_time * (order.quantity / max(1, order.quantity))
            pred_time = base_time
            if self.predictor and self.predictor.is_trained:
                pred_time = self.predictor.predict_processing_time(
                    part=order.part,
                    material=order.material,
                    operation=order.operation,
                    fallback_time=base_time,
                )

            machine_info = next((m for m in self.machines if m.machine_id == target_machine), None)
            speed_factor = machine_info.speed_factor if machine_info else 1.0
            adjusted_processing_time = pred_time / max(0.1, speed_factor)

            last_part = machine_last_part.get(target_machine, "")
            changeover = (
                order.setup_time
                if (last_part and last_part != order.part)
                else (0.1 if last_part == "" else 0.0)
            )
            if objective == "Minimum changeover time" and last_part == order.part:
                changeover = 0.0

            start_time = machine_ready_time.get(target_machine, 0.0) + changeover
            finish_time = start_time + adjusted_processing_time

            machine_ready_time[target_machine] = finish_time
            machine_last_part[target_machine] = order.part

            is_late = finish_time > order.due_date
            lateness = max(0.0, finish_time - order.due_date)

            scheduled_jobs.append(
                ScheduledJob(
                    job_id=order.job_id,
                    part=order.part,
                    machine_id=target_machine,
                    quantity=order.quantity,
                    material=order.material,
                    operation=order.operation,
                    setup_time=order.setup_time,
                    base_processing_time=order.processing_time,
                    predicted_processing_time=round(adjusted_processing_time, 2),
                    start_time=round(start_time, 2),
                    finish_time=round(finish_time, 2),
                    due_date=order.due_date,
                    priority=order.priority,
                    is_late=is_late,
                    lateness=round(lateness, 2),
                    changeover_time=round(changeover, 2),
                )
            )

        makespan = max((j.finish_time for j in scheduled_jobs), default=0.0)
        total_processing = sum(j.predicted_processing_time for j in scheduled_jobs)
        total_changeover = sum(j.changeover_time for j in scheduled_jobs)
        total_available_time = makespan * len(active_machines) if active_machines else 1.0
        total_idle_time = max(0.0, total_available_time - (total_processing + total_changeover))

        machine_workload: dict[str, float] = {m.machine_id: 0.0 for m in active_machines}
        for j in scheduled_jobs:
            machine_workload[j.machine_id] = (
                machine_workload.get(j.machine_id, 0.0)
                + j.predicted_processing_time
                + j.changeover_time
            )

        machine_utilization: dict[str, float] = {}
        for m_id, workload in machine_workload.items():
            util = (workload / makespan * 100.0) if makespan > 0 else 0.0
            machine_utilization[m_id] = min(100.0, round(util, 2))

        late_count = sum(1 for j in scheduled_jobs if j.is_late)
        bottleneck = max(machine_workload, key=machine_workload.get) if machine_workload else "None"

        return ScheduleResult(
            scheduled_jobs=scheduled_jobs,
            makespan=round(makespan, 2),
            total_processing_time=round(total_processing, 2),
            total_changeover_time=round(total_changeover, 2),
            total_idle_time=round(total_idle_time, 2),
            machine_utilization=machine_utilization,
            machine_workload=machine_workload,
            late_jobs_count=late_count,
            bottleneck_machine=bottleneck,
            optimization_objective=objective,
        )

    def _sort_orders_by_objective(
        self, orders: list[ProductionOrder], objective: OptimizationObjective
    ) -> list[ProductionOrder]:
        if objective == "Minimum late jobs":
            return sorted(orders, key=lambda o: (o.due_date, -o.priority))
        elif objective == "Minimum production time":
            return sorted(orders, key=lambda o: (o.processing_time, -o.priority))
        elif objective == "Minimum changeover time":
            return sorted(orders, key=lambda o: (o.part, o.material))
        elif objective == "Minimum machine idle time":
            return sorted(orders, key=lambda o: (-o.processing_time, o.due_date))
        else:
            return sorted(orders, key=lambda o: (-o.priority, o.due_date))


# ==========================================
# 5. RESCHEDULING SIMULATOR
# ==========================================


class ReschedulingSimulator:
    def __init__(self, optimizer: ProductionOptimizer) -> None:
        self.optimizer = optimizer

    def simulate_failure(
        self,
        downed_machines: list[str],
        reschedule_time: float = 0.0,
    ) -> ReschedulingComparison:
        before_result = self.optimizer.schedule(objective="Balanced optimization")
        after_result = self.optimizer.schedule(
            objective=before_result.optimization_objective,
            excluded_machines=downed_machines,
        )

        makespan_diff = after_result.makespan - before_result.makespan
        late_diff = after_result.late_jobs_count - before_result.late_jobs_count
        impact = (
            f"Machine(s) {', '.join(downed_machines)} marked DOWN at t={reschedule_time}h. "
            f"Makespan changed by {makespan_diff:+.2f}h (Before: {before_result.makespan}h -> After: {after_result.makespan}h). "
            f"Late jobs changed by {late_diff:+d} (Before: {before_result.late_jobs_count} -> After: {after_result.late_jobs_count}). "
            f"New bottleneck machine: {after_result.bottleneck_machine}."
        )

        return ReschedulingComparison(
            before_result=before_result,
            after_result=after_result,
            downed_machines=downed_machines,
            rescheduled_at_time=reschedule_time,
            impact_summary=impact,
        )


# ==========================================
# 6. DATA EXPORTER
# ==========================================


class DataExporter:
    @staticmethod
    def export_orders_csv(orders: list[ProductionOrder]) -> str:
        df = pd.DataFrame(
            [
                {
                    "Job ID": o.job_id,
                    "Part": o.part,
                    "Machine": o.machine_id,
                    "Quantity": o.quantity,
                    "Material": o.material,
                    "Operation": o.operation,
                    "Processing Time": o.processing_time,
                    "Setup Time": o.setup_time,
                    "Priority": o.priority,
                    "Due Date": o.due_date,
                }
                for o in orders
            ]
        )
        return df.to_csv(index=False)

    @staticmethod
    def export_machines_csv(machines: list[MachineInfo]) -> str:
        df = pd.DataFrame(
            [
                {
                    "Machine ID": m.machine_id,
                    "Machine Type": m.machine_type,
                    "Status": m.status,
                    "Current Job": m.current_job,
                    "Speed Factor": m.speed_factor,
                }
                for m in machines
            ]
        )
        return df.to_csv(index=False)

    @staticmethod
    def export_schedule_csv(result: ScheduleResult) -> str:
        df = pd.DataFrame(
            [
                {
                    "Job ID": j.job_id,
                    "Part": j.part,
                    "Machine": j.machine_id,
                    "Quantity": j.quantity,
                    "Material": j.material,
                    "Operation": j.operation,
                    "Setup Time": j.setup_time,
                    "Processing Time (Pred)": j.predicted_processing_time,
                    "Start Time": j.start_time,
                    "Finish Time": j.finish_time,
                    "Due Date": j.due_date,
                    "Priority": j.priority,
                    "Is Late": j.is_late,
                    "Lateness (h)": j.lateness,
                    "Changeover Time": j.changeover_time,
                }
                for j in result.scheduled_jobs
            ]
        )
        return df.to_csv(index=False)

    @staticmethod
    def export_kpi_comparison_csv(comparison: ReschedulingComparison) -> str:
        b = comparison.before_result
        a = comparison.after_result
        df = pd.DataFrame(
            [
                {
                    "Metric": "Makespan (hours)",
                    "Before Failure": b.makespan,
                    "After Failure": a.makespan,
                    "Difference": a.makespan - b.makespan,
                },
                {
                    "Metric": "Total Processing Time (hours)",
                    "Before Failure": b.total_processing_time,
                    "After Failure": a.total_processing_time,
                    "Difference": a.total_processing_time - b.total_processing_time,
                },
                {
                    "Metric": "Total Changeover Time (hours)",
                    "Before Failure": b.total_changeover_time,
                    "After Failure": a.total_changeover_time,
                    "Difference": a.total_changeover_time - b.total_changeover_time,
                },
                {
                    "Metric": "Total Idle Time (hours)",
                    "Before Failure": b.total_idle_time,
                    "After Failure": a.total_idle_time,
                    "Difference": a.total_idle_time - b.total_idle_time,
                },
                {
                    "Metric": "Late Jobs Count",
                    "Before Failure": b.late_jobs_count,
                    "After Failure": a.late_jobs_count,
                    "Difference": a.late_jobs_count - b.late_jobs_count,
                },
                {
                    "Metric": "Bottleneck Machine",
                    "Before Failure": b.bottleneck_machine,
                    "After Failure": a.bottleneck_machine,
                    "Difference": "N/A",
                },
            ]
        )
        return df.to_csv(index=False)


# ==========================================
# 7. STREAMLIT APP UI & DASHBOARDS
# ==========================================


def initialize_session_state() -> None:
    if "orders" not in st.session_state:
        st.session_state.orders = get_sample_production_orders()
    if "machines" not in st.session_state:
        st.session_state.machines = get_sample_machines()
    if "historical" not in st.session_state:
        st.session_state.historical = get_sample_historical_data()
    if "predictor" not in st.session_state:
        st.session_state.predictor = ProcessingTimePredictor()
    if "trained_ml" not in st.session_state:
        st.session_state.trained_ml = False


initialize_session_state()

# Sidebar Navigation
st.sidebar.title("🏭 Smart Scheduler")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Dashboard",
        "📂 Data Inputs & Uploads",
        "🤖 ML Processing Time Predictor",
        "⚡ Optimization & Scheduling Engine",
        "📈 Interactive Visualizations",
        "🚨 Real-Time Failure & Rescheduling",
        "📥 Reports & CSV Exports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**: Online\n\n"
    "**Engine**: Python / Scikit-Learn / Plotly / Streamlit\n"
    "**Version**: 5.13.11"
)

# 1. Executive Dashboard
if page == "📊 Executive Dashboard":
    st.markdown(
        '<div class="main-header">AI-Based Smart Production Scheduling & Real-Time Rescheduling</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Real-time manufacturing optimization, machine learning estimation, and failure recovery.</div>',
        unsafe_allow_html=True,
    )

    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Makespan", f"{result.makespan} hrs")
    with col2:
        st.metric("Total Processing", f"{result.total_processing_time} hrs")
    with col3:
        st.metric("Total Changeover", f"{result.total_changeover_time} hrs")
    with col4:
        st.metric(
            "Late Jobs",
            f"{result.late_jobs_count}",
            delta="Warning" if result.late_jobs_count > 0 else "Optimal",
            delta_color="inverse" if result.late_jobs_count > 0 else "normal",
        )
    with col5:
        st.metric("Bottleneck Machine", f"{result.bottleneck_machine}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("⚡ Machine Utilization Overview")
        util_df = pd.DataFrame(
            list(result.machine_utilization.items()),
            columns=["Machine ID", "Utilization (%)"],
        )
        fig_util = px.bar(
            util_df,
            x="Machine ID",
            y="Utilization (%)",
            color="Utilization (%)",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_util.update_layout(
            yaxis_range=[0, 100], margin={"l": 20, "r": 20, "t": 30, "b": 20}
        )
        st.plotly_chart(fig_util)

    with col_b:
        st.subheader("💡 AI Production Recommendations")
        recommendations = []
        if result.late_jobs_count > 0:
            recommendations.append(
                f"⚠️ **Due Date Risk**: {result.late_jobs_count} job(s) exceed due dates. Consider prioritizing 'Minimum late jobs' objective."
            )
        if result.bottleneck_machine != "None":
            max_load = result.machine_workload.get(result.bottleneck_machine, 0)
            recommendations.append(
                f"🔧 **Bottleneck Identified**: Machine `{result.bottleneck_machine}` carries the highest workload ({max_load} hrs)."
            )
        if not st.session_state.trained_ml:
            recommendations.append(
                "🤖 **Machine Learning**: Historical data is not trained. Train the RandomForestRegressor in the ML tab."
            )
        if not recommendations:
            recommendations.append(
                "✅ All systems operating within optimal parameters. Schedule is balanced and on track."
            )

        for rec in recommendations:
            st.markdown(f"- {rec}")

    st.markdown("---")
    st.subheader("📋 Scheduled Production Summary")
    summary_df = pd.DataFrame(
        [
            {
                "Job ID": j.job_id,
                "Part": j.part,
                "Machine": j.machine_id,
                "Qty": j.quantity,
                "Start (h)": j.start_time,
                "Finish (h)": j.finish_time,
                "Due (h)": j.due_date,
                "Late?": "YES" if j.is_late else "NO",
            }
            for j in result.scheduled_jobs
        ]
    )
    st.dataframe(summary_df)

# 2. Data Inputs & Uploads
elif page == "📂 Data Inputs & Uploads":
    st.markdown(
        '<div class="main-header">Production Orders & Machine Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Manage production orders and machine status using interactive dropdown selectors.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Production Orders", "Machine Information", "Historical Manufacturing Data"]
    )

    available_machines = [m.machine_id for m in st.session_state.machines]
    available_parts = ["Gear Shaft A", "Housing Cover B", "Drive Axle C", "Flange Plate D", "Bracket E", "Spindle Shaft F"]
    available_materials = ["Steel 4140", "Aluminum 6061", "Steel 4340", "Stainless Steel", "Aluminum 7075"]
    available_operations = ["Turning & Milling", "Boring", "Lathe Cut", "Surface Grinding", "Milling", "Turning"]

    with tab1:
        st.subheader("Manage Production Orders (with Dropdown Selectors)")

        with st.form("add_order_form"):
            st.markdown("### Add / Update Production Order")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                f_job_id = st.text_input("Job ID", value=f"JOB-{100 + len(st.session_state.orders) + 1}")
                f_part = st.selectbox("Part", options=available_parts)
                f_machine = st.selectbox("Assigned Machine", options=available_machines)
                f_quantity = st.number_input("Quantity", min_value=1, value=40, step=1)
            with col_f2:
                f_material = st.selectbox("Material", options=available_materials)
                f_operation = st.selectbox("Operation", options=available_operations)
                f_proc_time = st.number_input("Base Processing Time (hrs)", min_value=0.1, value=2.0, step=0.1)
                f_setup_time = st.number_input("Setup Time (hrs)", min_value=0.0, value=0.5, step=0.1)
            with col_f3:
                f_priority = st.slider("Priority (1=Low, 5=High)", min_value=1, max_value=5, value=3)
                f_due_date = st.number_input("Due Date (hrs)", min_value=1.0, value=24.0, step=1.0)
                st.markdown("")
                submitted_order = st.form_submit_button("➕ Add Order to Schedule", type="primary")

            if submitted_order:
                existing_idx = next((i for i, o in enumerate(st.session_state.orders) if o.job_id == f_job_id), None)
                new_ord = ProductionOrder(
                    job_id=f_job_id,
                    part=f_part,
                    machine_id=f_machine,
                    quantity=int(f_quantity),
                    material=f_material,
                    operation=f_operation,
                    processing_time=float(f_proc_time),
                    setup_time=float(f_setup_time),
                    priority=int(f_priority),
                    due_date=float(f_due_date),
                )
                if existing_idx is not None:
                    st.session_state.orders[existing_idx] = new_ord
                    st.success(f"Updated order {f_job_id} successfully!")
                else:
                    st.session_state.orders.append(new_ord)
                    st.success(f"Added new order {f_job_id} successfully!")
                st.rerun()

        st.markdown("---")
        st.subheader("Current Production Orders List")

        for idx, order in enumerate(st.session_state.orders):
            cols = st.columns([2, 2, 2, 1, 1, 1])
            cols[0].write(f"**{order.job_id}**: {order.part}")
            cols[1].write(f"Machine: `{order.machine_id}`")
            cols[2].write(f"Qty: {order.quantity} | Due: {order.due_date}h")
            cols[3].write(f"Priority: {order.priority}")
            cols[4].write(f"Proc: {order.processing_time}h")
            if cols[5].button("🗑️ Delete", key=f"del_ord_{idx}"):
                st.session_state.orders.pop(idx)
                st.rerun()

        if st.button("Reset Orders to Default"):
            st.session_state.orders = get_sample_production_orders()
            st.success("Orders reset to sample data!")
            st.rerun()

    with tab2:
        st.subheader("Manage Machines & Status (with Dropdowns)")

        with st.form("add_machine_form"):
            st.markdown("### Add / Update Machine")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m_id = st.text_input("Machine ID", value=f"CNC-0{len(st.session_state.machines) + 1}")
                m_type = st.text_input("Machine Type", value="5-Axis CNC")
                m_status = st.selectbox("Status", options=["AVAILABLE", "DOWN"])
            with col_m2:
                m_job = st.text_input("Current Job", value="Idle")
                m_speed = st.number_input("Speed Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
                st.markdown("")
                submitted_machine = st.form_submit_button("➕ Add / Update Machine", type="primary")

            if submitted_machine:
                existing_m_idx = next((i for i, m in enumerate(st.session_state.machines) if m.machine_id == m_id), None)
                new_mach = MachineInfo(
                    machine_id=m_id,
                    machine_type=m_type,
                    status=m_status,
                    current_job=m_job,
                    speed_factor=float(m_speed),
                )
                if existing_m_idx is not None:
                    st.session_state.machines[existing_m_idx] = new_mach
                    st.success(f"Updated machine {m_id} successfully!")
                else:
                    st.session_state.machines.append(new_mach)
                    st.success(f"Added machine {m_id} successfully!")
                st.rerun()

        st.markdown("---")
        st.subheader("Current Machines List")
        for idx, mach in enumerate(st.session_state.machines):
            cols = st.columns([2, 2, 2, 1])
            cols[0].write(f"**{mach.machine_id}** ({mach.machine_type})")
            status_color = "🔴" if mach.status == "DOWN" else "🟢"
            cols[1].write(f"Status: {status_color} **{mach.status}**")
            cols[2].write(f"Speed Factor: {mach.speed_factor}x")
            if cols[3].button("🗑️ Delete", key=f"del_mach_{idx}"):
                st.session_state.machines.pop(idx)
                st.rerun()

        if st.button("Reset Machines to Default"):
            st.session_state.machines = get_sample_machines()
            st.success("Machines reset to sample data!")
            st.rerun()

    with tab3:
        st.subheader("Historical Manufacturing Data (ML Training)")

        with st.form("add_hist_form"):
            st.markdown("### Add Historical Record")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                h_part = st.selectbox("Part Name", options=available_parts, key="h_part")
                h_mat = st.selectbox("Material", options=available_materials, key="h_mat")
                h_op = st.selectbox("Operation", options=available_operations, key="h_op")
                h_speed = st.number_input("Cutting Speed", value=250.0)
            with col_h2:
                h_feed = st.number_input("Feed Rate", value=0.25)
                h_depth = st.number_input("Depth of Cut", value=2.0)
                h_tool = st.number_input("Tool Condition", min_value=0.1, max_value=1.0, value=0.9)
                h_actual = st.number_input("Actual Processing Time (hrs)", value=2.2)
                st.markdown("")
                submitted_hist = st.form_submit_button("➕ Add Historical Record", type="primary")

            if submitted_hist:
                st.session_state.historical.append(
                    HistoricalDataRecord(
                        part=h_part,
                        material=h_mat,
                        operation=h_op,
                        cutting_speed=float(h_speed),
                        feed_rate=float(h_feed),
                        depth_of_cut=float(h_depth),
                        tool_condition=float(h_tool),
                        actual_processing_time=float(h_actual),
                    )
                )
                st.success("Historical record added successfully!")
                st.rerun()

        st.markdown("---")
        st.subheader(f"Historical Records ({len(st.session_state.historical)} total)")
        for idx, rec in enumerate(st.session_state.historical[-10:]):
            cols = st.columns([3, 2, 2, 1])
            cols[0].write(f"**{rec.part}** ({rec.material})")
            cols[1].write(f"Op: {rec.operation}")
            cols[2].write(f"Actual Time: {rec.actual_processing_time}h")
            if cols[3].button("🗑️", key=f"del_hist_{idx}"):
                st.session_state.historical.pop(idx)
                st.rerun()

        if st.button("Reset Historical Data to Default"):
            st.session_state.historical = get_sample_historical_data()
            st.success("Historical data reset to sample data!")
            st.rerun()

# 3. ML Processing Time Predictor
elif page == "🤖 ML Processing Time Predictor":
    st.markdown('<div class="main-header">Machine Learning Processing Time Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train a RandomForestRegressor using historical manufacturing parameters.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Training")
        st.write(f"Available historical records: **{len(st.session_state.historical)}**")
        if st.button("Train RandomForest Model", type="primary"):
            with st.spinner("Training Random Forest Regressor..."):
                predictor = ProcessingTimePredictor()
                res = predictor.train(st.session_state.historical)
                st.session_state.predictor = predictor
                st.session_state.trained_ml = True
                if res["status"] == "success":
                    st.success("Model trained successfully!")
                else:
                    st.warning("Insufficient historical data for robust training.")

        if st.session_state.trained_ml:
            st.metric("Mean Absolute Error (MAE)", f"{st.session_state.predictor.mae:.4f} hrs")
            st.metric("R² Score", f"{st.session_state.predictor.r2:.4f}")
        else:
            st.info("Model not trained yet.")

    with col2:
        st.subheader("Feature Importance Analysis")
        if st.session_state.trained_ml and st.session_state.predictor.feature_importances:
            imp_dict = st.session_state.predictor.feature_importances
            imp_df = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Random Forest Feature Importances", color="Importance", color_continuous_scale="Viridis")
            st.plotly_chart(fig_imp)
        else:
            st.info("Train the model to visualize feature importances.")

# 4. Optimization & Scheduling Engine
elif page == "⚡ Optimization & Scheduling Engine":
    st.markdown('<div class="main-header">Optimization & Scheduling Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select your strategic manufacturing objective and generate the optimized production schedule.</div>', unsafe_allow_html=True)

    objective = st.selectbox(
        "Select Optimization Objective",
        [
            "Balanced optimization",
            "Minimum production time",
            "Minimum machine idle time",
            "Minimum late jobs",
            "Minimum changeover time",
        ],
    )

    if st.button("Run Optimizer", type="primary"):
        with st.spinner("Running scheduling optimization..."):
            optimizer = ProductionOptimizer(
                st.session_state.orders,
                st.session_state.machines,
                st.session_state.predictor if st.session_state.trained_ml else None,
            )
            result = optimizer.schedule(objective=objective)
            st.session_state.last_schedule_result = result
            st.success("Optimization completed successfully!")

    if "last_schedule_result" in st.session_state:
        res = st.session_state.last_schedule_result
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Makespan", f"{res.makespan} hrs")
        with col2:
            st.metric("Total Changeover", f"{res.total_changeover_time} hrs")
        with col3:
            st.metric("Late Jobs", f"{res.late_jobs_count}")
        with col4:
            st.metric("Bottleneck", f"{res.bottleneck_machine}")

        st.subheader("Optimized Job Schedule")
        sched_df = pd.DataFrame(
            [
                {
                    "Job ID": j.job_id,
                    "Part": j.part,
                    "Machine": j.machine_id,
                    "Qty": j.quantity,
                    "Setup (h)": j.setup_time,
                    "Proc (h)": j.predicted_processing_time,
                    "Start (h)": j.start_time,
                    "Finish (h)": j.finish_time,
                    "Due (h)": j.due_date,
                    "Late?": "YES" if j.is_late else "NO",
                    "Lateness (h)": j.lateness,
                }
                for j in res.scheduled_jobs
            ]
        )
        st.dataframe(sched_df)

# 5. Interactive Visualizations
elif page == "📈 Interactive Visualizations":
    st.markdown('<div class="main-header">Production Visualizations & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Gantt charts, workload distribution, and machine utilization analytics.</div>', unsafe_allow_html=True)

    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    if not result.scheduled_jobs:
        st.warning("No scheduled jobs found. Please check your production orders.")
    else:
        st.subheader("Production Gantt Chart (Machine Timelines)")
        gantt_data = [
            {
                "Task": f"{j.job_id} ({j.part})",
                "Machine": j.machine_id,
                "Start": j.start_time,
                "Finish": j.finish_time,
                "Part": j.part,
            }
            for j in result.scheduled_jobs
        ]
        df_gantt = pd.DataFrame(gantt_data)
        fig_gantt = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="Finish",
            y="Machine",
            color="Part",
            title="Machine-wise Job Timelines (Gantt)",
            labels={"Start": "Start Time (hours)", "Finish": "Finish Time (hours)"},
        )
        fig_gantt.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_gantt)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Machine Workload Distribution")
            workload_df = pd.DataFrame(list(result.machine_workload.items()), columns=["Machine ID", "Workload (hrs)"])
            fig_workload = px.bar(workload_df, x="Machine ID", y="Workload (hrs)", color="Workload (hrs)", color_continuous_scale="Blues", text_auto=True)
            st.plotly_chart(fig_workload)

        with col2:
            st.subheader("Machine Utilization (%)")
            util_df = pd.DataFrame(list(result.machine_utilization.items()), columns=["Machine ID", "Utilization (%)"])
            fig_util = px.bar(util_df, x="Machine ID", y="Utilization (%)", color="Utilization (%)", color_continuous_scale="Greens", text_auto=True)
            fig_util.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_util)

# 6. Real-Time Failure & Rescheduling
elif page == "🚨 Real-Time Failure & Rescheduling":
    st.markdown('<div class="main-header">Real-Time Machine Failure Simulation & Rescheduling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate unexpected machine breakdowns and view before-and-after KPI comparisons.</div>', unsafe_allow_html=True)

    machine_ids = [m.machine_id for m in st.session_state.machines]
    downed_selection = st.multiselect("Select Machine(s) to Simulate DOWN", machine_ids)
    reschedule_time = st.slider("Failure Time (hours from start)", 0.0, 24.0, 2.0, 0.5)

    if st.button("Trigger Failure & Reschedule", type="primary"):
        optimizer = ProductionOptimizer(
            st.session_state.orders,
            st.session_state.machines,
            st.session_state.predictor if st.session_state.trained_ml else None,
        )
        simulator = ReschedulingSimulator(optimizer)
        comparison = simulator.simulate_failure(downed_machines=downed_selection, reschedule_time=reschedule_time)
        st.session_state.last_reschedule_comparison = comparison
        st.success("Rescheduling simulation completed!")

    if "last_reschedule_comparison" in st.session_state:
        comp = st.session_state.last_reschedule_comparison
        st.markdown("---")
        st.info(comp.impact_summary)

        st.subheader("Before & After KPI Comparison")
        b = comp.before_result
        a = comp.after_result

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Makespan", f"{a.makespan} hrs", delta=f"{a.makespan - b.makespan:+.2f} hrs")
        with col2:
            st.metric("Total Changeover", f"{a.total_changeover_time} hrs", delta=f"{a.total_changeover_time - b.total_changeover_time:+.2f} hrs")
        with col3:
            st.metric("Late Jobs", f"{a.late_jobs_count}", delta=f"{a.late_jobs_count - b.late_jobs_count:+d}")

        col_b, col_a_sched = st.columns(2)
        with col_b:
            st.markdown("**Baseline Schedule**")
            base_df = pd.DataFrame([{"Job": j.job_id, "Machine": j.machine_id, "Finish": j.finish_time, "Late": j.is_late} for j in b.scheduled_jobs])
            st.dataframe(base_df)

        with col_a_sched:
            st.markdown("**Rescheduled After Failure**")
            after_df = pd.DataFrame([{"Job": j.job_id, "Machine": j.machine_id, "Finish": j.finish_time, "Late": j.is_late} for j in a.scheduled_jobs])
            st.dataframe(after_df)

        st.markdown("---")
        st.subheader("⏱️ Job-by-Job Delay & Variance Analysis (Hours)")

        base_dict = {j.job_id: j for j in b.scheduled_jobs}
        after_dict = {j.job_id: j for j in a.scheduled_jobs}

        delay_rows = []
        for job_id, b_job in base_dict.items():
            a_job = after_dict.get(job_id)
            if a_job:
                finish_diff = round(a_job.finish_time - b_job.finish_time, 2)
                lateness_diff = round(a_job.lateness - b_job.lateness, 2)
                status = "On Time"
                if finish_diff > 0:
                    status = f"Delayed by +{finish_diff}h"
                elif finish_diff < 0:
                    status = f"Early by {finish_diff}h"

                if a_job.is_late and not b_job.is_late:
                    status = f"⚠️ Newly Late (+{lateness_diff}h)"
                elif a_job.is_late and b_job.is_late:
                    status = f"🔴 Still Late (+{lateness_diff}h)"

                delay_rows.append({
                    "Job ID": job_id,
                    "Part": b_job.part,
                    "Baseline Machine": b_job.machine_id,
                    "Rescheduled Machine": a_job.machine_id,
                    "Baseline Finish (h)": b_job.finish_time,
                    "Rescheduled Finish (h)": a_job.finish_time,
                    "Delay / Slip (hrs)": finish_diff,
                    "Status": status
                })

        delay_df = pd.DataFrame(delay_rows)
        st.dataframe(delay_df)

# 7. Reports & CSV Exports
elif page == "📥 Reports & CSV Exports":
    st.markdown('<div class="main-header">Reports & Data Exports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Download production inputs, optimized schedules, and KPI comparison reports as CSV files.</div>', unsafe_allow_html=True)

    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. User Production Input")
        orders_csv = DataExporter.export_orders_csv(st.session_state.orders)
        st.download_button(
            label="Download User Production Input.csv",
            data=orders_csv,
            file_name="User Production Input.csv",
            mime="text/csv",
        )

        st.subheader("2. Machine Status")
        machines_csv = DataExporter.export_machines_csv(st.session_state.machines)
        st.download_button(
            label="Download Machine Status.csv",
            data=machines_csv,
            file_name="Machine Status.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("3. Optimized Production Schedule")
        sched_csv = DataExporter.export_schedule_csv(result)
        st.download_button(
            label="Download Optimized Production Schedule.csv",
            data=sched_csv,
            file_name="Optimized Production Schedule.csv",
            mime="text/csv",
        )

        if "last_reschedule_comparison" in st.session_state:
            st.subheader("4. KPI Before & After Failure")
            kpi_csv = DataExporter.export_kpi_comparison_csv(st.session_state.last_reschedule_comparison)
            st.download_button(
                label="Download KPI Before After Failure.csv",
                data=kpi_csv,
                file_name="KPI Before After Failure.csv",
                mime="text/csv",
            )
        else:
            st.info("Run a failure simulation in the previous tab to enable KPI Before After Failure export.")
