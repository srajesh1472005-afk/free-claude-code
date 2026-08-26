"""AI-Based Smart Production Scheduling and Real-Time Rescheduling System.
Self-contained Streamlit application for Streamlit Community Cloud and local execution.
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
        st.plotly_chart(fig_util, use_container_width=True)

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
    st.dataframe(summary_df, width=None)

# 2. Data Inputs & Uploads
elif page == "📂 Data Inputs & Uploads":
    st.markdown(
        '<div class="main-header">Production Orders & Machine Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Manage production orders, machine status, and historical data via CSV upload or manual entry.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Production Orders", "Machine Information", "Historical Manufacturing Data"]
    )

    with tab1:
        st.subheader("Production Orders")
        col_up, col_btn = st.columns([3, 1])
        with col_up:
            order_file = st.file_uploader(
                "Upload Production Orders CSV", type=["csv"], key="order_csv"
            )
        with col_btn:
            if st.button("Load Sample Orders"):
                st.session_state.orders = get_sample_production_orders()
                st.success("Sample orders loaded!")

        if order_file is not None:
            try:
                df_in = pd.read_csv(order_file)
                new_orders = []
                for _, row in df_in.iterrows():
                    new_orders.append(
                        ProductionOrder(
                            job_id=str(row.get("Job ID", f"JOB-{_}")),
                            part=str(row.get("Part", "Part A")),
                            machine_id=str(row.get("Machine", "CNC-01")),
                            quantity=int(row.get("Quantity", 10)),
                            material=str(row.get("Material", "Steel")),
                            operation=str(row.get("Operation", "Milling")),
                            processing_time=float(row.get("Processing Time", 2.0)),
                            setup_time=float(row.get("Setup Time", 0.5)),
                            priority=int(row.get("Priority", 3)),
                            due_date=float(row.get("Due Date", 24.0)),
                        )
                    )
                st.session_state.orders = new_orders
                st.success(f"Successfully loaded {len(new_orders)} production orders from CSV.")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

        orders_df = pd.DataFrame(
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
                for o in st.session_state.orders
            ]
        )
        edited_orders_df = st.data_editor(orders_df, num_rows="dynamic", width=None)
        if st.button("Save Orders Edits"):
            updated_orders = []
            for _, r in edited_orders_df.iterrows():
                updated_orders.append(
                    ProductionOrder(
                        job_id=str(r["Job ID"]),
                        part=str(r["Part"]),
                        machine_id=str(r["Machine"]),
                        quantity=int(r["Quantity"]),
                        material=str(r["Material"]),
                        operation=str(r["Operation"]),
                        processing_time=float(r["Processing Time"]),
                        setup_time=float(r["Setup Time"]),
                        priority=int(r["Priority"]),
                        due_date=float(r["Due Date"]),
                    )
                )
            st.session_state.orders = updated_orders
            st.success("Production orders updated successfully!")

    with tab2:
        st.subheader("Machine Information & Status")
        col_up_m, col_btn_m = st.columns([3, 1])
        with col_up_m:
            machine_file = st.file_uploader("Upload Machine Info CSV", type=["csv"], key="machine_csv")
        with col_btn_m:
            if st.button("Load Sample Machines"):
                st.session_state.machines = get_sample_machines()
                st.success("Sample machines loaded!")

        if machine_file is not None:
            try:
                df_m = pd.read_csv(machine_file)
                new_machines = []
                for _, row in df_m.iterrows():
                    new_machines.append(
                        MachineInfo(
                            machine_id=str(row.get("Machine ID", "CNC-01")),
                            machine_type=str(row.get("Machine Type", "CNC")),
                            status=str(row.get("Status", "AVAILABLE")),
                            current_job=str(row.get("Current Job", "Idle")),
                            speed_factor=float(row.get("Speed Factor", 1.0)),
                        )
                    )
                st.session_state.machines = new_machines
                st.success(f"Successfully loaded {len(new_machines)} machines.")
            except Exception as e:
                st.error(f"Error parsing machine CSV: {e}")

        machines_df = pd.DataFrame(
            [
                {
                    "Machine ID": m.machine_id,
                    "Machine Type": m.machine_type,
                    "Status": m.status,
                    "Current Job": m.current_job,
                    "Speed Factor": m.speed_factor,
                }
                for m in st.session_state.machines
            ]
        )
        edited_machines_df = st.data_editor(machines_df, num_rows="dynamic", width=None)
        if st.button("Save Machine Edits"):
            updated_machines = []
            for _, r in edited_machines_df.iterrows():
                updated_machines.append(
                    MachineInfo(
                        machine_id=str(r["Machine ID"]),
                        machine_type=str(r["Machine Type"]),
                        status=str(r["Status"]),
                        current_job=str(r["Current Job"]),
                        speed_factor=float(r["Speed Factor"]),
                    )
                )
            st.session_state.machines = updated_machines
            st.success("Machine information updated successfully!")

    with tab3:
        st.subheader("Historical Manufacturing Data (for ML Training)")
        col_up_h, col_btn_h = st.columns([3, 1])
        with col_up_h:
            hist_file = st.file_uploader("Upload Historical Data CSV", type=["csv"], key="hist_csv")
        with col_btn_h:
            if st.button("Load Sample Historical Data"):
                st.session_state.historical = get_sample_historical_data()
                st.success("Sample historical data loaded!")

        if hist_file is not None:
            try:
                df_h = pd.read_csv(hist_file)
                new_hist = []
                for _, row in df_h.iterrows():
                    new_hist.append(
                        HistoricalDataRecord(
                            part=str(row.get("Part", "Part A")),
                            material=str(row.get("Material", "Steel")),
                            operation=str(row.get("Operation", "Milling")),
                            cutting_speed=float(row.get("Cutting Speed", 250.0)),
                            feed_rate=float(row.get("Feed Rate", 0.25)),
                            depth_of_cut=float(row.get("Depth of Cut", 2.0)),
                            tool_condition=float(row.get("Tool Condition", 0.9)),
                            actual_processing_time=float(row.get("Actual Processing Time", 2.0)),
                        )
                    )
                st.session_state.historical = new_hist
                st.success(f"Successfully loaded {len(new_hist)} historical records.")
            except Exception as e:
                st.error(f"Error parsing historical CSV: {e}")

        hist_df = pd.DataFrame(
            [
                {
                    "Part": h.part,
                    "Material": h.material,
                    "Operation": h.operation,
                    "Cutting Speed": h.cutting_speed,
                    "Feed Rate": h.feed_rate,
                    "Depth of Cut": h.depth_of_cut,
                    "Tool Condition": h.tool_condition,
                    "Actual Processing Time": h.actual_processing_time,
                }
                for h in st.session_state.historical
            ]
        )
        edited_hist_df = st.data_editor(hist_df, num_rows="dynamic", width=None)
        if st.button("Save Historical Data Edits"):
            updated_hist = []
            for _, r in edited_hist_df.iterrows():
                updated_hist.append(
                    HistoricalDataRecord(
                        part=str(r["Part"]),
                        material=str(r["Material"]),
                        operation=str(r["Operation"]),
                        cutting_speed=float(r["Cutting Speed"]),
                        feed_rate=float(r["Feed Rate"]),
                        depth_of_cut=float(r["Depth of Cut"]),
                        tool_condition=float(r["Tool Condition"]),
                        actual_processing_time=float(r["Actual Processing Time"]),
                    )
                )
            st.session_state.historical = updated_hist
            st.success("Historical data updated successfully!")

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
            st.plotly_chart(fig_imp, width=None)
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

    if st.button("Run Optimizer", type="primary", width=None):
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
        st.dataframe(sched_df, width=None)

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
        st.plotly_chart(fig_gantt, width=None)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Machine Workload Distribution")
            workload_df = pd.DataFrame(list(result.machine_workload.items()), columns=["Machine ID", "Workload (hrs)"])
            fig_workload = px.bar(workload_df, x="Machine ID", y="Workload (hrs)", color="Workload (hrs)", color_continuous_scale="Blues", text_auto=True)
            st.plotly_chart(fig_workload, width=None)

        with col2:
            st.subheader("Machine Utilization (%)")
            util_df = pd.DataFrame(list(result.machine_utilization.items()), columns=["Machine ID", "Utilization (%)"])
            fig_util = px.bar(util_df, x="Machine ID", y="Utilization (%)", color="Utilization (%)", color_continuous_scale="Greens", text_auto=True)
            fig_util.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_util, width=None)

# 6. Real-Time Failure & Rescheduling
elif page == "🚨 Real-Time Failure & Rescheduling":
    st.markdown('<div class="main-header">Real-Time Machine Failure Simulation & Rescheduling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate unexpected machine breakdowns and view before-and-after KPI comparisons.</div>', unsafe_allow_html=True)

    machine_ids = [m.machine_id for m in st.session_state.machines]
    downed_selection = st.multiselect("Select Machine(s) to Simulate DOWN", machine_ids)
    reschedule_time = st.slider("Failure Time (hours from start)", 0.0, 24.0, 2.0, 0.5)

    if st.button("Trigger Failure & Reschedule", type="primary", width=None):
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
            st.dataframe(base_df, width=None)

        with col_a_sched:
            st.markdown("**Rescheduled After Failure**")
            after_df = pd.DataFrame([{"Job": j.job_id, "Machine": j.machine_id, "Finish": j.finish_time, "Late": j.is_late} for j in a.scheduled_jobs])
            st.dataframe(after_df, width=None)

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
