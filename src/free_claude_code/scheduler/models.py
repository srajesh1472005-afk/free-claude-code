"""Data models for Smart Production Scheduling and Real-Time Rescheduling System."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ProductionOrder:
    job_id: str
    part: str
    machine_id: str
    quantity: int
    material: str
    operation: str
    processing_time: float  # Base processing time per unit or total (hours/mins)
    setup_time: float  # Setup/changeover time when switching parts
    priority: int  # 1 (lowest) to 5 (highest)
    due_date: float  # Due time/deadline in hours from start


@dataclass
class MachineInfo:
    machine_id: str
    machine_type: str
    status: Literal["AVAILABLE", "DOWN"]
    current_job: str = "None"
    speed_factor: float = 1.0  # Efficiency multiplier


@dataclass
class HistoricalDataRecord:
    part: str
    material: str
    operation: str
    cutting_speed: float
    feed_rate: float
    depth_of_cut: float
    tool_condition: float  # e.g., 0.0 (worn) to 1.0 (new)
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
