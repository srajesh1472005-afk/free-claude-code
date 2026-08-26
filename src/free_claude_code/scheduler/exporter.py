"""CSV data exporter for production schedules, machine statuses, and KPI reports."""

import pandas as pd

from free_claude_code.scheduler.models import (
    MachineInfo,
    ProductionOrder,
    ReschedulingComparison,
    ScheduleResult,
)


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
