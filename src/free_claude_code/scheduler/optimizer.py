"""Optimization and scheduling engine supporting 5 optimization objectives and changeover management."""

from typing import Literal

from free_claude_code.scheduler.ml import ProcessingTimePredictor
from free_claude_code.scheduler.models import (
    MachineInfo,
    ProductionOrder,
    ScheduledJob,
    ScheduleResult,
)

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
            m
            for m in self.machines
            if m.status == "AVAILABLE" and m.machine_id not in excluded
        ]

        if not active_machines:
            # Fallback if all machines are downed
            active_machines = [
                MachineInfo(
                    machine_id="EMERGENCY-01", machine_type="Backup", status="AVAILABLE"
                )
            ]

        # Sort orders based on optimization objective
        sorted_orders = self._sort_orders_by_objective(self.orders, objective)

        # Machine availability tracking (machine_id -> finish_time)
        machine_ready_time: dict[str, float] = {
            m.machine_id: 0.0 for m in active_machines
        }
        # Last part processed on machine for changeover calculation
        machine_last_part: dict[str, str] = {m.machine_id: "" for m in active_machines}

        scheduled_jobs: list[ScheduledJob] = []

        for order in sorted_orders:
            # Determine target machine
            target_machine = order.machine_id
            if target_machine not in machine_ready_time:
                # Assign to machine with earliest ready time
                target_machine = min(
                    machine_ready_time, key=lambda m: machine_ready_time[m]
                )

            # Calculate processing time (using ML predictor if available)
            base_time = order.processing_time * (
                order.quantity / max(1, order.quantity)
            )  # scaled or per batch
            pred_time = base_time
            if self.predictor and self.predictor.is_trained:
                pred_time = self.predictor.predict_processing_time(
                    part=order.part,
                    material=order.material,
                    operation=order.operation,
                    fallback_time=base_time,
                )

            # Adjust by machine speed factor
            machine_info = next(
                (m for m in self.machines if m.machine_id == target_machine), None
            )
            speed_factor = machine_info.speed_factor if machine_info else 1.0
            adjusted_processing_time = pred_time / max(0.1, speed_factor)

            # Calculate changeover time
            last_part = machine_last_part.get(target_machine, "")
            changeover = (
                order.setup_time
                if (last_part and last_part != order.part)
                else (0.1 if last_part == "" else 0.0)
            )
            if objective == "Minimum changeover time" and last_part == order.part:
                changeover = 0.0  # Encourage batching same parts

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

        # Calculate KPIs
        makespan = max((j.finish_time for j in scheduled_jobs), default=0.0)
        total_processing = sum(j.predicted_processing_time for j in scheduled_jobs)
        total_changeover = sum(j.changeover_time for j in scheduled_jobs)
        total_available_time = (
            makespan * len(active_machines) if active_machines else 1.0
        )
        total_idle_time = max(
            0.0, total_available_time - (total_processing + total_changeover)
        )

        machine_workload: dict[str, float] = {
            m.machine_id: 0.0 for m in active_machines
        }
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
        bottleneck = (
            max(machine_workload, key=machine_workload.get)
            if machine_workload
            else "None"
        )

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
            # Sort by Due Date ASC, then Priority DESC
            return sorted(orders, key=lambda o: (o.due_date, -o.priority))
        elif objective == "Minimum production time":
            # Sort by Processing Time ASC (Shortest Processing Time first)
            return sorted(orders, key=lambda o: (o.processing_time, -o.priority))
        elif objective == "Minimum changeover time":
            # Group by Part/Material to minimize changeovers
            return sorted(orders, key=lambda o: (o.part, o.material))
        elif objective == "Minimum machine idle time":
            # Sort by Processing Time DESC to keep machines loaded
            return sorted(orders, key=lambda o: (-o.processing_time, o.due_date))
        else:
            # Balanced optimization: Priority DESC, Due Date ASC
            return sorted(orders, key=lambda o: (-o.priority, o.due_date))
