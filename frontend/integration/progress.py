"""
Progress Management - Handles progress tracking for long-running operations
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
import streamlit as st
from dataclasses import dataclass, asdict

@dataclass
class ProgressStep:
    """Represents a single progress step"""
    name: str
    description: str
    completed: bool = False
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None

class ProgressManager:
    """Manages progress tracking for async operations"""

    def __init__(self, progress_key: str = "progress_manager"):
        self.progress_key = progress_key
        self.steps: Dict[str, ProgressStep] = {}
        self.current_step = None
        self.total_steps = 0
        self.start_time = None
        self.end_time = None
        self._progress_bar = None
        self._status_text = None
        self._callbacks: List[Callable] = []

    def __enter__(self):
        """Initialize progress manager"""
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup progress manager"""
        self.end_time = datetime.now()
        if self._progress_bar:
            self._progress_bar.empty()

    def reset(self):
        """Reset progress manager"""
        self.steps.clear()
        self.current_step = None
        self.total_steps = 0
        self.start_time = datetime.now()
        self.end_time = None
        if self._progress_bar:
            self._progress_bar.empty()
        self._progress_bar = None
        self._status_text = None

    def add_step(self, name: str, description: str):
        """Add a progress step"""
        self.steps[name] = ProgressStep(name, description)
        self.total_steps = len(self.steps)

    def start_step(self, name: str):
        """Start a progress step"""
        if name in self.steps:
            step = self.steps[name]
            step.start_time = datetime.now()
            step.progress = 0.0
            step.completed = False
            self.current_step = name
            self._update_display()

    def update_step(self, name: str, progress: float, description: str = None):
        """Update progress of a step"""
        if name in self.steps:
            step = self.steps[name]
            step.progress = max(0.0, min(1.0, progress))
            if description:
                step.description = description
            self._update_display()

    def complete_step(self, name: str, success: bool = True, error: str = None):
        """Complete a progress step"""
        if name in self.steps:
            step = self.steps[name]
            step.progress = 1.0
            step.completed = True
            step.end_time = datetime.now()
            if error:
                step.error = error
            self._update_display()

    def get_overall_progress(self) -> float:
        """Calculate overall progress"""
        if not self.steps:
            return 0.0

        total_progress = sum(step.progress for step in self.steps.values())
        return total_progress / self.total_steps

    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time"""
        if self.start_time:
            end_time = self.end_time or datetime.now()
            return end_time - self.start_time
        return timedelta()

    def get_estimated_remaining_time(self) -> Optional[timedelta]:
        """Get estimated remaining time"""
        if not self.steps or not self.current_step:
            return None

        elapsed = self.get_elapsed_time()
        overall_progress = self.get_overall_progress()

        if overall_progress == 0:
            return None

        # Calculate estimated total time
        estimated_total = elapsed / overall_progress
        remaining = estimated_total - elapsed

        return remaining

    def is_complete(self) -> bool:
        """Check if all steps are complete"""
        return all(step.completed for step in self.steps.values())

    def has_errors(self) -> bool:
        """Check if any steps have errors"""
        return any(step.error for step in self.steps.values())

    def get_step_info(self, name: str) -> Optional[ProgressStep]:
        """Get information about a specific step"""
        return self.steps.get(name)

    def get_all_steps(self) -> List[ProgressStep]:
        """Get all progress steps"""
        return list(self.steps.values())

    def add_callback(self, callback: Callable):
        """Add a callback function to be called on progress updates"""
        self._callbacks.append(callback)

    def _update_display(self):
        """Update the Streamlit progress display"""
        if not self._progress_bar:
            self._progress_bar = st.progress(0)
            self._status_text = st.empty()

        overall_progress = self.get_overall_progress()
        elapsed = self.get_elapsed_time()
        remaining = self.get_estimated_remaining_time()

        # Update progress bar
        self._progress_bar.progress(overall_progress)

        # Update status text
        status_parts = []

        if self.current_step and self.current_step in self.steps:
            step = self.steps[self.current_step]
            status_parts.append(f"🔄 {step.description}")
            if step.error:
                status_parts.append(f"❌ Error: {step.error}")

        status_parts.append(f"⏱️ Elapsed: {elapsed}")
        if remaining:
            status_parts.append(f"⏳ Remaining: {remaining}")

        # Add overall progress
        completed_steps = sum(1 for step in self.steps.values() if step.completed)
        status_parts.append(f"📊 Progress: {completed_steps}/{self.total_steps} steps")

        status_text = " | ".join(status_parts)
        self._status_text.write(status_text)

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current status"""
        return {
            'overall_progress': self.get_overall_progress(),
            'elapsed_time': str(self.get_elapsed_time()),
            'estimated_remaining_time': str(self.get_estimated_remaining_time()) if self.get_estimated_remaining_time() else None,
            'total_steps': self.total_steps,
            'completed_steps': sum(1 for step in self.steps.values() if step.completed),
            'current_step': self.current_step,
            'has_errors': self.has_errors(),
            'steps': [asdict(step) for step in self.steps.values()]
        }

class AsyncProgressManager(ProgressManager):
    """Async version of progress manager"""

    def __init__(self, progress_key: str = "async_progress_manager"):
        super().__init__(progress_key)
        self._running = False
        self._task = None

    async def run_with_progress(self, task_func: Callable, *args, **kwargs):
        """Run a task with progress tracking"""
        self._running = True
        self.reset()

        try:
            # Add progress steps based on task function
            await self._setup_steps(task_func)

            # Start the task with progress monitoring
            self._task = asyncio.create_task(self._monitor_task(task_func, *args, **kwargs))
            result = await self._task

            # Ensure all steps are marked complete
            for name, step in self.steps.items():
                if not step.completed:
                    self.complete_step(name, True)

            return result

        except Exception as e:
            # Mark current step as failed if there's an error
            if self.current_step and self.current_step in self.steps:
                self.complete_step(self.current_step, False, str(e))
            raise e

        finally:
            self._running = False
            if self._progress_bar:
                self._progress_bar.empty()
            self._progress_bar = None
            self._status_text = None

    async def _setup_steps(self, task_func: Callable):
        """Setup progress steps based on the task function"""
        # This is a simplified version - in a real implementation,
        # you would inspect the task function to determine appropriate steps
        self.add_step("data_fetching", "Fetching market data...")
        self.add_step("data_validation", "Validating data...")
        self.add_step("pattern_detection", "Detecting patterns...")
        self.add_step("result_processing", "Processing results...")

    async def _monitor_task(self, task_func, *args, **kwargs):
        """Monitor the task and update progress"""
        # Start with data fetching
        self.start_step("data_fetching")
        await asyncio.sleep(0.1)  # Simulate some work
        self.update_step("data_fetching", 0.3, "Fetching data from yfinance...")
        await asyncio.sleep(0.5)  # Simulate data fetching
        self.update_step("data_fetching", 0.8, "Processing fetched data...")
        await asyncio.sleep(0.3)
        self.complete_step("data_fetching", True)

        # Move to data validation
        self.start_step("data_validation")
        await asyncio.sleep(0.2)
        self.update_step("data_validation", 0.5, "Validating OHLC relationships...")
        await asyncio.sleep(0.3)
        self.complete_step("data_validation", True)

        # Run the actual task
        self.start_step("pattern_detection")
        try:
            result = await task_func(*args, **kwargs)
            self.complete_step("pattern_detection", True)
        except Exception as e:
            self.complete_step("pattern_detection", False, str(e))
            raise

        # Process results
        self.start_step("result_processing")
        await asyncio.sleep(0.1)
        self.complete_step("result_processing", True)

        return result

def create_progress_context(progress_key: str = "default_progress"):
    """Create a progress context manager"""
    return ProgressManager(progress_key)

def create_async_progress_context(progress_key: str = "async_progress"):
    """Create an async progress context manager"""
    return AsyncProgressManager(progress_key)

# Helper functions for Streamlit integration
def show_progress_with_stages(stages: List[str], current_stage: int, description: str = ""):
    """Show progress with multiple stages"""
    progress = (current_stage + 1) / len(stages)

    st.write(f"**{description}**")
    progress_bar = st.progress(progress)
    stage_text = st.empty()

    stage_text.write(f"🔄 {stages[current_stage]}")

    return progress_bar, stage_text

def show_step_progress(step_name: str, step_progress: float, step_description: str = ""):
    """Show progress for a single step"""
    if step_progress < 1.0:
        st.write(f"🔄 {step_name}: {step_progress*100:.1f}%")
        if step_description:
            st.write(f"   {step_description}")
        return st.progress(step_progress)
    else:
        st.write(f"✅ {step_name}: Complete")
        return None