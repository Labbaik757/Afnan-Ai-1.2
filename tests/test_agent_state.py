import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.state import (
    AgentState,
    Observation,
    StepRecord,
    StepStatus,
    TaskStatus,
    ToolResult,
)


class TestAgentStateCreation(unittest.TestCase):
    def test_create_defaults(self):
        state = AgentState.create("Open Chrome")
        self.assertEqual(state.goal, "Open Chrome")
        self.assertEqual(state.status, TaskStatus.PENDING)
        self.assertIsNone(state.current_step)
        self.assertEqual(state.completed_steps, [])
        self.assertEqual(state.failed_steps, [])
        self.assertEqual(state.observations, [])
        self.assertEqual(state.tool_results, [])
        self.assertTrue(state.task_id)
        self.assertTrue(state.created_at)
        self.assertTrue(state.updated_at)

    def test_create_unique_ids(self):
        a = AgentState.create("task a")
        b = AgentState.create("task b")
        self.assertNotEqual(a.task_id, b.task_id)

    def test_create_custom_id_and_metadata(self):
        state = AgentState.create("task", task_id="abc123", metadata={"os": "test"})
        self.assertEqual(state.task_id, "abc123")
        self.assertEqual(state.metadata, {"os": "test"})

    def test_create_empty_goal_rejected(self):
        with self.assertRaises(ValueError):
            AgentState.create("   ")

    def test_state_is_platform_independent(self):
        # State must round-trip with plain JSON types only, no OS paths
        state = AgentState.create("do something")
        data = state.to_dict()
        json.dumps(data)  # must not raise
        for value in data.values():
            self.assertNotIn("\\", str(value) if isinstance(value, str) else "")


class TestAgentStateUpdate(unittest.TestCase):
    def test_start_and_complete_step(self):
        state = AgentState.create("Search the web")
        state.start_task()
        self.assertEqual(state.status, TaskStatus.RUNNING)

        step = state.start_step("search google")
        self.assertEqual(state.current_step, "search google")
        self.assertEqual(step.status, StepStatus.RUNNING)

        done = state.complete_step("search google", result="3 results")
        self.assertIsNone(state.current_step)
        self.assertEqual([s.name for s in state.completed_steps], ["search google"])
        self.assertEqual(done.result, "3 results")
        self.assertEqual(done.status, StepStatus.COMPLETED)
        self.assertIsNotNone(done.finished_at)

    def test_add_observation_and_tool_result(self):
        state = AgentState.create("Open YouTube")
        obs = state.add_observation("User said open youtube", source="microphone")
        self.assertIsInstance(obs, Observation)
        self.assertEqual(state.observations[0].source, "microphone")

        result = state.add_tool_result("youtube", success=True, output="opened")
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(state.tool_results[0].success)

    def test_start_step_sets_running_status(self):
        state = AgentState.create("task")
        self.assertEqual(state.status, TaskStatus.PENDING)
        state.start_step("first")
        self.assertEqual(state.status, TaskStatus.RUNNING)

    def test_complete_task_completes_running_step(self):
        state = AgentState.create("task")
        state.start_step("only step")
        state.complete_task(result="done")
        self.assertEqual(state.status, TaskStatus.COMPLETED)
        self.assertTrue(state.is_terminal)
        self.assertEqual([s.name for s in state.completed_steps], ["only step"])

    def test_pause_and_summary(self):
        state = AgentState.create("task")
        state.start_step("a")
        state.complete_step("a")
        state.pause_task()
        summary = state.summary()
        self.assertEqual(summary["status"], "paused")
        self.assertEqual(summary["completed"], ["a"])
        self.assertEqual(summary["failed"], [])


class TestAgentStateSerialization(unittest.TestCase):
    def make_populated_state(self):
        state = AgentState.create("Open Chrome and search", metadata={"user": "test"})
        state.start_task()
        state.add_observation("open chrome", source="microphone")
        state.start_step("open chrome")
        state.add_tool_result("chrome", success=True, output="opened")
        state.complete_step("open chrome", result="opened")
        state.start_step("search")
        state.complete_step("search", result="done")
        state.complete_task()
        return state

    def test_to_dict_is_json_serializable(self):
        state = self.make_populated_state()
        text = json.dumps(state.to_dict())
        self.assertIn("Open Chrome", text)

    def test_json_round_trip_is_lossless(self):
        state = self.make_populated_state()
        restored = AgentState.from_json(state.to_json())
        self.assertEqual(restored.goal, state.goal)
        self.assertEqual(restored.task_id, state.task_id)
        self.assertEqual(restored.status, TaskStatus.COMPLETED)
        self.assertEqual(
            [s.name for s in restored.completed_steps],
            ["open chrome", "search"],
        )
        self.assertEqual(restored.observations[0].text, "open chrome")
        self.assertEqual(restored.tool_results[0].tool, "chrome")
        self.assertEqual(restored.metadata, {"user": "test"})

    def test_dict_round_trip(self):
        state = self.make_populated_state()
        restored = AgentState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_save_and_load_file_cross_platform(self):
        state = self.make_populated_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "state.json"
            saved = state.save(path)
            self.assertTrue(saved.exists())
            loaded = AgentState.load(path)
            self.assertEqual(loaded.goal, state.goal)
            self.assertEqual(loaded.status, TaskStatus.COMPLETED)

    def test_from_json_rejects_bad_status_gracefully(self):
        # unknown fields are ignored, known ones are restored
        state = AgentState.from_dict(
            {"goal": "x", "status": "running", "completed_steps": [], "failed_steps": []}
        )
        self.assertEqual(state.status, TaskStatus.RUNNING)


class TestAgentStateFailure(unittest.TestCase):
    def test_fail_step_records_failure(self):
        state = AgentState.create("Open an app")
        state.start_step("open app")
        failed = state.fail_step("open app", "app not found")
        self.assertEqual([s.name for s in state.failed_steps], ["open app"])
        self.assertEqual(failed.error, "app not found")
        self.assertEqual(failed.status, StepStatus.FAILED)
        self.assertIsNone(state.current_step)
        self.assertEqual(state.completed_steps, [])

    def test_fail_task_sets_failed_status_and_error(self):
        state = AgentState.create("task")
        state.start_step("step 1")
        state.fail_task("network down")
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual(state.error, "network down")
        self.assertTrue(state.is_terminal)
        self.assertFalse(state.is_successful)
        # the running step is recorded as failed, not lost
        self.assertEqual([s.name for s in state.failed_steps], ["step 1"])

    def test_failed_state_survives_serialization(self):
        state = AgentState.create("task")
        state.start_step("a")
        state.complete_step("a")
        state.start_step("b")
        state.fail_step("b", "boom")
        state.add_tool_result("tool-b", success=False, error="boom")

        restored = AgentState.from_json(state.to_json())
        self.assertEqual([s.name for s in restored.completed_steps], ["a"])
        self.assertEqual([s.name for s in restored.failed_steps], ["b"])
        self.assertEqual(restored.failed_steps[0].error, "boom")
        self.assertFalse(restored.tool_results[0].success)

    def test_cancel_task(self):
        state = AgentState.create("task")
        state.start_step("a")
        state.cancel_task()
        self.assertEqual(state.status, TaskStatus.CANCELLED)
        self.assertIsNone(state.current_step)
        self.assertTrue(state.is_terminal)


class FakeAdapter(PlatformAdapter):
    name = "linux"

    def __init__(self):
        self.spoken = []
        self.opened = []
        self.launched = []

    def speak_system(self, text):
        self.spoken.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def launch_app(self, app_key):
        self.launched.append(app_key)
        return True

    def find_folder(self, foldername):
        return None


class TestAgentStateIntegration(unittest.TestCase):
    def make_agent(self):
        agent = AfnanAgent(adapter=FakeAdapter())
        agent.speak = lambda text: None
        return agent

    def test_agent_tracks_successful_command(self):
        agent = self.make_agent()
        agent.process_command("open chrome")
        self.assertIsNotNone(agent.state)
        self.assertEqual(agent.state.observations[0].text, "open chrome")
        self.assertEqual(
            [s.name for s in agent.state.completed_steps], ["open chrome"]
        )
        self.assertEqual(agent.state.failed_steps, [])

    def test_agent_tracks_multiple_commands_in_one_task(self):
        agent = self.make_agent()
        with mock.patch("webbrowser.open"):
            agent.process_command("open chrome")
            agent.process_command("open youtube")
        names = [s.name for s in agent.state.completed_steps]
        self.assertEqual(names, ["open chrome", "open youtube"])

    def test_agent_start_task_api(self):
        agent = self.make_agent()
        state = agent.start_task("Search for Python")
        self.assertEqual(state.goal, "Search for Python")
        self.assertEqual(state.status, TaskStatus.RUNNING)
        self.assertIs(agent.state, state)

    def test_agent_can_be_given_a_shared_state(self):
        shared = AgentState.create("Shared goal")
        agent = AfnanAgent(adapter=FakeAdapter(), state=shared)
        agent.speak = lambda text: None
        agent.process_command("open chrome")
        # the shared object is the one that was updated
        self.assertIn("open chrome", [o.text for o in shared.observations])

    def test_agent_state_can_be_saved_mid_task(self):
        agent = self.make_agent()
        agent.process_command("open chrome")
        with tempfile.TemporaryDirectory() as tmp:
            path = agent.state.save(Path(tmp) / "state.json")
            loaded = AgentState.load(path)
            self.assertEqual(loaded.goal, agent.state.goal)

    def test_tracking_can_be_disabled(self):
        agent = AfnanAgent(adapter=FakeAdapter(), track_state=False)
        agent.speak = lambda text: None
        agent.process_command("open chrome")
        self.assertIsNone(agent.state)


if __name__ == "__main__":
    unittest.main()
