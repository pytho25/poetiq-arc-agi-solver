#!/usr/bin/env python3
"""
Interactive TUI application for ARC-AGI puzzle solver using Textual.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from rich.console import RenderableType
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Footer, Input, Label, Log, RichLog, Static, TextArea

from arc_agi.config import CONFIG_LIST
from arc_agi.llm import llm
from arc_agi.sandbox import run


# Color mapping for grid values 0-9
COLOR_MAP = {
    0: "black",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "red",
    5: "magenta",
    6: "cyan",
    7: "white",
    8: "orange3",
    9: "maroon",
}


@dataclass
class ChatEntry:
    """Represents a chat message entry."""
    role: str  # "user" or "assistant"
    content: str
    code: Optional[str] = None  # Extracted Python code
    timestamp: datetime = field(default_factory=datetime.now)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GridDisplay(Static):
    """Widget to display a 2D grid with colored backgrounds or numeric display."""
    
    def __init__(self, grid: list[list[int]], label: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = grid
        self.label = label
        self.colored_mode = True
    
    def toggle_display(self):
        """Toggle between colored blocks and numeric ASCII display."""
        self.colored_mode = not self.colored_mode
        self.update_display()
    
    def update_display(self):
        """Update the display based on current mode."""
        if not self.grid:
            self.update("")
            return
        
        if self.colored_mode:
            renderable = self._render_colored()
            self.update(renderable)
        else:
            self.update(self._render_numeric())
    
    def _render_colored(self) -> RenderableType:
        """Render grid as colored blocks using Rich."""
        if not self.grid:
            return Text("")
        
        text = Text()
        if self.label:
            text.append(self.label + "\n", style="bold")
        
        for row in self.grid:
            for val in row:
                color = COLOR_MAP.get(val, "white")
                # Use Rich styling with background color
                text.append(f" {val} ", style=f"on {color} black" if val == 0 else f"on {color}")
            text.append("\n")
        
        return text
    
    def _render_numeric(self) -> str:
        """Render grid as numeric ASCII."""
        lines = []
        if self.label:
            lines.append(self.label)
        for row in self.grid:
            line = " ".join(str(val) for val in row)
            lines.append(line)
        return "\n".join(lines)
    
    def on_mount(self):
        """Called when widget is mounted."""
        self.update_display()


class ChallengeDisplay(ScrollableContainer):
    """Container for displaying challenge examples and test inputs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.challenge_data: Optional[dict] = None
        self.grid_displays: list[GridDisplay] = []
    
    def load_challenge(self, challenge_data: dict):
        """Load and display challenge data."""
        self.challenge_data = challenge_data
        self.grid_displays.clear()
        self.clear_children()
        
        # Display training examples
        if "train" in challenge_data:
            train_label = Label("Training Examples:", classes="section-label")
            self.mount(train_label)
            
            for i, example in enumerate(challenge_data["train"]):
                example_container = Vertical(classes="example-container")
                
                input_grid = example.get("input", [])
                output_grid = example.get("output", [])
                
                input_display = GridDisplay(input_grid, f"Train {i+1} Input:")
                output_display = GridDisplay(output_grid, f"Train {i+1} Output:")
                
                self.grid_displays.extend([input_display, output_display])
                
                example_container.mount(input_display)
                example_container.mount(Static(" → "))
                example_container.mount(output_display)
                
                self.mount(example_container)
        
        # Display test inputs
        if "test" in challenge_data:
            test_label = Label("Test Inputs:", classes="section-label")
            self.mount(test_label)
            
            for i, test_input in enumerate(challenge_data["test"]):
                input_grid = test_input.get("input", [])
                test_display = GridDisplay(input_grid, f"Test {i+1}:")
                self.grid_displays.append(test_display)
                self.mount(test_display)
    
    def toggle_all_displays(self):
        """Toggle display mode for all grids."""
        for display in self.grid_displays:
            display.toggle_display()


class ChatHistory(ScrollableContainer):
    """Container for displaying chat history."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries: list[ChatEntry] = []
    
    def add_entry(self, entry: ChatEntry):
        """Add a chat entry to the history."""
        self.entries.append(entry)
        self._render_entry(entry)
    
    def _render_entry(self, entry: ChatEntry):
        """Render a single chat entry."""
        container = Vertical(classes=f"chat-entry chat-{entry.role}")
        
        # Header with role and timestamp
        header = Static(f"[{entry.role.upper()}] {entry.timestamp.strftime('%H:%M:%S')}")
        if entry.prompt_tokens > 0 or entry.completion_tokens > 0:
            header.update(f"[{entry.role.upper()}] {entry.timestamp.strftime('%H:%M:%S')} | Tokens: {entry.prompt_tokens}+{entry.completion_tokens}")
        container.mount(header)
        
        # Content
        content = Static(entry.content)
        container.mount(content)
        
        # Code section if present
        if entry.code:
            code_container = Vertical(classes="code-container")
            code_label = Static("Extracted Code:")
            code_container.mount(code_label)
            code_display = TextArea(entry.code, language="python", read_only=True, classes="code-display")
            code_container.mount(code_display)
            
            # Use This Code button
            use_button = Button("Use This Code", id=f"use-code-{len(self.entries)-1}", classes="use-code-btn")
            code_container.mount(use_button)
            
            container.mount(code_container)
        
        self.mount(container)
        self.scroll_end(animate=False)


class ExecutionResults(Static):
    """Widget to display code execution results."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results: Optional[list] = None
    
    def update_results(self, results: list):
        """Update with execution results."""
        self.results = results
        if not results:
            self.update("No results")
            return
        
        lines = ["Execution Results:"]
        for i, result in enumerate(results):
            if isinstance(result, dict):
                success = result.get("success", False)
                status = "✓ PASS" if success else "✗ FAIL"
                lines.append(f"  Example {i+1}: {status}")
                if not success and "error" in result:
                    lines.append(f"    Error: {result['error']}")
            else:
                lines.append(f"  Example {i+1}: {result}")
        
        self.update("\n".join(lines))


class ARCAGITUI(App):
    """Main TUI application for ARC-AGI puzzle solver."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #left-panel {
        width: 35%;
        border-right: wide $primary;
    }
    
    #right-panel {
        width: 65%;
    }
    
    .section-label {
        text-style: bold;
        margin: 1;
    }
    
    .example-container {
        height: auto;
        margin: 1;
    }
    
    .chat-entry {
        margin: 1;
        padding: 1;
        border: solid $primary;
    }
    
    .chat-user {
        background: $surface;
    }
    
    .chat-assistant {
        background: $panel;
    }
    
    .code-container {
        margin-top: 1;
        border: solid $accent;
    }
    
    .code-display {
        height: 10;
    }
    
    .use-code-btn {
        margin: 1;
    }
    
    #execution-results {
        height: 10;
        border: solid $warning;
        padding: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "toggle_display", "Toggle Display"),
        Binding("r", "run_code", "Run Code"),
        Binding("ctrl+j", "send_message", "Send Message"),
    ]
    
    def __init__(self):
        super().__init__()
        self.challenge_id_input: Optional[Input] = None
        self.challenge_display: Optional[ChallengeDisplay] = None
        self.chat_history: Optional[ChatHistory] = None
        self.message_input: Optional[TextArea] = None
        self.execution_results: Optional[ExecutionResults] = None
        self.current_challenge: Optional[dict] = None
        self.current_code: Optional[str] = None
        self.config = CONFIG_LIST[0]
        self.challenges_data: dict = {}
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Horizontal():
            # Left panel (35%)
            with Vertical(id="left-panel"):
                yield Label("Challenge ID:", classes="section-label")
                yield Input(placeholder="Enter challenge ID", id="challenge-id-input")
                yield Button("Load Challenge", id="load-btn", variant="primary")
                yield Button("Toggle Display", id="toggle-display-btn")
                yield ChallengeDisplay(id="challenge-display")
            
            # Right panel (65%)
            with Vertical(id="right-panel"):
                yield Label("Chat History:", classes="section-label")
                yield ChatHistory(id="chat-history")
                yield Label("Execution Results:", classes="section-label")
                yield ExecutionResults(id="execution-results")
                yield Label("Message:", classes="section-label")
                yield TextArea(placeholder="Type your message here... (Ctrl+Enter to send)", id="message-input")
                with Horizontal():
                    yield Button("Send", id="send-btn", variant="primary")
                    yield Button("Run Code", id="run-btn")
        
        yield Footer()
    
    def on_mount(self):
        """Called when app is mounted."""
        self.challenge_id_input = self.query_one("#challenge-id-input", Input)
        self.challenge_display = self.query_one("#challenge-display", ChallengeDisplay)
        self.chat_history = self.query_one("#chat-history", ChatHistory)
        self.message_input = self.query_one("#message-input", TextArea)
        self.execution_results = self.query_one("#execution-results", ExecutionResults)
        
        # Load challenges data
        self.load_challenges_file()
    
    def load_challenges_file(self):
        """Load challenges from JSON file."""
        challenges_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "arc-prize-2024",
            "arc-agi_evaluation_challenges.json"
        )
        try:
            with open(challenges_path, "r") as f:
                self.challenges_data = json.load(f)
            self.notify(f"Loaded {len(self.challenges_data)} challenges", severity="success")
        except Exception as e:
            self.notify(f"Failed to load challenges: {e}", severity="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "load-btn":
            self.load_challenge()
        elif event.button.id == "toggle-display-btn":
            self.action_toggle_display()
        elif event.button.id == "send-btn":
            self.action_send_message()
        elif event.button.id == "run-btn":
            self.action_run_code()
        elif event.button.id and event.button.id.startswith("use-code-"):
            # Extract index from button ID
            idx = int(event.button.id.split("-")[-1])
            self.use_code_from_entry(idx)
    
    def load_challenge(self):
        """Load a challenge by ID."""
        challenge_id = self.challenge_id_input.value.strip()
        if not challenge_id:
            self.notify("Please enter a challenge ID", severity="warning")
            return
        
        if challenge_id not in self.challenges_data:
            self.notify(f"Challenge '{challenge_id}' not found", severity="error")
            return
        
        self.current_challenge = self.challenges_data[challenge_id]
        self.challenge_display.load_challenge(self.current_challenge)
        self.notify(f"Loaded challenge: {challenge_id}", severity="success")
    
    def action_toggle_display(self):
        """Toggle display mode for all grids."""
        # Toggle all grids in challenge display
        if self.challenge_display:
            self.challenge_display.toggle_all_displays()
        # Also toggle any grids in chat history if needed
        for widget in self.query(GridDisplay):
            widget.toggle_display()
    
    def action_send_message(self):
        """Send a message to the LLM."""
        message = self.message_input.text.strip()
        if not message:
            self.notify("Please enter a message", severity="warning")
            return
        
        if not self.current_challenge:
            self.notify("Please load a challenge first", severity="warning")
            return
        
        # Add user message to chat
        user_entry = ChatEntry(role="user", content=message)
        self.chat_history.add_entry(user_entry)
        
        # Clear input
        self.message_input.text = ""
        
        # Send to LLM
        self.send_to_llm(message)
    
    def action_run_code(self):
        """Run the current code against training examples."""
        if not self.current_code:
            self.notify("No code available. Send a message to generate code first.", severity="warning")
            return
        
        if not self.current_challenge:
            self.notify("Please load a challenge first", severity="warning")
            return
        
        self.run_code_async(self.current_code)
    
    def use_code_from_entry(self, entry_idx: int):
        """Use code from a specific chat entry."""
        if entry_idx < 0 or entry_idx >= len(self.chat_history.entries):
            return
        
        entry = self.chat_history.entries[entry_idx]
        if entry.code:
            self.current_code = entry.code
            self.notify(f"Using code from {entry.role} message at {entry.timestamp.strftime('%H:%M:%S')}", severity="success")
            # Optionally run it immediately
            self.action_run_code()
    
    @work(exclusive=True)
    async def send_to_llm(self, message: str):
        """Send message to LLM and handle response."""
        try:
            self.notify("Sending message to LLM...", severity="info")
            
            # Build prompt with challenge context
            prompt = self._build_prompt(message)
            
            # Call LLM
            response, duration, _, _, prompt_tokens, completion_tokens = await llm(
                model=self.config["llm_id"],
                message=prompt,
                temperature=self.config["solver_temperature"],
                request_timeout=self.config.get("request_timeout"),
                max_remaining_time=self.config.get("max_total_time"),
                max_remaining_timeouts=self.config.get("max_total_timeouts"),
                problem_id=None,
                retries=self.config.get("per_iteration_retries", 3),
            )
            
            # Extract code
            code = self._extract_code(response)
            
            # Add assistant response to chat
            assistant_entry = ChatEntry(
                role="assistant",
                content=response,
                code=code,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self.chat_history.add_entry(assistant_entry)
            
            # Update current code if extracted
            if code:
                self.current_code = code
                self.notify(f"Code extracted. Duration: {duration:.1f}s", severity="success")
            else:
                self.notify("No code found in response", severity="warning")
        
        except Exception as e:
            self.notify(f"Error calling LLM: {e}", severity="error")
            error_entry = ChatEntry(
                role="assistant",
                content=f"Error: {str(e)}",
            )
            self.chat_history.add_entry(error_entry)
    
    @work(exclusive=True)
    async def run_code_async(self, code: str):
        """Run code against training examples."""
        if not self.current_challenge or "train" not in self.current_challenge:
            return
        
        try:
            self.notify("Running code...", severity="info")
            results = []
            
            train_examples = self.current_challenge["train"]
            for i, example in enumerate(train_examples):
                input_grid = example.get("input", [])
                expected_output = example.get("output", [])
                
                # Run code
                ok, output_str = await run(code, input_grid, timeout_s=5.0)
                
                if not ok:
                    results.append({
                        "success": False,
                        "error": output_str,
                        "example": i + 1,
                    })
                else:
                    # Parse output
                    try:
                        output_data = json.loads(output_str)
                        computed_output = np.array(output_data)
                        expected_output_arr = np.array(expected_output)
                        
                        # Compare
                        success = (
                            computed_output.shape == expected_output_arr.shape
                            and np.array_equal(computed_output, expected_output_arr)
                        )
                        
                        results.append({
                            "success": success,
                            "example": i + 1,
                            "computed": computed_output.tolist(),
                            "expected": expected_output.tolist(),
                        })
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": f"Failed to parse output: {e}",
                            "example": i + 1,
                        })
            
            # Update results display
            self.execution_results.update_results(results)
            
            # Show summary
            passed = sum(1 for r in results if r.get("success", False))
            total = len(results)
            self.notify(f"Results: {passed}/{total} passed", severity="success" if passed == total else "warning")
        
        except Exception as e:
            self.notify(f"Error running code: {e}", severity="error")
            self.execution_results.update_results([{"success": False, "error": str(e)}])
    
    def _build_prompt(self, user_message: str) -> str:
        """Build prompt with challenge context."""
        if not self.current_challenge:
            return user_message
        
        prompt_parts = [user_message]
        
        # Add training examples
        if "train" in self.current_challenge:
            prompt_parts.append("\n\nTraining Examples:")
            for i, example in enumerate(self.current_challenge["train"]):
                prompt_parts.append(f"\nExample {i+1}:")
                prompt_parts.append(f"Input: {example.get('input', [])}")
                prompt_parts.append(f"Output: {example.get('output', [])}")
        
        # Add test inputs
        if "test" in self.current_challenge:
            prompt_parts.append("\n\nTest Inputs:")
            for i, test_input in enumerate(self.current_challenge["test"]):
                prompt_parts.append(f"\nTest {i+1}:")
                prompt_parts.append(f"Input: {test_input.get('input', [])}")
        
        return "\n".join(prompt_parts)
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "challenge-id-input":
            self.load_challenge()
    
    def on_key(self, event) -> None:
        """Handle key events."""
        # Handle Ctrl+Enter to send message when focus is on message input
        if event.key == "ctrl+j":
            focused = self.focused
            if isinstance(focused, TextArea) and focused.id == "message-input":
                self.action_send_message()
                event.prevent_default()
                return
        # Let other keys be handled normally
        return super().on_key(event)


def main():
    """Main entry point."""
    app = ARCAGITUI()
    app.run()


if __name__ == "__main__":
    main()
