#!/usr/bin/env python3
"""Interactive TUI for ARC-AGI challenge solving."""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TextArea,
)

from arc_agi.config import CONFIG_LIST
from arc_agi.llm import llm
from arc_agi.sandbox import run as sandbox_run

load_dotenv()

# Rich color mappings for ARC grid values (0-9)
RICH_COLORS = {
    0: ("black", "  "),
    1: ("blue", "  "),
    2: ("green", "  "),
    3: ("yellow", "  "),
    4: ("red", "  "),
    5: ("magenta", "  "),
    6: ("cyan", "  "),
    7: ("white", "  "),
    8: ("rgb(255,165,0)", "  "),  # Orange
    9: ("rgb(128,0,0)", "  "),    # Maroon/Brown
}

# Data paths
CHALLENGES_PATH = "data/arc-prize-2024/arc-agi_evaluation_challenges.json"
SOLUTIONS_PATH = "data/arc-prize-2024/arc-agi_evaluation_solutions.json"


@dataclass
class ChatEntry:
    """Single chat message entry."""
    role: str  # "user" or "assistant"
    content: str
    code: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: tuple[int, int] | None = None  # (prompt, completion)


def extract_code_from_response(response: str) -> str | None:
    """Extract Python code block from LLM response."""
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def format_grid_as_python(grid: list[list[int]]) -> str:
    """Format a grid as a Python list string."""
    lines = ["["]
    for i, row in enumerate(grid):
        comma = "," if i < len(grid) - 1 else ""
        lines.append(f"    {list(row)}{comma}")
    lines.append("]")
    return "\n".join(lines)


def format_challenge_for_prompt(challenge: dict) -> str:
    """Format challenge data for LLM prompt."""
    lines = ["# Training Examples\n"]

    for i, example in enumerate(challenge["train"], 1):
        lines.append(f"## Example {i}\n")
        lines.append(f"Input ({len(example['input'])}x{len(example['input'][0])}):")
        lines.append("```python")
        lines.append(f"input_{i} = {format_grid_as_python(example['input'])}")
        lines.append("```\n")
        lines.append(f"Output ({len(example['output'])}x{len(example['output'][0])}):")
        lines.append("```python")
        lines.append(f"output_{i} = {format_grid_as_python(example['output'])}")
        lines.append("```\n")

    lines.append("# Test Input\n")
    for i, test in enumerate(challenge["test"], 1):
        lines.append(f"Test {i} ({len(test['input'])}x{len(test['input'][0])}):")
        lines.append("```python")
        lines.append(f"test_input_{i} = {format_grid_as_python(test['input'])}")
        lines.append("```\n")

    return "\n".join(lines)


class GridDisplay(Static):
    """Widget to display an ARC grid with colors or numbers."""

    def __init__(
        self,
        grid: list[list[int]],
        label: str = "",
        colored: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grid = grid
        self.label = label
        self.colored = colored

    def render(self) -> Text:
        """Render the grid."""
        if not self.grid:
            return Text("(empty grid)")

        text = Text()

        # Add label with dimensions
        if self.label:
            h, w = len(self.grid), len(self.grid[0]) if self.grid else 0
            text.append(f"{self.label} ({h}x{w})\n", style="bold")

        for row in self.grid:
            for cell in row:
                if self.colored:
                    color, chars = RICH_COLORS.get(cell, ("white", f"{cell:2}"))
                    text.append(chars, style=f"on {color}")
                else:
                    text.append(f"{cell} ")
            text.append("\n")

        return text


class TrainingExample(Static):
    """Widget showing a single training example."""

    def __init__(
        self,
        example_num: int,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
        colored: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.example_num = example_num
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.colored = colored

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]Example {self.example_num}[/]")
        with Horizontal(classes="example-grids"):
            yield GridDisplay(self.input_grid, "Input", self.colored)
            yield Static("  ->  ", classes="arrow")
            yield GridDisplay(self.output_grid, "Output", self.colored)


class ChatMessage(Static):
    """Widget for displaying a single chat message."""

    def __init__(self, entry: ChatEntry, message_id: int, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry
        self.message_id = message_id
        self.add_class(f"chat-{entry.role}")

    def compose(self) -> ComposeResult:
        role_style = "bold cyan" if self.entry.role == "user" else "bold green"
        yield Static(f"[{role_style}]{self.entry.role.upper()}[/]")

        # Show content (truncated for display if very long)
        content = self.entry.content
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        yield Static(content, classes="message-content")

        # Show extracted code if present
        if self.entry.code:
            yield Static("[bold yellow]Extracted Code:[/]")
            syntax = Syntax(self.entry.code, "python", theme="monokai", line_numbers=True)
            yield Static(syntax, classes="code-block")
            yield Button(f"Use This Code", id=f"use-code-{self.message_id}", classes="use-code-btn")

        # Show token info for assistant messages
        if self.entry.tokens:
            prompt, completion = self.entry.tokens
            yield Static(f"[dim]Tokens: {prompt} prompt, {completion} completion[/]", classes="token-info")


class ExecutionResult(Static):
    """Widget showing execution results."""

    def __init__(
        self,
        results: list[dict],
        expected: list[list[list[int]]] | None = None,
        colored: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.results = results
        self.expected = expected
        self.colored = colored

    def compose(self) -> ComposeResult:
        yield Static("[bold]Execution Results[/]", classes="results-header")

        for i, result in enumerate(self.results):
            status = "[green]PASS[/]" if result.get("success") else "[red]FAIL[/]"
            yield Static(f"Example {i + 1}: {status}")

            if result.get("error"):
                yield Static(f"[red]Error: {result['error']}[/]", classes="error-msg")
            elif result.get("output"):
                yield Static("Computed output:", classes="output-label")
                yield GridDisplay(result["output"], "", self.colored, classes="result-grid")

                # Show expected if available and different
                if self.expected and i < len(self.expected):
                    if result["output"] != self.expected[i]:
                        yield Static("Expected output:", classes="output-label")
                        yield GridDisplay(self.expected[i], "", self.colored, classes="result-grid")


class ARCTUIApp(App):
    """Main TUI application for ARC-AGI interactive solving."""

    CSS = """
    /* Main layout */
    #main-container {
        layout: horizontal;
        height: 100%;
    }

    #left-panel {
        width: 35%;
        border-right: solid $surface-lighten-1;
        padding: 0 1;
    }

    #right-panel {
        width: 65%;
        padding: 0 1;
    }

    /* Challenge controls - compact */
    #challenge-controls {
        height: auto;
        padding: 0;
        margin-bottom: 1;
    }

    #challenge-controls Label {
        height: 1;
        padding: 0;
        margin: 0;
    }

    #challenge-id-input {
        width: 100%;
        height: 1;
        margin: 0 0 1 0;
    }

    #control-buttons {
        height: auto;
        margin-bottom: 1;
    }

    /* Compact buttons */
    Button {
        min-width: 6;
        height: 3;
        margin: 0;
    }

    .control-btn {
        margin-right: 1;
    }

    /* Training examples - maximize space */
    #examples-container {
        height: 1fr;
        padding: 0;
        scrollbar-size: 1 1;
    }

    .example-grids {
        height: auto;
        margin-bottom: 1;
    }

    .arrow {
        width: auto;
        padding-top: 1;
        content-align: center middle;
    }

    /* Chat area - maximize space */
    #chat-container {
        height: 1fr;
    }

    #chat-history {
        height: 1fr;
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin-bottom: 1;
        scrollbar-size: 1 1;
    }

    .chat-user {
        background: $surface;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick $primary;
    }

    .chat-assistant {
        background: $surface-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick $success;
    }

    .message-content {
        margin: 0;
        padding: 0;
    }

    .code-block {
        margin: 0;
        padding: 0;
        background: $surface-darken-2;
    }

    .use-code-btn {
        margin: 1 0;
    }

    .token-info {
        margin: 0;
        height: 1;
    }

    /* Chat input - compact */
    #chat-input-area {
        height: auto;
    }

    #chat-input {
        height: 3;
        margin-bottom: 1;
    }

    #action-buttons {
        height: auto;
    }

    .action-btn {
        margin-right: 1;
    }

    /* Results - compact, hidden when empty */
    #results-container {
        height: auto;
        max-height: 25%;
        border: solid $warning;
        padding: 0 1;
        margin-bottom: 1;
        scrollbar-size: 1 1;
    }

    #results-container:empty {
        display: none;
    }

    .results-header {
        margin: 0;
    }

    .error-msg {
        padding: 0;
    }

    .output-label {
        margin: 0;
    }

    .result-grid {
        margin-left: 1;
    }

    /* Status bar - minimal */
    #status-bar {
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
    }

    /* Compact header/footer */
    Header {
        height: 1;
    }

    Footer {
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+enter", "send_message", "Send"),
        Binding("r", "run_code", "Run Code"),
        Binding("t", "toggle_display", "Toggle Display"),
    ]

    # Reactive state
    challenge_id: reactive[str] = reactive("")
    colored_display: reactive[bool] = reactive(True)
    is_loading: reactive[bool] = reactive(False)

    def __init__(self):
        super().__init__()
        self.challenges: dict = {}
        self.solutions: dict = {}
        self.current_challenge: dict | None = None
        self.chat_history: list[ChatEntry] = []
        self.active_code: str | None = None
        self.execution_results: list[dict] | None = None
        self.config = CONFIG_LIST[0]

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-container"):
            # Left panel - Challenge display
            with Vertical(id="left-panel"):
                with Vertical(id="challenge-controls"):
                    yield Input(placeholder="Challenge ID", id="challenge-id-input")
                    with Horizontal(id="control-buttons"):
                        yield Button("Load", id="load-btn", classes="control-btn", variant="primary")
                        yield Button("[t]oggle", id="toggle-btn", classes="control-btn")

                yield ScrollableContainer(id="examples-container")

            # Right panel - Chat and execution
            with Vertical(id="right-panel"):
                with Vertical(id="chat-container"):
                    yield ScrollableContainer(id="chat-history")
                    yield ScrollableContainer(id="results-container")

                    with Vertical(id="chat-input-area"):
                        yield TextArea(id="chat-input")
                        with Horizontal(id="action-buttons"):
                            yield Button("Send", id="send-btn", classes="action-btn", variant="primary")
                            yield Button("[r]un", id="run-btn", classes="action-btn", variant="success")

                yield Static("Ready", id="status-bar")

        yield Footer()

    async def on_mount(self) -> None:
        """Load challenges on startup."""
        self.update_status("Loading challenges...")

        # Load challenges
        challenges_path = Path(CHALLENGES_PATH)
        if challenges_path.exists():
            with open(challenges_path) as f:
                self.challenges = json.load(f)
            self.update_status(f"Loaded {len(self.challenges)} challenges")
        else:
            self.update_status(f"[red]Challenges file not found: {CHALLENGES_PATH}[/]")

        # Load solutions if available
        solutions_path = Path(SOLUTIONS_PATH)
        if solutions_path.exists():
            with open(solutions_path) as f:
                self.solutions = json.load(f)

    def update_status(self, message: str) -> None:
        """Update the status bar."""
        status = self.query_one("#status-bar", Static)
        status.update(message)

    @on(Button.Pressed, "#load-btn")
    async def load_challenge(self) -> None:
        """Load the specified challenge."""
        input_widget = self.query_one("#challenge-id-input", Input)
        challenge_id = input_widget.value.strip()

        if not challenge_id:
            self.update_status("[red]Please enter a challenge ID[/]")
            return

        if challenge_id not in self.challenges:
            self.update_status(f"[red]Challenge '{challenge_id}' not found[/]")
            return

        self.challenge_id = challenge_id
        self.current_challenge = self.challenges[challenge_id]
        self.chat_history = []
        self.active_code = None
        self.execution_results = None

        # Clear and update examples display
        await self.refresh_examples()
        await self.refresh_chat()
        await self.refresh_results()

        self.update_status(f"Loaded challenge: {challenge_id}")

    async def refresh_examples(self) -> None:
        """Refresh the training examples display."""
        container = self.query_one("#examples-container", ScrollableContainer)
        await container.remove_children()

        if not self.current_challenge:
            return

        # Add training examples
        await container.mount(Static("[bold]Training Examples[/]"))
        for i, example in enumerate(self.current_challenge["train"], 1):
            widget = TrainingExample(
                i,
                example["input"],
                example["output"],
                self.colored_display
            )
            await container.mount(widget)

        # Add test inputs
        await container.mount(Static("\n[bold]Test Inputs[/]"))
        for i, test in enumerate(self.current_challenge["test"], 1):
            await container.mount(Static(f"[bold]Test {i}[/]"))
            await container.mount(GridDisplay(test["input"], "Input", self.colored_display))

    async def refresh_chat(self) -> None:
        """Refresh the chat history display."""
        container = self.query_one("#chat-history", ScrollableContainer)
        await container.remove_children()

        for i, entry in enumerate(self.chat_history):
            widget = ChatMessage(entry, i)
            await container.mount(widget)

        # Scroll to bottom
        container.scroll_end()

    async def refresh_results(self) -> None:
        """Refresh the execution results display."""
        container = self.query_one("#results-container", ScrollableContainer)
        await container.remove_children()

        if self.execution_results:
            expected = None
            if self.current_challenge:
                expected = [ex["output"] for ex in self.current_challenge["train"]]
            widget = ExecutionResult(self.execution_results, expected, self.colored_display)
            await container.mount(widget)

    @on(Button.Pressed, "#toggle-btn")
    def action_toggle_display(self) -> None:
        """Toggle between colored and numeric display."""
        self.colored_display = not self.colored_display
        asyncio.create_task(self.refresh_examples())
        asyncio.create_task(self.refresh_results())
        mode = "colored" if self.colored_display else "numeric"
        self.update_status(f"Display mode: {mode}")

    @on(Button.Pressed, "#send-btn")
    async def action_send_message(self) -> None:
        """Send message to LLM."""
        if not self.current_challenge:
            self.update_status("[red]Please load a challenge first[/]")
            return

        input_widget = self.query_one("#chat-input", TextArea)
        user_message = input_widget.text.strip()

        if not user_message:
            self.update_status("[red]Please enter a message[/]")
            return

        # Add user message to history
        user_entry = ChatEntry(role="user", content=user_message)
        self.chat_history.append(user_entry)
        await self.refresh_chat()

        # Clear input
        input_widget.clear()

        # Send to LLM
        self.is_loading = True
        self.update_status("Sending to LLM...")
        self.call_llm(user_message)

    @work(exclusive=True)
    async def call_llm(self, user_message: str) -> None:
        """Call LLM in background worker."""
        try:
            # Build the full prompt
            challenge_text = format_challenge_for_prompt(self.current_challenge)

            full_prompt = f"""You are an expert at solving ARC-AGI puzzles. Analyze the patterns in the training examples and write a Python transform function.

{challenge_text}

User request: {user_message}

Write a Python function called `transform` that takes a numpy array as input and returns a numpy array as output. The function should implement the pattern you observe in the training examples.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    # Your implementation here
    pass
```"""

            # Call LLM
            response, duration, _, _, prompt_tokens, completion_tokens = await llm(
                model=self.config["llm_id"],
                message=full_prompt,
                temperature=self.config["solver_temperature"],
                request_timeout=self.config.get("request_timeout"),
                max_remaining_time=None,
                max_remaining_timeouts=None,
                problem_id=self.challenge_id,
                retries=3,
            )

            # Extract code from response
            code = extract_code_from_response(response)
            if code:
                self.active_code = code

            # Add assistant response to history
            assistant_entry = ChatEntry(
                role="assistant",
                content=response,
                code=code,
                tokens=(prompt_tokens, completion_tokens)
            )
            self.chat_history.append(assistant_entry)

            self.update_status(f"Response received ({duration:.1f}s, {prompt_tokens + completion_tokens} tokens)")

        except Exception as e:
            self.update_status(f"[red]LLM Error: {str(e)}[/]")

            # Add error message to chat
            error_entry = ChatEntry(
                role="assistant",
                content=f"Error calling LLM: {str(e)}"
            )
            self.chat_history.append(error_entry)

        finally:
            self.is_loading = False
            await self.refresh_chat()

    @on(Button.Pressed, "#run-btn")
    async def action_run_code(self) -> None:
        """Run the active code against training examples."""
        if not self.current_challenge:
            self.update_status("[red]Please load a challenge first[/]")
            return

        if not self.active_code:
            self.update_status("[red]No code to run. Send a message to generate code first.[/]")
            return

        self.update_status("Running code...")
        self.run_code_worker()

    @work(exclusive=True)
    async def run_code_worker(self) -> None:
        """Execute code in background worker."""
        try:
            results = []

            # Run against training examples
            for i, example in enumerate(self.current_challenge["train"]):
                ok, output_str = await sandbox_run(
                    self.active_code,
                    example["input"],
                    timeout_s=5.0
                )

                result = {"success": False, "output": None, "error": None}

                if not ok:
                    result["error"] = output_str or "Execution failed"
                else:
                    try:
                        output = json.loads(output_str)
                        result["output"] = output
                        result["success"] = (output == example["output"])
                    except json.JSONDecodeError:
                        result["error"] = f"Invalid output: {output_str}"

                results.append(result)

            self.execution_results = results

            # Count passes
            passes = sum(1 for r in results if r["success"])
            total = len(results)

            if passes == total:
                self.update_status(f"[green]All {total} training examples passed![/]")
            else:
                self.update_status(f"[yellow]{passes}/{total} training examples passed[/]")

        except Exception as e:
            self.update_status(f"[red]Execution error: {str(e)}[/]")

        finally:
            await self.refresh_results()

    @on(Button.Pressed)
    async def on_use_code_button(self, event: Button.Pressed) -> None:
        """Handle 'Use This Code' button presses."""
        button_id = event.button.id
        if button_id and button_id.startswith("use-code-"):
            try:
                message_id = int(button_id.replace("use-code-", ""))
                if 0 <= message_id < len(self.chat_history):
                    entry = self.chat_history[message_id]
                    if entry.code:
                        self.active_code = entry.code
                        self.update_status(f"Selected code from message {message_id + 1}")
            except ValueError:
                pass


def main():
    """Run the TUI application."""
    app = ARCTUIApp()
    app.run()


if __name__ == "__main__":
    main()
