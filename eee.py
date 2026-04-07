"""
eee.py — EEE Model ("Triple-E"): Hybrid LLM + Bayesian agent for causal
discovery in unknown grid-based puzzle games.

The agent learns causal rules purely through interaction, without being told
the rules, objectives, or any game-specific knowledge in advance.

Component order:
  1. Imports and shared type definitions
  2. StateEncoder
  3. CausalHistory
  4. BayesianCausalModel
  5. DFAController
  6. LLMReasoningEngine
  7. Logger (standalone functions)
  8. run_agent()
"""

# SECTION       #? IMPORTS & SHARED TYPE DEFINITIONS
#___________________________________________________________________________

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import TypedDict
import anthropic
import re

# TYPE ALIASES
# Raw game state is received from the frontend JavaScript environment -- keys that are stripped by StateEncoder
# but must never be forwarded to the LLM
class GameState(TypedDict, total=False):
    grid: list[list[str]]          # 2D array of object letters / empty strings
    playerPosition: dict           # {"row": int, "col": int}
    avatar_color: str              # visually observable
    legend: dict                   # STRIP — reveals object roles
    directionCapability: list      # STRIP — reveals movement constraints
    currentGameState: str          # STRIP — internal A/B flag
    timestamp: float

# Encodes game-turn for LLM into shape-tokens
class EncodedState(TypedDict):
    grid_text: str              # multi-line grid rendered with shape tokens
    action: str                 # key pressed (e.g. "ArrowDown")
    delta: tuple[int, int]      # movement vector (row_delta, col_delta)
    timestamp: float
    avatar_color: str | None
    avatar_position: tuple[int, int] | None  # (row, col) after action

# Parses sections from one LLM reasoning trace
class LLMResponse(TypedDict):
    observations: str
    hypothesis_update: str
    structural_check: str
    predicted_outcome: str
    mode_assessment: str

_SHAPE_TOKENS = ["◆", "●", "■", "▲", "✦", "⬟"]
# Shape token vocabulary for non-avatar objects, assigned in discovery order

AVATAR_TOKEN = "\u263a\ufe0e"  # ☺︎
# Avatar token is a two-character sequence: U+263A followed by U+FE0E (VS-15).
# Do NOT treat this as a single character. Always compare full strings.


# SECTION       #? STATE ENCODER
#___________________________________________________________________________
# Filters and encodes raw JavaScript game state for LLM input

class StateEncoder:
    """
    Responsibilities:
      - Remap meaningful letter symbols to abstract shape tokens so the LLM
        cannot use prior world knowledge to infer object roles.
      - Strip non-visual fields (legend, directionCapability, currentGameState).
      - Render the grid as plain text.
      - Compute a positional diff between consecutive grid states.
      - Compute the prediction residual by comparing actual diff to the LLM's
        most recent PREDICTED OUTCOME.

    Symbol-remapping rules:
      - The avatar always maps to AVATAR_TOKEN (☺︎). Fixed.
      - Other object types are assigned shape tokens from _SHAPE_TOKENS in the
        order they are first observed across the game session.
      - Empty cells are rendered as a single space.
      - Remapping is consistent within a game; reset between games.
    """

    # Initializes encoder with empty symbol map (resets per game)
    def __init__(self) -> None:
        # Maps raw letter symbol → assigned shape token string
        self._symbol_map: dict[str, str] = {}
        self._next_token_idx: int = 0
        # The most recent grid (as list[list[str]]) after remapping, for diff
        self._prev_remapped_grid: list[list[str]] | None = None

    def reset(self) -> None:
        """Reset all state. Call between games to rediscover object roles."""
        self._symbol_map = {}
        self._next_token_idx = 0
        self._prev_remapped_grid = None

    # Symbol remapping
    def _get_or_assign_token(self, raw_symbol: str) -> str:
        
        # Returns the shape token for raw_symbol, assigning one if needed.
        # The avatar symbol is detected by checking for a known sentinel value and
        # passed by the frontend (e.g. "P" or "@"). Callers that know the avatar
        # symbol should pass it here; it will always return AVATAR_TOKEN.
        
        if raw_symbol in self._symbol_map:
            return self._symbol_map[raw_symbol]

        if self._next_token_idx >= len(_SHAPE_TOKENS):
            # Vocabulary exhausted — fall back to a numbered marker.
            token = f"[{self._next_token_idx}]"
        else:
            token = _SHAPE_TOKENS[self._next_token_idx]
        self._next_token_idx += 1
        self._symbol_map[raw_symbol] = token
        return token

    # Converts one raw grid cell value to its shape token (or AVATAR_TOKEN)
    def _remap_cell(self, raw: str, avatar_symbol: str) -> str:
        """
        Args:
            raw:           Raw value from the JavaScript grid (e.g. "P", "G").
            avatar_symbol: The raw symbol that represents the avatar in this game.

        Returns:
            The shape token string for this cell, or " " if the cell is empty.
        """
        if raw == "" or raw is None:
            return " "
        if raw == avatar_symbol:
            return AVATAR_TOKEN
        return self._get_or_assign_token(raw)

    def _remap_grid(
        self, raw_grid: list[list[str]], avatar_symbol: str
    ) -> list[list[str]]:
        """Return a new 2D grid with all symbols remapped to shape tokens."""
        return [
            [self._remap_cell(cell, avatar_symbol) for cell in row]
            for row in raw_grid
        ]

    # Grid rendering
    def _render_grid(self, remapped_grid: list[list[str]]) -> str:
        """
        Render a remapped grid as a plain-text string.

        Cells are separated by single spaces. The avatar token (☺︎) is a
        two-character sequence and is treated as a unit — do not split it.
        """
        lines: list[str] = []
        for row in remapped_grid:
            # Join tokens with a space separator; each token is a full string.
            lines.append(" ".join(token for token in row))
        return "\n".join(lines)

    # Positional delta
    def _compute_diff(
        self,
        prev: list[list[str]] | None,
        curr: list[list[str]],
    ) -> list[str]:
        """
        Compute a positional diff between two remapped grids.

        Uses raw, non-interpretive language only. Detects:
          - Tokens that appeared (were empty before, non-empty now)
          - Tokens that disappeared (were non-empty before, empty now)
          - Tokens that moved (same token, different position)
          - Cells that became empty

        Args:
            prev: Previous remapped grid, or None if first turn.
            curr: Current remapped grid.

        Returns:
            List of diff strings in non-interpretive language.
        """
        if prev is None:
            return ["(No previous state — first turn)"]

        diff: list[str] = []
        rows = len(curr)
        cols = len(curr[0]) if rows > 0 else 0

        # Build position maps: token → set of (row, col)
        def position_map(grid: list[list[str]]) -> dict[str, set[tuple[int, int]]]:
            pos: dict[str, set[tuple[int, int]]] = {}
            for r, row in enumerate(grid):
                for c, token in enumerate(row):
                    if token != " ":
                        pos.setdefault(token, set()).add((r, c))
            return pos

        prev_map = position_map(prev)
        curr_map = position_map(curr)

        all_tokens = set(prev_map) | set(curr_map)

        for token in sorted(all_tokens):
            prev_positions = prev_map.get(token, set())
            curr_positions = curr_map.get(token, set())

            added = curr_positions - prev_positions
            removed = prev_positions - curr_positions

            # Pair up moves greedily (one-to-one)
            moves: list[tuple[tuple[int, int], tuple[int, int]]] = []
            rem_added = list(added)
            rem_removed = list(removed)
            while rem_added and rem_removed:
                src = rem_removed.pop()
                dst = rem_added.pop()
                moves.append((src, dst))

            for src, dst in moves:
                diff.append(
                    f"Token {token} moved from {src} to {dst}"
                )
            for pos in rem_added:
                diff.append(f"Token {token} appeared at {pos}")
            for pos in rem_removed:
                diff.append(f"Token {token} disappeared from {pos}")

        # Detect cells that became empty (non-empty → empty)
        for r in range(rows):
            for c in range(cols):
                prev_cell = prev[r][c] if prev else " "
                curr_cell = curr[r][c]
                if prev_cell != " " and curr_cell == " ":
                    # Already captured above via token disappearance;
                    # also emit the cell-level description.
                    diff.append(f"Cell ({r},{c}) is now empty")

        if not diff:
            diff.append("No positional changes detected")

        return diff

    # Avatar detection helper
    def _detect_avatar_symbol(
        self, raw_grid: list[list[str]], player_pos: dict | None
    ) -> str:
        """
        Determine the raw symbol used for the avatar in this game.

        Looks up the raw grid cell at the player's reported position. Falls back
        to scanning the grid for a symbol that moves in response to actions
        (handled externally). Returns empty string if undetermined.
        """
        if player_pos is None:
            return ""
        row = player_pos.get("row", player_pos.get("y", -1))
        col = player_pos.get("col", player_pos.get("x", -1))
        if row < 0 or col < 0:
            return ""
        try:
            return raw_grid[row][col]
        except (IndexError, TypeError):
            return ""

    # PUBLIC API
    def encode(
        self,
        game_state: GameState,
        prev_game_state: GameState | None,
        action: str,
        delta: tuple[int, int],
        timestamp: float,
    ) -> EncodedState:
        """
        Encode a raw game state into a structured LLM-ready form.

        Strips legend, directionCapability, and currentGameState. Remaps grid
        symbols to shape tokens. Computes positional diff vs. previous state.

        Args:
            game_state:      Current raw game state from JavaScript.
            prev_game_state: Previous raw game state, or None on first turn.
            action:          Key pressed (e.g. "ArrowDown").
            delta:           Movement vector as (row_delta, col_delta).
            timestamp:       Turn timestamp.

        Returns:
            EncodedState ready for LLM consumption.
        """
        raw_grid: list[list[str]] = game_state.get("grid", [])
        player_pos: dict | None = game_state.get("playerPosition")
        avatar_color: str | None = game_state.get("avatar_color")

        avatar_symbol = self._detect_avatar_symbol(raw_grid, player_pos)

        remapped_grid = self._remap_grid(raw_grid, avatar_symbol)
        grid_text = self._render_grid(remapped_grid)

        prev_remapped: list[list[str]] | None = None
        if prev_game_state is not None:
            prev_raw: list[list[str]] = prev_game_state.get("grid", [])
            prev_avatar_symbol = self._detect_avatar_symbol(
                prev_raw, prev_game_state.get("playerPosition")
            )
            prev_remapped = self._remap_grid(prev_raw, prev_avatar_symbol)

        diff = self._compute_diff(prev_remapped, remapped_grid)

        # Determine avatar position in remapped grid
        avatar_position: tuple[int, int] | None = None
        if player_pos is not None:
            r = player_pos.get("row", player_pos.get("y", -1))
            c = player_pos.get("col", player_pos.get("x", -1))
            if r >= 0 and c >= 0:
                avatar_position = (r, c)

        # Store current remapped grid for next turn's diff
        self._prev_remapped_grid = remapped_grid

        return EncodedState(
            grid_text=grid_text,
            action=action,
            delta=delta,
            timestamp=timestamp,
            avatar_color=avatar_color,
            avatar_position=avatar_position,
        )

    @staticmethod
    def _diff_to_direction(diff_item: str) -> str | None:
        """
        Convert a coordinate-based diff string into a direction word.

        E.g. "Token ☺︎ moved from (5,5) to (4,5)" → "up"

        Returns None if no movement coordinates are found.
        """
        match = re.search(r"moved from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)", diff_item)
        if not match:
            return None
        r1, c1, r2, c2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        dr = r2 - r1
        dc = c2 - c1
        if dr < 0:
            return "up"
        if dr > 0:
            return "down"
        if dc < 0:
            return "left"
        if dc > 0:
            return "right"
        return None

    def compute_residual(
        self,
        actual_diff: list[str],
        predicted_diff: str,
    ) -> bool:
        """
        Determine whether the actual outcome differs from the LLM's prediction.

        Compares the actual diff list (from StateEncoder) against the predicted
        outcome string (from the LLM's PREDICTED OUTCOME section). A residual
        of True means the outcome was surprising — a reasoning trace should fire.

        Converts coordinate diffs to direction words before matching, so that
        "moved from (5,5) to (4,5)" correctly matches a prediction of "up".

        Args:
            actual_diff:    List of diff strings from the current turn.
            predicted_diff: The LLM's free-text prediction from the prior turn.

        Returns:
            True if residual != 0 (trace should fire), False otherwise.
        """
        if not predicted_diff:
            return True
        if actual_diff == ["No positional changes detected"]:
            return "no change" not in predicted_diff.lower()

        predicted_lower = predicted_diff.lower()
        for item in actual_diff:
            if item == "(No previous state — first turn)":
                continue
            # Try direction-word matching first
            direction = self._diff_to_direction(item)
            if direction is not None:
                if direction not in predicted_lower:
                    return True
                continue
            # Fall back to loose string matching
            item_clean = item.lower().replace(AVATAR_TOKEN, "avatar")
            if item_clean not in predicted_lower:
                return True
        return False


# SECTION       #? CAUSAL HISTORY MODULE
#___________________________________________________________________________

_TRACKS = ("mechanics", "objective")
_CATEGORIES = ("confirmed", "hypothesized", "disconfirmed", "uncertain")

class CausalHistory:
    """
    Structured belief log across turns.

    Maintains hypotheses in four categories (confirmed, hypothesized,
    disconfirmed, uncertain) for two learning tracks (mechanics, objective).
    Formats the full history as a prompt-ready string for LLM input.

    Token promotions: once a shape token's role is confirmed through the
    discovery process, it can be promoted to a labeled form (e.g. "GOAL (●)")
    for use in reasoning traces.
    """

    def __init__(self) -> None:
        """Initialize empty history for both tracks."""
        # Structure: {track: {category: [hypothesis_str, ...]}}
        self._history: dict[str, dict[str, list[str]]] = {
            track: {cat: [] for cat in _CATEGORIES}
            for track in _TRACKS
        }
        # Token promotion map: shape_token → "LABEL (token)"
        self._promotions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_token(self, shape_token: str, confirmed_label: str) -> None:
        """
        Mark a shape token as having a confirmed role label.

        After promotion, subsequent references to this token in to_prompt_string
        will include the label (e.g. "GOAL (●)"). The shape token is preserved
        for traceability.

        Promotion should only happen after the discovery process has confirmed
        the role — not through pattern recognition of raw letter names.

        Args:
            shape_token:     The abstract shape token (e.g. "●").
            confirmed_label: Human-readable role name (e.g. "GOAL").
        """
        self._promotions[shape_token] = f"{confirmed_label} ({shape_token})"

    def _apply_promotions(self, hypothesis: str) -> str:
        """Replace any promoted shape tokens in a hypothesis string with their labels."""
        for token, label in self._promotions.items():
            hypothesis = hypothesis.replace(token, label)
        return hypothesis

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------

    def add_hypothesis(
        self,
        hypothesis: str,
        track: str,
        category: str = "uncertain",
    ) -> None:
        """
        Add a new hypothesis to the specified track and category.

        If the hypothesis already exists in any category on this track,
        this is a no-op (use update_category to move it).

        Args:
            hypothesis: Free-text description of the hypothesis.
            track:      "mechanics" or "objective".
            category:   Initial category; defaults to "uncertain".
        """
        if track not in _TRACKS:
            raise ValueError(f"Unknown track: {track!r}. Must be one of {_TRACKS}.")
        if category not in _CATEGORIES:
            raise ValueError(f"Unknown category: {category!r}.")
        # Check for duplicates across all categories on this track
        for cat in _CATEGORIES:
            if hypothesis in self._history[track][cat]:
                return
        self._history[track][category].append(hypothesis)

    def update_category(self, hypothesis: str, new_category: str) -> None:
        """
        Move a hypothesis to a different category (on whichever track it lives).

        Searches both tracks. If the hypothesis is not found, this is a no-op.

        Args:
            hypothesis:   The hypothesis string to find and move.
            new_category: The target category.
        """
        if new_category not in _CATEGORIES:
            raise ValueError(f"Unknown category: {new_category!r}.")
        for track in _TRACKS:
            for cat in _CATEGORIES:
                lst = self._history[track][cat]
                if hypothesis in lst:
                    lst.remove(hypothesis)
                    self._history[track][new_category].append(hypothesis)
                    return

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_prompt_string(self) -> str:
        """
        Format the full hypothesis history as a prompt-ready string.

        Returns a multi-section string with both tracks and all four categories,
        applying any token promotions to hypothesis text. Each hypothesis is
        assigned a unique ID (e.g. [H1], [H2]) so the LLM can reference them
        by ID in its updates rather than by rephrasing the text.
        """
        lines: list[str] = []

        section_labels = {
            "confirmed": "Confirmed:",
            "hypothesized": "Active hypotheses:",
            "disconfirmed": "Disconfirmed:",
            "uncertain": "Uncertain (untested):",
        }

        # Assign a stable ID to every hypothesis across all tracks/categories
        self._hypothesis_ids: dict[str, str] = {}
        counter = 1
        for track in _TRACKS:
            for cat in _CATEGORIES:
                for item in self._history[track][cat]:
                    if item not in self._hypothesis_ids:
                        self._hypothesis_ids[item] = f"[H{counter}]"
                        counter += 1

        for track in _TRACKS:
            lines.append(f"{track.upper()} TRACK")
            lines.append("-" * 15)
            for cat in _CATEGORIES:
                lines.append(section_labels[cat])
                items = self._history[track][cat]
                if items:
                    for item in items:
                        h_id = self._hypothesis_ids.get(item, "")
                        lines.append(f"- {h_id} {self._apply_promotions(item)}")
                else:
                    lines.append("- (none)")
                lines.append("")
            lines.append("")

        return "\n".join(lines).rstrip()

    def to_dict(self) -> dict:
        """
        Serialise the full history to a plain dict (for logger and BayesianCausalModel).

        Returns a nested dict: {track: {category: [hypothesis_str, ...]}}.
        """
        return {
            track: {cat: list(lst) for cat, lst in cats.items()}
            for track, cats in self._history.items()
        }


# SECTION       #? BAYESIEN CAUSAL MODEL
#___________________________________________________________________________

_MECHANICS_THRESHOLD = 3 # Observation thresholds before Beta-Binomial updating begins
_OBJECTIVE_THRESHOLD = 1

class BayesianCausalModel:
    """
    Beta-Binomial posterior distributions over all causal hypotheses.

    Implemented directly without probabilistic programming libraries.
    Each hypothesis starts at Beta(1,1) — a flat prior (mean = 0.5).

    Mechanics track:   3 raw observations before a hypothesis is formally
                       created and updating begins. Errs on the side of caution
                       because mechanics errors propagate downstream.

    Objective track:   1 observation before a hypothesis is formed (hot start /
                       simulated annealing logic). Information gain from early
                       hypotheses outweighs the risk of a wrong one.

    Posterior mean     = α / (α + β)
    Posterior variance = (α * β) / ((α + β)² * (α + β + 1))
    """

    def __init__(self) -> None:
        """Initialise with empty hypothesis sets."""
        # {hypothesis: {"alpha": float, "beta": float, "track": str}}
        self._posteriors: dict[str, dict] = {}
        # Accumulation buffer before threshold is reached:
        # {hypothesis: {"track": str, "consistent": int, "inconsistent": int}}
        self._raw_observations: dict[str, dict] = {}

    def add_hypothesis(self, hypothesis: str, track: str) -> None:
        """
        Register a new hypothesis at Beta(1,1).

        If the hypothesis already exists, this is a no-op.
        Args:
            hypothesis: Free-text hypothesis string.
            track:      "mechanics" or "objective".
        """
        if hypothesis in self._posteriors:
            return
        self._posteriors[hypothesis] = {"alpha": 1.0, "beta": 1.0, "track": track}

    def record_observation(
        self,
        hypothesis: str,
        track: str,
        consistent: bool,
    ) -> float | None:
        """
        Record one observation for a hypothesis, respecting track thresholds.

        For the mechanics track, accumulates raw observations until 3 have been
        seen, then registers the hypothesis and begins Beta-Binomial updating.
        For the objective track, begins immediately after 1 observation.

        Args:
            hypothesis: The hypothesis being observed.
            track:      "mechanics" or "objective".
            consistent: True if the observation supports the hypothesis.

        Returns:
            Current posterior mean once updating has begun, else None.
        """
        threshold = _MECHANICS_THRESHOLD if track == "mechanics" else _OBJECTIVE_THRESHOLD

        # Initialise raw accumulation buffer if needed
        if hypothesis not in self._raw_observations:
            self._raw_observations[hypothesis] = {
                "track": track,
                "consistent": 0,
                "inconsistent": 0,
                "total": 0,
            }

        buf = self._raw_observations[hypothesis]
        if consistent:
            buf["consistent"] += 1
        else:
            buf["inconsistent"] += 1
        buf["total"] += 1

        total = buf["total"]

        if total < threshold:
            # Still accumulating — hypothesis not yet formed
            return None

        # Threshold reached — ensure hypothesis exists in posteriors
        if hypothesis not in self._posteriors:
            self.add_hypothesis(hypothesis, track)
            # Replay accumulated observations into the Beta distribution
            # (subtract the initial +1 counts from the flat prior)
            h = self._posteriors[hypothesis]
            h["alpha"] += buf["consistent"] - 1   # -1 because prior already has 1
            h["beta"] += buf["inconsistent"] - 1  # -1 because prior already has 1
            # Clamp to minimum of 1 to keep Beta valid
            h["alpha"] = max(1.0, h["alpha"])
            h["beta"] = max(1.0, h["beta"])
        else:
            # Already registered — just do an incremental update for this observation
            self.update(hypothesis, consistent)

        return self.get_posterior(hypothesis)

    def update(self, hypothesis: str, consistent: bool) -> float:
        """
        Apply one Beta-Binomial update to an already-registered hypothesis.

        Args:
            hypothesis: Must already exist in self._posteriors.
            consistent: True → α += 1; False → β += 1.

        Returns:
            New posterior mean.
        """
        if hypothesis not in self._posteriors:
            raise KeyError(f"Hypothesis not registered: {hypothesis!r}")
        h = self._posteriors[hypothesis]
        if consistent:
            h["alpha"] += 1.0
        else:
            h["beta"] += 1.0
        return self.get_posterior(hypothesis)

    def get_posterior(self, hypothesis: str) -> float:
        """
        Return the posterior mean P(hypothesis is true) = α / (α + β).

        Args:
            hypothesis: Must already exist in self._posteriors.

        Returns:
            Posterior mean in [0, 1].
        """
        h = self._posteriors[hypothesis]
        return h["alpha"] / (h["alpha"] + h["beta"])

    def get_uncertainty(self, hypothesis: str) -> float:
        """
        Return the posterior variance — a proxy for epistemic uncertainty.

        Higher variance → higher uncertainty → prioritise for testing.

        Formula: (α * β) / ((α + β)² * (α + β + 1))

        Args:
            hypothesis: Must already exist in self._posteriors.

        Returns:
            Posterior variance in [0, 0.25].
        """
        h = self._posteriors[hypothesis]
        a, b = h["alpha"], h["beta"]
        s = a + b
        if s == 0:
            return 0.25  # Maximum uncertainty — treat as uninformative prior
        return (a * b) / (s * s * (s + 1))

    def get_most_uncertain(
        self, n: int = 3, track: str | None = None
    ) -> list[str]:
        """
        Return the top n hypotheses ranked by posterior variance (descending).
        Uses a list snapshot to prevent RuntimeError if the dictionary 
        is modified during iteration.

        Args:
            n:      Number of hypotheses to return.
            track:  If given, filter to only hypotheses on this track.

        Returns:
            List of hypothesis strings, most uncertain first.
        """
        # Create a list snapshot of the dictionary items
        # This prevents "RuntimeError: dictionary changed size during iteration"
        items_snapshot = list(self._posteriors.items())

        candidates = [
            h for h, v in items_snapshot
            if track is None or v["track"] == track
        ]

        # Sort based on the calculated variance (uncertainty)
        candidates.sort(key=lambda h: self.get_uncertainty(h), reverse=True)
        return candidates[:n]
    
    def get_all_posteriors(self, track: str | None = None) -> dict[str, float]:
        """
        Return a dict of {hypothesis: posterior_mean} for all known hypotheses.

        Args:
            track: If given, filter to only hypotheses on this track.

        Returns:
            Dict mapping hypothesis string → posterior mean.
        """
        return {
            h: self.get_posterior(h)
            for h, v in self._posteriors.items()
            if track is None or v["track"] == track
        }


# ===========================================================================
# 5. DFA CONTROLLER
# ===========================================================================

# DFA states
_DFA_STATES = ("EXPLORE", "ESTABLISH", "EXPLOIT")

# Exploration ordering (most → least exploratory) — used by get_global_mode()
_DFA_EXPLORE_ORDER = {"EXPLORE": 0, "ESTABLISH": 1, "EXPLOIT": 2}

# Confirmed posterior threshold
_CONFIRMED_THRESHOLD = 0.85

# Disconfirmed posterior threshold
_DISCONFIRMED_THRESHOLD = 0.15

# Number of EXPLOIT-mode actions before falling back to ESTABLISH
_EXPLOIT_STUCK_LIMIT = 10


# SECTION       #? DFA CONTROLLER
#___________________________________________________________________________

class DFAController:
    """
    Governs the agent's reasoning mode at the level of individual hypotheses.

    States: EXPLORE → ESTABLISH → EXPLOIT
    The agent can simultaneously hold different modes for different hypotheses.

    Transition rules:
      EXPLORE → ESTABLISH:
        mechanics track: >= 3 observations accumulated
        objective track: >= 1 observation accumulated

      ESTABLISH → EXPLOIT:
        local posterior > 0.85 (hypothesis confirmed)

      EXPLOIT → ESTABLISH (NOT back to EXPLORE):
        agent has taken > 10 actions in EXPLOIT on this hypothesis without
        progress. Hypothesis is revised, not discarded.

    See get_global_mode() for the aggregate mode used in prompt labelling.
    """

    def __init__(self) -> None:
        """Initialise with no per-hypothesis states."""
        # {hypothesis: "EXPLORE" | "ESTABLISH" | "EXPLOIT"}
        self._modes: dict[str, str] = {}
        # Counter of consecutive EXPLOIT actions without progress per hypothesis
        self._exploit_stuck: dict[str, int] = {}

    def get_mode(self, hypothesis: str) -> str:
        """
        Return the current DFA state for this hypothesis.

        Defaults to EXPLORE if the hypothesis has not been seen before.

        Args:
            hypothesis: The hypothesis string.

        Returns:
            One of "EXPLORE", "ESTABLISH", "EXPLOIT".
        """
        return self._modes.get(hypothesis, "EXPLORE")
    
    def select_action(
        self, 
        bayesian_model: BayesianCausalModel, 
        uncertain_targets: list[str]
    ) -> str:
        """
        DFA decision logic:
        1. If in EXPLORE/ESTABLISH: Move toward an 'uncertain_target'.
        2. If in EXPLOIT: Move toward the confirmed 'goal' token.
        """
        mode = self.get_global_mode()
        
        # Default fallback action
        available_actions = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
        
        if mode in ("EXPLORE", "ESTABLISH") and uncertain_targets:
            # Logic: In a real implementation, you'd calculate pathfinding 
            # to the grid coordinates of the uncertain_targets.
            # For now, we return a placeholder exploration move.
            return available_actions[int(time.time()) % 4]
            
        # If we are exploiting, we assume we found a path to the objective
        return "ArrowUp"

    def check_transitions(
        self,
        hypothesis: str,
        bayes_model: BayesianCausalModel,
        observation_count: int,
        stuck_count: int,
        track: str,
    ) -> str:
        """
        Evaluate and apply DFA transition conditions for one hypothesis.

        Updates the stored mode and returns the new mode.

        Args:
            hypothesis:        The hypothesis to evaluate.
            bayes_model:       The current Bayesian model (to read posteriors).
            observation_count: Total raw observations accumulated for this hypothesis.
            stuck_count:       Number of consecutive EXPLOIT steps without progress.
            track:             "mechanics" or "objective".

        Returns:
            New DFA mode string for this hypothesis.
        """
        if hypothesis not in self._modes:
            self._modes[hypothesis] = "EXPLORE"
        if hypothesis not in self._exploit_stuck:
            self._exploit_stuck[hypothesis] = 0

        current = self._modes[hypothesis]
        threshold = _MECHANICS_THRESHOLD if track == "mechanics" else _OBJECTIVE_THRESHOLD

        if current == "EXPLORE":
            if observation_count >= threshold:
                self._modes[hypothesis] = "ESTABLISH"

        elif current == "ESTABLISH":
            if hypothesis in bayes_model._posteriors:
                posterior = bayes_model.get_posterior(hypothesis)
                if posterior > _CONFIRMED_THRESHOLD:
                    self._modes[hypothesis] = "EXPLOIT"
                    self._exploit_stuck[hypothesis] = 0

        elif current == "EXPLOIT":
            self._exploit_stuck[hypothesis] += stuck_count
            if self._exploit_stuck[hypothesis] > _EXPLOIT_STUCK_LIMIT:
                # Fall back to ESTABLISH — revise, don't discard
                self._modes[hypothesis] = "ESTABLISH"
                self._exploit_stuck[hypothesis] = 0

        return self._modes[hypothesis]

    def get_global_mode(self) -> str:
        """
        Return the most exploratory DFA mode currently active across all hypotheses.

        EXPLORE > ESTABLISH > EXPLOIT. This is used for reasoning trace labels
        and prompt context headers.

        Returns:
            The most exploratory mode string, or "EXPLORE" if no hypotheses exist.
        """
        if not self._modes:
            return "EXPLORE"
        return min(self._modes.values(), key=lambda m: _DFA_EXPLORE_ORDER[m])


# Determine which step (1–7) the agent is in based on global DFA mode and
# whether mechanics are confirmed.  Step mapping:
#   1 — EXPLORE, no confirmed mechanics, no hypothesis yet (blank slate)
#   2 — EXPLORE, mechanics track has active hypotheses
#   3 — ESTABLISH, mechanics track
#   4 — EXPLOIT, mechanics track
#   5 — EXPLORE, objective track probing
#   6 — ESTABLISH, objective track
#   7 — EXPLOIT, objective track


# SECTION       #? LLM REASONING ENGINE
#___________________________________________________________________________

_USER_PROMPT_TEMPLATE = """\
CURRENT GAME STATE:
{grid_text}

Avatar position: {avatar_position}
Action taken: {action}
Movement delta: {delta}
Timestamp: {timestamp}
Avatar color: {avatar_color}

POSITIONAL DIFF:
{diff}

CAUSAL HISTORY:
{causal_history}

CURRENT MODE: {global_mode}

HIGHEST UNCERTAINTY HYPOTHESES (prioritize testing these):
{uncertain_hypotheses}
"""

_TRACE_LABEL_TEMPLATES = {
    "explore_env": (
        "Exploring environment — moving {direction} to probe unvisited cell at {position}"
    ),
    "explore_obj": (
        "Exploring objective — observed {event} — flagging as potentially "
        "objective-relevant — cannot be explained by known mechanics"
    ),
    "establish_mechanics": (
        "Testing mechanic hypothesis: {hypothesis} — observation {n} of 3 — "
        "{verdict} — posterior now {posterior:.3f}"
    ),
    "exploit_mechanics": (
        "Applying confirmed mechanic: {hypothesis} — moving strategically toward {target}"
    ),
    "establish_objective": (
        "Establishing possible objective: {hypothesis} — {verdict} — "
        "posterior now {posterior:.3f}"
    ),
    "exploit_objective": (
        "Pursuing confirmed objective: {hypothesis} — executing {action}"
    ),
}


class LLMReasoningEngine:
    """
    Constructs prompts, calls the Anthropic API, parses responses, and decides
    whether a reasoning trace fires each turn.

    A trace fires when: actual state change − what confirmed mechanics predict ≠ 0.
    When residual = 0 (outcome fully explained by confirmed mechanics), the turn
    is silent — no API call is made and run_turn() returns None.

    Objective-relevant observation filter (three qualifying categories):
      1. Terminal/significant state changes: cell disappears, counter changes,
         board resets.
      2. Asymmetries: some objects are more reactive than others in ways not
         explained by confirmed mechanics.
      3. Environment-induced recurrence: states that recur in ways not explained
         by any confirmed mechanic.
    Excluded: movement-induced recurrence where the avatar revisits a cell
    (fully explained by movement mechanics).

    Uses claude-sonnet-4-6 via the Anthropic Python SDK.
    Retries on API errors with exponential backoff (max 3 attempts).
    """

    def __init__(self) -> None:
        """Initialise the Anthropic client and reset prediction state."""
        self._client = anthropic.Anthropic()
        self._last_predicted_outcome: str = ""

    def should_fire_trace(self, residual: bool) -> bool:
        """
        Return True if a reasoning trace should fire this turn.

        A trace fires whenever residual != 0 — i.e. the actual outcome differs
        from what confirmed mechanics predicted.

        Args:
            residual: True if the actual diff differs from the prediction.

        Returns:
            True if a trace should be generated.
        """
        return residual

    def build_prompt(
        self,
        encoded_state: EncodedState,
        causal_history: CausalHistory,
        dfa_controller: DFAController,
        bayesian_model: BayesianCausalModel,
    ) -> str:
        """
        Construct the full user-turn prompt from all component state.

        Args:
            encoded_state:  Output of StateEncoder.encode().
            causal_history: The current CausalHistory instance.
            dfa_controller: The current DFAController instance.
            bayesian_model: The current BayesianCausalModel instance.

        Returns:
            A formatted prompt string ready for the Anthropic API.
        """
        uncertain = bayesian_model.get_most_uncertain(n=3)
        uncertain_str = (
            "\n".join(f"- {h}" for h in uncertain)
            if uncertain
            else "- (no hypotheses yet)"
        )

        diff_str = (
            "\n".join(f"- {item}" for item in encoded_state.get("diff", []))
            if "diff" in encoded_state
            else "(no diff available)"
        )

        user_content = _USER_PROMPT_TEMPLATE.format(
            grid_text=encoded_state["grid_text"],
            avatar_position=encoded_state.get("avatar_position", "unknown"),
            action=encoded_state["action"],
            delta=list(encoded_state["delta"]),
            timestamp=encoded_state["timestamp"],
            avatar_color=encoded_state.get("avatar_color") or "unknown",
            diff=diff_str,
            causal_history=causal_history.to_prompt_string(),
            global_mode=dfa_controller.get_global_mode(),
            uncertain_hypotheses=uncertain_str,
        )
        return user_content

    def call_llm(self, prompt: str) -> str:
        """
        Call claude-sonnet-4-6 with the given prompt and return the response text.

        Retries up to 3 times with exponential backoff on API errors.

        Args:
            prompt: The user-turn prompt string.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If all 3 attempts fail.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.APIError as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"LLM call failed after {max_attempts} attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
        # Should never reach here
        raise RuntimeError("Unexpected exit from retry loop")

    def parse_response(self, response: str) -> LLMResponse:
        """
        Parse the LLM's six-section response into a structured TypedDict.

        Sections are delimited by their header lines (e.g. "OBSERVATIONS:").
        If a section is missing, its value is an empty string.

        Args:
            response: Raw text response from the LLM.

        Returns:
            LLMResponse with keys: observations, hypothesis_update,
            structural_check, predicted_outcome, mode_assessment.
        """
        sections = {
            "observations": "",
            "hypothesis_update": "",
            "structural_check": "",
            "predicted_outcome": "",
            "next_action": "",
            "mode_assessment": "",
        }
        headers = [
            ("OBSERVATIONS:", "observations"),
            ("HYPOTHESIS UPDATE:", "hypothesis_update"),
            ("STRUCTURAL CHECK:", "structural_check"),
            ("PREDICTED OUTCOME:", "predicted_outcome"),
            ("NEXT ACTION:", "next_action"),
            ("MODE ASSESSMENT:", "mode_assessment"),
        ]

        lines = response.splitlines()
        current_key: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()

        for line in lines:
            matched = False
            for header, key in headers:
                if line.strip().startswith(header):
                    flush()
                    current_key = key
                    current_lines = []
                    # Capture any text after the header on the same line
                    after = line.strip()[len(header):].strip()
                    if after:
                        current_lines.append(after)
                    matched = True
                    break
            if not matched and current_key is not None:
                current_lines.append(line)

        flush()
        return LLMResponse(**sections)  # type: ignore[arg-type]

    def get_trace_label(
        self,
        global_mode: str,
        step: int,
        context: dict,
    ) -> str:
        """
        Return the appropriate reasoning trace label for the current step.

        Args:
            global_mode: Current global DFA mode string.
            step:        Current step number (1–7).
            context:     Dict with keys needed by the label template
                         (hypothesis, n, verdict, posterior, event, direction,
                          position, target, action — only those relevant to
                          the step are required).

        Returns:
            Formatted trace label string.
        """
        if step in (1, 2):
            return _TRACE_LABEL_TEMPLATES["explore_env"].format(
                direction=context.get("direction", "?"),
                position=context.get("position", "?"),
            )
        if step == 3:
            return _TRACE_LABEL_TEMPLATES["establish_mechanics"].format(
                hypothesis=context.get("hypothesis", "?"),
                n=context.get("n", "?"),
                verdict=context.get("verdict", "?"),
                posterior=context.get("posterior", 0.0),
            )
        if step == 4:
            return _TRACE_LABEL_TEMPLATES["exploit_mechanics"].format(
                hypothesis=context.get("hypothesis", "?"),
                target=context.get("target", "?"),
            )
        if step == 5:
            return _TRACE_LABEL_TEMPLATES["explore_obj"].format(
                event=context.get("event", "?"),
            )
        if step == 6:
            return _TRACE_LABEL_TEMPLATES["establish_objective"].format(
                hypothesis=context.get("hypothesis", "?"),
                verdict=context.get("verdict", "?"),
                posterior=context.get("posterior", 0.0),
            )
        if step == 7:
            return _TRACE_LABEL_TEMPLATES["exploit_objective"].format(
                hypothesis=context.get("hypothesis", "?"),
                action=context.get("action", "?"),
            )
        return f"Unknown step: {step}"

    def run_turn( 
        self,
        encoded_state: EncodedState,
        causal_history: CausalHistory,
        bayesian_model: BayesianCausalModel,
        uncertain_targets: list[str],
        intended_action: str
    ) -> LLMResponse:   # Triggers an LLM reasoning trace; LLM acts as a sensor, focusing on Bayesian uncertainty 
        
        # 1. Prepares the focus guidance
        target_focus = ", ".join(uncertain_targets) if uncertain_targets else "General Observation"

        # 2. Constructs the prompt with structural constraints to ensure LLM stays in "Observer" lane
        prompt = f"""
        TRANSCRIPT OF CURRENT WORLD STATE:
        {encoded_state['grid_text']}

        CURRENT CAUSAL BELIEFS:
        {causal_history.to_prompt_string()}

        BAYESIAN PRIORITIES:
        The Bayesian model is currently uncertain about: {target_focus}.
        Focus your analysis on these tokens/interactions.

        INTENDED NEXT ACTION (chosen by the DFA controller):
        {intended_action}

        INSTRUCTIONS:
        1. Update your hypotheses based on the grid delta. Use a separate bullet point for each, referencing it by ID (e.g. "- [H1] is consistent", "- [H2] is inconsistent").
        2. Predict the outcome of the INTENDED NEXT ACTION above. Do not choose a different action.
        3. DO NOT suggest a 'next_action'. Your output must follow the LLMResponse schema strictly.
        """

        # 3. Call the API (Placeholder for your Anthropic/LiteLLM call)
        # response = self.client.messages.create(...)
        
        # For now, we return a dummy valid LLMResponse for testing
        return {
            "observations": "Observing the movement near " + target_focus,
            "hypothesis_update": "The token likely acts as a solid boundary.",
            "structural_check": "Logic consistent with previous turn.",
            "predicted_outcome": "Avatar will remain at current position if blocked.",
            "mode_assessment": "EXPLORE"
        }


# SECTION       #? LOGGER (standalone functions)
#___________________________________________________________________________

def _ensure_log_dir(game_id: str) -> Path:
    """
    Create (if necessary) and return the log directory for a game.

    Args:
        game_id: Unique game identifier string.

    Returns:
        Path to logs/game_{game_id}/.
    """
    log_dir = Path("logs") / f"game_{game_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_turn(
    game_id: str,
    turn: int,
    timestamp: float,
    step: int,
    global_dfa_mode: str,
    trace_fired: bool,
    trace_label: str | None,
    encoded_state: EncodedState,
    diff: list[str],
    llm_response: LLMResponse | None,
    residual: bool,
    bayesian_posteriors: dict[str, float],
    causal_history_snapshot: dict,
    action_taken: str,
    delta: tuple[int, int] | list[int],
) -> None:
    """
    Write one turn's full data to a JSON log file.

    Log is written to: logs/game_{game_id}/turn_{turn:04d}.json

    The reasoning traces are primary research data and will be compared against
    human think-aloud protocols, so every turn is logged regardless of whether
    a trace fired.

    Args:
        game_id:                 Unique game identifier.
        turn:                    Turn number (0-indexed).
        timestamp:               Unix timestamp of the turn.
        step:                    Current agent step (1–7).
        global_dfa_mode:         Global DFA mode string.
        trace_fired:             Whether a reasoning trace was generated.
        trace_label:             Trace label string, or None if no trace.
        encoded_state:           The EncodedState dict for this turn.
        diff:                    Positional diff list.
        llm_response:            Parsed LLMResponse, or None if silent turn.
        residual:                Whether the prediction residual was non-zero.
        bayesian_posteriors:     Dict of {hypothesis: posterior_mean}.
        causal_history_snapshot: Snapshot of CausalHistory.to_dict().
        action_taken:            The action that was executed this turn.
        delta:                   Movement vector as [row_delta, col_delta].
    """
    log_dir = _ensure_log_dir(game_id)
    log_path = log_dir / f"turn_{turn:04d}.json"

    entry = {
        "turn": turn,
        "timestamp": timestamp,
        "step": step,
        "global_dfa_mode": global_dfa_mode,
        "trace_fired": trace_fired,
        "trace_label": trace_label,
        "encoded_state": {
            "grid_text": encoded_state.get("grid_text", ""),
            "action": encoded_state.get("action", ""),
            "delta": list(encoded_state.get("delta", [])),
            "timestamp": encoded_state.get("timestamp", 0.0),
            "avatar_color": encoded_state.get("avatar_color"),
            "avatar_position": (
                list(encoded_state["avatar_position"])
                if encoded_state.get("avatar_position") is not None
                else None
            ),
        },
        "diff": diff,
        "llm_response": (
            {
                "observations": llm_response["observations"],
                "hypothesis_update": llm_response["hypothesis_update"],
                "structural_check": llm_response["structural_check"],
                "predicted_outcome": llm_response["predicted_outcome"],
                "next_action": llm_response["next_action"],
                "mode_assessment": llm_response["mode_assessment"],
            }
            if llm_response is not None
            else None
        ),
        "residual": residual,
        "bayesian_posteriors": bayesian_posteriors,
        "causal_history_snapshot": causal_history_snapshot,
        "action_taken": action_taken,
        "delta": list(delta),
    }

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, ensure_ascii=False)


# SECTION       #? RUN_AGENT
#___________________________________________________________________________

def get_game_state() -> GameState:
    """
    Fetch the current game state from the Flask server on localhost:5000.

    Polls GET /state until a state is available, retrying every 0.5 s.

    Returns:
        The latest GameState dict pushed by the JS game.
    """
    import requests as _requests

    while True:
        try:
            resp = _requests.get("http://localhost:5000/state", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except _requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)


def execute_action(action: str) -> None:
    """
    Send an action to the JS game via the Flask server on localhost:5000.

    Posts {"action": action} to POST /action.

    Args:
        action: Action string (e.g. "ArrowDown", "ArrowLeft").
    """
    import requests as _requests

    _requests.post(
        "http://localhost:5000/action",
        json={"action": action},
        timeout=5,
    )

def _infer_step(
    global_mode: str,
    causal_history: CausalHistory,
) -> int:
    """
    Infer the current agent step (1–7) from global DFA mode and history state.

    Step mapping:
      1 — EXPLORE, mechanics track has no hypotheses at all (blank slate)
      2 — EXPLORE, mechanics track has active hypotheses
      3 — ESTABLISH, mechanics track dominant
      4 — EXPLOIT, mechanics track dominant
      5 — EXPLORE, objective track has active hypotheses
      6 — ESTABLISH, objective track dominant
      7 — EXPLOIT, objective track dominant

    Args:
        global_mode:    Current global DFA mode.
        causal_history: Current CausalHistory instance.

    Returns:
        Integer step in [1, 7].
    """
    hist = causal_history.to_dict()
    mech = hist.get("mechanics", {})
    obj = hist.get("objective", {})

    has_mech = any(mech.get(cat) for cat in _CATEGORIES)
    has_obj = any(obj.get(cat) for cat in _CATEGORIES)
    mech_confirmed = bool(mech.get("confirmed"))
    obj_confirmed = bool(obj.get("confirmed"))

    if global_mode == "EXPLORE":
        if not has_mech:
            return 1
        if has_obj:
            return 5
        return 2

    if global_mode == "ESTABLISH":
        if obj.get("hypothesized"):
            return 6
        return 3

    if global_mode == "EXPLOIT":
        if obj_confirmed:
            return 7
        return 4

    return 1  # default

def run_agent(game_id: str, max_turns: int = 50) -> None:
    """
    Main agent loop: orchestrates all EEE components across turns.

    Initialises all components fresh for each call. Stubs for get_game_state()
    and execute_action() must be replaced with real frontend integration before
    the loop can run.

    Args:
        game_id:   Unique identifier for this game session (used for log paths).
        max_turns: Maximum number of turns before halting. Default 50.
    """
    # -- Initialise components --
    causal_history = CausalHistory()
    bayesian_model = BayesianCausalModel()
    dfa_controller = DFAController()
    state_encoder = StateEncoder()
    llm_engine = LLMReasoningEngine()

    prev_game_state: GameState | None = None
    last_action: str = ""
    last_delta: tuple[int, int] = (0, 0)

    # Tracks raw observation counts per hypothesis (for DFA transition checks)
    observation_counts: dict[str, int] = {}

    for turn in range(max_turns):
        turn_timestamp = time.time()

        # (a) Fetch current game state [TBD: frontend integration]
        current_game_state = get_game_state()

        # (b) Encode current state
        encoded_state = state_encoder.encode(
            game_state=current_game_state,
            prev_game_state=prev_game_state,
            action=last_action,
            delta=last_delta,
            timestamp=turn_timestamp,
        )

        # Compute diff and attach to encoded_state for LLM engine
        actual_diff = state_encoder._compute_diff(
            state_encoder._prev_remapped_grid,
            state_encoder._remap_grid(
                current_game_state.get("grid", []),
                state_encoder._detect_avatar_symbol(
                    current_game_state.get("grid", []),
                    current_game_state.get("playerPosition"),
                ),
            ),
        )
        encoded_state["diff"] = actual_diff  # type: ignore[typeddict-unknown-key]
        encoded_state["predicted_diff"] = llm_engine._last_predicted_outcome  # type: ignore[typeddict-unknown-key]

        # (c) Compute residual against last prediction
        residual = state_encoder.compute_residual(
            actual_diff, llm_engine._last_predicted_outcome
        )

        global_mode = dfa_controller.get_global_mode()
        step = _infer_step(global_mode, causal_history)

        # (d) Silent turn: no residual, just log and continue
        if not residual:
            log_turn(
                game_id=game_id,
                turn=turn,
                timestamp=turn_timestamp,
                step=step,
                global_dfa_mode=global_mode,
                trace_fired=False,
                trace_label=None,
                encoded_state=encoded_state,
                diff=actual_diff,
                llm_response=None,
                residual=False,
                bayesian_posteriors=bayesian_model.get_all_posteriors(),
                causal_history_snapshot=causal_history.to_dict(),
                action_taken=last_action,
                delta=list(last_delta),
            )
            # Still need to decide and execute next action — re-use the last
            # known prediction's NEXT ACTION if available. In a real integration
            # the action would come from the prior LLM response.
            # [TBD: frontend integration] execute_action(next_action)
            prev_game_state = current_game_state
            continue

 # (e) Residual != 0: fire reasoning trace
        # First, identify what we are confused about
        uncertain_targets = bayesian_model.get_most_uncertain(n=2)
        
# (f) DFA chooses action FIRST
        next_action = dfa_controller.select_action(
            bayesian_model=bayesian_model, 
            uncertain_targets=uncertain_targets
        )

        # Then, send it to the LLM and include the action
        llm_response = llm_engine.run_turn(
            encoded_state=encoded_state,
            causal_history=causal_history,
            bayesian_model=bayesian_model,
            uncertain_targets=uncertain_targets,
            intended_action=next_action # <--- Add this too
        )

        # (g) Update Bayesian model and DFA for each hypothesis in HYPOTHESIS UPDATE
        trace_label: str | None = None
        if llm_response is not None:
            hypothesis_update_text = llm_response["hypothesis_update"]
            # Parse hypothesis updates using [H1]-style IDs for stable matching.
            id_to_hypothesis = {
                v: k for k, v in getattr(causal_history, "_hypothesis_ids", {}).items()
            }
            for track in _TRACKS:
                track_hist = causal_history.to_dict().get(track, {})
                active_hypotheses: list[str] = (
                    track_hist.get("hypothesized", [])
                    + track_hist.get("uncertain", [])
                    + track_hist.get("confirmed", [])
                )
                for match in re.finditer(r"(\[H\d+\])[^\n]*?(inconsistent|consistent)", hypothesis_update_text, re.IGNORECASE):
                    h_id = match.group(1)
                    verdict = match.group(2).lower()
                    hypothesis = id_to_hypothesis.get(h_id)
                    if hypothesis is None or hypothesis not in active_hypotheses:
                        continue
                    consistent = (verdict == "consistent")

                    observation_counts[hypothesis] = (
                        observation_counts.get(hypothesis, 0) + 1
                    )
                    posterior = bayesian_model.record_observation(
                        hypothesis, track, consistent
                    )

                    if posterior is not None:
                        # Sync posterior thresholds to CausalHistory
                        if posterior > _CONFIRMED_THRESHOLD:
                            causal_history.update_category(hypothesis, "confirmed")
                        elif posterior < _DISCONFIRMED_THRESHOLD:
                            causal_history.update_category(hypothesis, "disconfirmed")
                        else:
                            causal_history.update_category(hypothesis, "hypothesized")

                    # DFA transition check
                    dfa_controller.check_transitions(
                        hypothesis=hypothesis,
                        bayes_model=bayesian_model,
                        observation_count=observation_counts.get(hypothesis, 0),
                        stuck_count=0,  # Progress tracking TBD
                        track=track,
                    )

            # Build trace label
            trace_context: dict = {
                "direction": _extract_direction(next_action),
                "position": _extract_position(next_action),
                "hypothesis": _first_hypothesis(causal_history),
                "n": min(
                    observation_counts.get(_first_hypothesis(causal_history), 0),
                    3,
                ),
                "verdict": "consistent",  # Simplified; refine with parser above
                "posterior": (
                    bayesian_model.get_posterior(_first_hypothesis(causal_history))
                    if _first_hypothesis(causal_history) in bayesian_model._posteriors
                    else 0.5
                ),
                "event": actual_diff[0] if actual_diff else "?",
                "target": "?",
                "action": next_action,
            }
            global_mode = dfa_controller.get_global_mode()
            step = _infer_step(global_mode, causal_history)
            trace_label = llm_engine.get_trace_label(global_mode, step, trace_context)

        # (h) Log full turn
        log_turn(
            game_id=game_id,
            turn=turn,
            timestamp=turn_timestamp,
            step=step,
            global_dfa_mode=dfa_controller.get_global_mode(),
            trace_fired=llm_response is not None,
            trace_label=trace_label,
            encoded_state=encoded_state,
            diff=actual_diff,
            llm_response=llm_response,
            residual=residual,
            bayesian_posteriors=bayesian_model.get_all_posteriors(),
            causal_history_snapshot=causal_history.to_dict(),
            action_taken=last_action,
            delta=list(last_delta),
        )

        # (i) Execute next action [TBD: frontend integration]
        if next_action:
            last_action = next_action
            # execute_action(next_action)

        # (j) Advance state
        prev_game_state = current_game_state


# ---------------------------------------------------------------------------
# Small helpers used in run_agent (not part of any class)
# ---------------------------------------------------------------------------


def _extract_direction(action_text: str) -> str:
    """Extract a direction keyword from a NEXT ACTION string."""
    for word in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "up", "down", "left", "right"):
        if word.lower() in action_text.lower():
            return word
    return "?"


def _extract_position(action_text: str) -> str:
    """Extract a (row, col) coordinate mention from a NEXT ACTION string."""
    match = re.search(r"\((\d+),\s*(\d+)\)", action_text)
    if match:
        return f"({match.group(1)},{match.group(2)})"
    return "?"


def _first_hypothesis(causal_history: CausalHistory) -> str:
    """Return the first active hypothesis from CausalHistory, or empty string."""
    snap = causal_history.to_dict()
    for track in _TRACKS:
        for cat in ("hypothesized", "uncertain"):
            items = snap.get(track, {}).get(cat, [])
            if items:
                return items[0]
    return ""
