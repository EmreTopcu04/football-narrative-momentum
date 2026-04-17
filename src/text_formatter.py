"""
Text Formatter Module

This module is responsible for converting structured event data into natural language sentences.
Since Language Models understand text better than raw numerical coordinates, this module 
translates coordinates into named pitch zones (e.g. 'penalty area', 'left flank') and constructs
coherent strings that describe match events with contextual metadata like the current scoreline.
"""
import pandas as pd


# StatsBomb pitch: x = 0–120 (length), y = 0–80 (width)
# Origin is the team's own goal; attacking direction is increasing x.
def _location_to_zone(location):
    """
    Converts raw StatsBomb [x, y] coordinates into a human-readable pitch zone.
    Raw numbers like [30.2, 45.1] are meaningless tokens to an LLM; named zones
    give the model learnable spatial context.
    """
    if location is None or not isinstance(location, (list, tuple)) or len(location) < 2:
        return "an unknown area"

    x, y = float(location[0]), float(location[1])

    # Depth (x-axis, 0 = own goal, 120 = opponent goal)
    if x < 40:
        depth = "defensive third"
    elif x < 80:
        depth = "middle third"
    elif x < 102:
        depth = "attacking third"
    else:
        depth = "penalty area"

    # Width (y-axis, 0 = left, 80 = right when facing goal)
    if y < 18:
        width = "left flank"
    elif y < 36:
        width = "left half-space"
    elif y < 44:
        width = "center"
    elif y < 62:
        width = "right half-space"
    else:
        width = "right flank"

    # Special case: penalty area is compact enough not to need width breakdown
    if depth == "penalty area":
        return "the penalty area"

    return f"the {depth} ({width})"


def _is_attacking_zone(location):
    """Returns True if the event is in the attacking third or penalty area."""
    if location is None or not isinstance(location, (list, tuple)) or len(location) < 2:
        return False
    return float(location[0]) >= 80


class MomentumContext:
    """
    Stateful tracker passed through data_pipeline to enrich each narrative line
    with game-flow context signals:
      - consecutive_attacks: how many straight possessions the team has had
      - recent_shots: shots by each team in the last 5 minutes
      - last_team: which team had the last action (for attack-streak counting)
    """
    def __init__(self):
        self.last_team           = None
        self.consecutive_attacks = 0
        self.shot_log            = []            # list of (minute, team)
        self.SHOT_WINDOW_MINS    = 5

    def update(self, team, minute, is_shot=False):
        # Consecutive attack counter
        if team == self.last_team:
            self.consecutive_attacks += 1
        else:
            self.consecutive_attacks = 1
            self.last_team = team

        # Track recent shots
        if is_shot:
            self.shot_log.append((minute, team))
        # Prune old shots outside the window
        self.shot_log = [(m, t) for m, t in self.shot_log
                         if minute - m <= self.SHOT_WINDOW_MINS]

    def recent_shots(self, team, minute):
        return sum(1 for m, t in self.shot_log
                   if t == team and minute - m <= self.SHOT_WINDOW_MINS)

    def pressure_tag(self, team, minute, location):
        """
        Returns a short context string appended to the narrative line.
        Examples:
          "(4th consecutive attack)"
          "(3 shots in last 5 min)"
          "(sustained pressure)"
        """
        tags = []

        streak = self.consecutive_attacks
        if streak >= 5:
            tags.append(f"{streak}th consecutive {team} attack")
        elif streak >= 3:
            tags.append(f"{streak}rd consecutive attack")

        shots = self.recent_shots(team, minute)
        if shots >= 3:
            tags.append(f"{shots} shots in last 5 min")
        elif shots == 2:
            tags.append("2 recent shots")

        if _is_attacking_zone(location) and streak >= 3:
            tags.append("sustained pressure")

        return f" [{', '.join(tags)}]" if tags else ""


def format_event_to_text(event_row, home_score=0, away_score=0,
                         home_team="Home", away_team="Away",
                         momentum_ctx=None):
    """
    Translates a single StatsBomb event row into a natural-language description.
    
    This incorporates:
    - Pitch locations translated into named zones (e.g. 'attacking third').
    - Current scoreline injected into the string for match context.
    - Contextual tags (like prolonged attacking pressure or recent shots).
    """
    minute     = int(event_row.get('minute', 0))
    second     = int(event_row.get('second', 0))
    team       = event_row.get('team', 'Unknown Team')
    player     = event_row.get('player', 'Unknown Player')
    event_type = event_row.get('type', 'Unknown Event')
    location   = event_row.get('location', None)

    is_home    = (team == home_team)
    opponent   = away_team if is_home else home_team
    score_str  = f"{home_team} {home_score}-{away_score} {away_team}"
    zone       = _location_to_zone(location)

    is_shot = (event_type == 'Shot')

    # Update momentum context if provided
    if momentum_ctx is not None:
        momentum_ctx.update(team, minute, is_shot=is_shot)
        ctx_tag = momentum_ctx.pressure_tag(team, minute, location)
    else:
        ctx_tag = ""

    timestamp  = f"[{minute:02d}:{second:02d}]"
    base       = f"{timestamp} [{score_str}] {team}: {player}"

    # ── Pass ──────────────────────────────────────────────────────────────
    if event_type == 'Pass':
        recipient = event_row.get('pass_recipient', 'a teammate')
        outcome   = event_row.get('pass_outcome', None)
        height    = event_row.get('pass_height', None)
        height_str = f" ({height.lower()})" if pd.notna(height) and height else ""

        if pd.isna(outcome) or outcome is None:
            return f"{base} completed a pass{height_str} from {zone} to {recipient}{ctx_tag}."
        else:
            return (f"{base} attempted a {str(outcome).lower()} pass{height_str} "
                    f"from {zone} to {recipient}{ctx_tag}.")

    # ── Shot ──────────────────────────────────────────────────────────────
    elif event_type == 'Shot':
        outcome  = event_row.get('shot_outcome', 'Unknown')
        xg_raw   = event_row.get('shot_statsbomb_xg', 0.0)
        xg       = float(xg_raw) if pd.notna(xg_raw) else 0.0
        body     = event_row.get('shot_body_part', None)
        body_str = f" with {body.lower()}" if pd.notna(body) and body else ""
        danger   = "high-danger" if xg >= 0.3 else ("chance" if xg >= 0.1 else "low-chance")
        return (f"{base} took a {danger} shot{body_str} from {zone} "
                f"({str(outcome).lower()}, xG: {xg:.2f}){ctx_tag}.")

    # ── Duel ──────────────────────────────────────────────────────────────
    elif event_type == 'Duel':
        duel_type = event_row.get('duel_type', 'duel')
        outcome   = event_row.get('duel_outcome', None)
        out_str   = f" ({str(outcome).lower()})" if pd.notna(outcome) and outcome else ""
        return f"{base} contested a {str(duel_type).lower()}{out_str} in {zone}{ctx_tag}."

    # ── Clearance ─────────────────────────────────────────────────────────
    elif event_type == 'Clearance':
        return f"{base} cleared the ball in {zone} under pressure from {opponent}{ctx_tag}."

    # ── Interception ──────────────────────────────────────────────────────
    elif event_type == 'Interception':
        outcome = event_row.get('interception_outcome', None)
        out_str = f" ({str(outcome).lower()})" if pd.notna(outcome) and outcome else ""
        return f"{base} intercepted a {opponent} pass in {zone}{out_str}{ctx_tag}."

    # ── Dribble ───────────────────────────────────────────────────────────
    elif event_type == 'Dribble':
        outcome = event_row.get('dribble_outcome', None)
        out_str = str(outcome).lower() if pd.notna(outcome) and outcome else "attempted"
        return f"{base} {out_str} a dribble in {zone}{ctx_tag}."

    # ── Fallback ──────────────────────────────────────────────────────────
    else:
        return f"{base} performed a {event_type.lower()} in {zone}{ctx_tag}."