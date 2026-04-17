"""
Data Pipeline Module

This module handles fetching and formatting raw football match events from the StatsBomb API.
It converts technical match event objects into sequential, human-readable narrative text lines 
and generates corresponding metadata for each event (e.g., minute, second, xG, running score).
"""
import os
import pandas as pd
from statsbombpy import sb
import warnings

from text_formatter import format_event_to_text, MomentumContext

warnings.filterwarnings('ignore')

# Event types to include in the narrative
ACTION_TYPES = ['Pass', 'Shot', 'Duel', 'Clearance', 'Interception', 'Dribble']


def fetch_and_process_match(match_id, home_team, output_dir="../data/raw"):
    """
    Fetches raw StatsBomb event data for one match and writes two files:
      - match_{id}_narrative.txt  : one natural-language line per action event
      - match_{id}_meta.csv       : parallel per-line metadata

    Meta columns
    ------------
    minute, second  : event time
    is_home         : True if the acting team is the home team
    xg              : shot xG (0.0 for non-shot events)
    event_type      : StatsBomb type string
    is_goal         : True if this event is a goal (shot outcome == 'Goal')
    home_score      : running home score AT the time of this event
    away_score      : running away score AT the time of this event
    cumulative_home_xg : cumulative home xG up to (and including) this event
    cumulative_away_xg : cumulative away xG up to (and including) this event
    """
    print(f"  Fetching events for match {match_id}...")
    events = sb.events(match_id=match_id)
    events = events.sort_values(by=['minute', 'second', 'timestamp']).reset_index(drop=True)

    # ── Running score tracker ─────────────────────────────────────────────
    home_score = 0
    away_score = 0
    score_before  = {}
    is_goal_flags = {}

    for idx, row in events.iterrows():
        score_before[idx]  = (home_score, away_score)
        this_is_goal = False
        if row.get('type') == 'Shot':
            outcome = row.get('shot_outcome', None)
            if pd.notna(outcome) and str(outcome) == 'Goal':
                this_is_goal = True
                if row.get('team') == home_team:
                    home_score += 1
                else:
                    away_score += 1
        is_goal_flags[idx] = this_is_goal

    # ── Filter to action events only ─────────────────────────────────────
    action_events = events[events['type'].isin(ACTION_TYPES)].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    narrative_file = os.path.join(output_dir, f"match_{match_id}_narrative.txt")
    meta_file      = os.path.join(output_dir, f"match_{match_id}_meta.csv")

    print(f"  Formatting {len(action_events)} events...")

    meta_rows    = []
    cum_home_xg  = 0.0
    cum_away_xg  = 0.0
    momentum_ctx = MomentumContext()   # tracks consecutive attacks & shot freq

    with open(narrative_file, 'w', encoding='utf-8') as f:
        for _, row in action_events.iterrows():
            orig_idx  = row.name
            h, a      = score_before.get(orig_idx, (0, 0))
            this_goal = is_goal_flags.get(orig_idx, False)

            # Narrative line: includes live scoreline + momentum context tags
            line = format_event_to_text(
                row,
                home_score=h,
                away_score=a,
                home_team=home_team,
                away_team="Opponent",
                momentum_ctx=momentum_ctx,
            )
            f.write(line + "\n")

            # xG for this event
            xg_val = 0.0
            if row.get('type') == 'Shot':
                raw = row.get('shot_statsbomb_xg', 0.0)
                xg_val = float(raw) if pd.notna(raw) else 0.0

            is_home_event = (row.get('team') == home_team)
            if is_home_event:
                cum_home_xg += xg_val
            else:
                cum_away_xg += xg_val

            meta_rows.append({
                'minute':              int(row['minute']),
                'second':              int(row['second']),
                'is_home':             is_home_event,
                'xg':                  xg_val,
                'event_type':          row['type'],
                'is_goal':             this_goal,
                'home_score':          h,
                'away_score':          a,
                'cumulative_home_xg':  round(cum_home_xg, 4),
                'cumulative_away_xg':  round(cum_away_xg, 4),
            })

    pd.DataFrame(meta_rows).to_csv(meta_file, index=False)
    print(f"  Saved narrative → {narrative_file}")
    print(f"  Saved meta      → {meta_file}")


if __name__ == "__main__":
    TEST_MATCH_ID = 3754058
    fetch_and_process_match(TEST_MATCH_ID, home_team="Leicester City",
                            output_dir="../data/raw")