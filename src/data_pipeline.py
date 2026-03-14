import os
import pandas as pd
from statsbombpy import sb
import warnings

from text_formatter import format_event_to_text

warnings.filterwarnings('ignore')


def fetch_and_process_match(match_id, home_team, output_dir="../data/processed"):
    """
    Fetches raw StatsBomb event data for one match and writes two files:
      - match_{id}_narrative.txt  : one natural-language line per action event
      - match_{id}_meta.csv       : parallel per-line metadata for label generation
                                    columns: minute, second, is_home, xg, event_type
    """
    print(f"  Fetching events for match {match_id}...")
    events = sb.events(match_id=match_id)
    events = events.sort_values(by=['minute', 'second', 'timestamp'])

    action_types = ['Pass', 'Shot', 'Duel', 'Clearance', 'Interception', 'Dribble']
    action_events = events[events['type'].isin(action_types)].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    narrative_file = os.path.join(output_dir, f"match_{match_id}_narrative.txt")
    meta_file      = os.path.join(output_dir, f"match_{match_id}_meta.csv")

    print(f"  Formatting {len(action_events)} events...")

    meta_rows = []
    with open(narrative_file, 'w', encoding='utf-8') as f:
        for _, row in action_events.iterrows():
            f.write(format_event_to_text(row) + "\n")
            meta_rows.append({
                'minute':     int(row['minute']),
                'second':     int(row['second']),
                'is_home':    row['team'] == home_team,
                # shot_statsbomb_xg is NaN for non-shot events; coerce to 0.0
                'xg':         float(row['shot_statsbomb_xg'])
                              if pd.notna(row.get('shot_statsbomb_xg')) else 0.0,
                'event_type': row['type'],
            })

    pd.DataFrame(meta_rows).to_csv(meta_file, index=False)
    print(f"  Saved narrative → {narrative_file}")
    print(f"  Saved meta      → {meta_file}")


if __name__ == "__main__":
    TEST_MATCH_ID = 3754058
    fetch_and_process_match(TEST_MATCH_ID, home_team="Leicester City",
                            output_dir="../data/processed")
