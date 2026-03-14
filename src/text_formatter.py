import pandas as pd

def format_event_to_text(event_row):
    """
    Translates a single StatsBomb event row into a natural language string 
    optimized for LLM tokenization.
    """
    # Extract common spatial and temporal fields
    minute = event_row.get('minute', 0)
    second = event_row.get('second', 0)
    team = event_row.get('team', 'Unknown Team')
    player = event_row.get('player', 'Unknown Player')
    event_type = event_row.get('type', 'Unknown Event')
    location = event_row.get('location', '[Unknown]')

    # Build the base timestamp and possession string
    timestamp = f"[{minute:02d}:{second:02d}]"
    base_str = f"{timestamp} {team} possession: {player}"

    # Handle Passes
    if event_type == 'Pass':
        recipient = event_row.get('pass_recipient', 'Unknown')
        outcome = event_row.get('pass_outcome', 'Complete') 
        # In StatsBomb, a complete pass usually has a NaN outcome
        if pd.isna(outcome):
            outcome_str = "successfully passed"
        else:
            outcome_str = f"attempted a pass that was {outcome.lower()}"
            
        return f"{base_str} {outcome_str} from {location} to {recipient}."
    
    # Handle Shots (Crucial for predicting momentum/scoring opportunities)
    elif event_type == 'Shot':
        outcome = event_row.get('shot_outcome', 'Unknown')
        xg = event_row.get('shot_statsbomb_xg', 0.0)
        return f"{base_str} took a shot from {location} (Outcome: {outcome}, xG: {xg:.2f})."
    
    # Handle Defensive Actions
    elif event_type in ['Duel', 'Clearance', 'Interception']:
        return f"{base_str} performed a {event_type.lower()} at {location}."

    # Catch-all for other events (Dribbles, Ball Receipts, etc.)
    else:
        return f"{base_str} initiated a {event_type.lower()} at {location}."