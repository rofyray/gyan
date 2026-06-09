"""World Football Elo ratings for GYAN (eloratings.net formula, source D2).

Implements the exact eloratings.net rating system and a date-ordered walk that
attaches each team's PRE-match Elo to every historical fixture. Pre-match Elo is
a core feature for the goal engine (Stage 1 Task 1.7) and the simulator anchor
(Stage 2), and is the most important single covariate in the literature
(Groll et al., R3).

Formula (all constants live in config.py):

    R_new = R_old + K * G * (W - W_e)

    K  = match-importance factor (config.ELO_K_BY_IMPORTANCE): 60 WC finals,
         50 continental finals / major intercontinental, 40 WC and continental
         qualifiers and other major tournaments, 30 other tournaments, 20 friendlies.
    G  = goal-difference index: 1 for a draw or one-goal win, 1.5 for two goals,
         1.75 for three, and 1.75 + (N-3)/8 == (11+N)/8 for N >= 4 goals.
    W_e = 1 / (10^(-dr/400) + 1), with dr = (R_home - R_away) + home_advantage,
         where home_advantage is config.ELO_HOME_ADVANTAGE (100) only at a genuine
         non-neutral venue, else 0.
    W  = 1 win / 0.5 draw / 0 loss. A penalty shootout counts as a draw for Elo
         (the score is level at the end of extra time, so G = 1 and W = 0.5).

New teams initialise at config.ELO_INITIAL_RATING (1500). The system is zero-sum
per match: the winner's gain equals the loser's loss.
"""

from __future__ import annotations  # modern type-hint syntax on all runtimes

from pathlib import Path  # accepts both string and Path inputs for reference files

import pandas as pd  # the canonical match table is a pandas DataFrame

# All numeric constants come from the single source of truth, config.py.
from gyan.config import (
    ELO_INITIAL_RATING,    # rating for a team with no history (1500)
    ELO_HOME_ADVANTAGE,    # rating points added to a genuine home side (100)
    ELO_DIVISOR,           # logistic divisor in the win-expectancy formula (400)
    ELO_K_BY_IMPORTANCE,   # label -> K mapping (60/50/40/30/20)
)


def final_ratings_from_elo_features(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Return the latest post-match Elo rating for every team.

    Parameters
    ----------
    matches_with_elo : pandas.DataFrame
        Match table produced by compute_elo_features.

    Returns
    -------
    pandas.DataFrame
        One row per team with team, elo_rating, and last_match_date.
    """
    ordered = matches_with_elo.sort_values("date", kind="mergesort")  # ensure latest means latest
    latest: dict[str, dict[str, object]] = {}  # team -> latest rating/date record
    for row in ordered.itertuples(index=False):  # chronological rows are required here
        latest[row.home_team] = {  # update the home team's latest post-match rating
            "team": row.home_team,
            "elo_rating": float(row.home_elo_post),
            "last_match_date": row.date,
        }
        latest[row.away_team] = {  # update the away team's latest post-match rating
            "team": row.away_team,
            "elo_rating": float(row.away_elo_post),
            "last_match_date": row.date,
        }
    ratings = pd.DataFrame(latest.values())  # convert the dict records to a table
    return ratings.sort_values("elo_rating", ascending=False).reset_index(drop=True)  # rank desc


def parse_eloratings_reference(world_tsv_path: Path | str, labels_tsv_path: Path | str) -> pd.DataFrame:
    """Parse eloratings.net World.tsv into team names and reference ratings.

    Parameters
    ----------
    world_tsv_path : Path | str
        Raw D2 World.tsv path from eloratings.net.
    labels_tsv_path : Path | str
        Raw D2 en.teams.tsv path from eloratings.net.

    Returns
    -------
    pandas.DataFrame
        Columns: team, elo_reference_rank, elo_reference_rating, eloratings_code.
    """
    label_lookup: dict[str, str] = {}  # two-letter code -> English team name
    labels_path = Path(labels_tsv_path)  # normalise label path inputs
    world_path = Path(world_tsv_path)  # normalise world-rating path inputs
    with labels_path.open(encoding="utf-8") as handle:  # read labels as text
        for line in handle:  # one code and one-or-more labels per row
            parts = line.rstrip("\n").split("\t")  # split tab-separated labels
            if len(parts) >= 2 and not parts[0].endswith("_loc"):  # ignore location labels
                label_lookup[parts[0]] = parts[1]  # first label is the canonical English name
    world = pd.read_csv(world_path, sep="\t", header=None)  # raw current world ratings
    parsed = pd.DataFrame(  # extract only fields needed for spot-checks
        {
            "team": world[2].map(label_lookup),  # decode eloratings team code
            "elo_reference_rank": world[0].astype(int),  # current eloratings rank
            "elo_reference_rating": world[3].astype(float),  # current eloratings rating
            "eloratings_code": world[2].astype(str),  # keep the source code for audit
        }
    )
    return parsed.dropna(subset=["team"]).reset_index(drop=True)  # drop unknown labels


def build_elo_spotcheck(
    final_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    selected_teams: tuple[str, ...],
    tolerance: float = 75.0,
) -> pd.DataFrame:
    """Compare computed final ratings to eloratings.net reference values.

    Parameters
    ----------
    final_ratings : pandas.DataFrame
        Output of final_ratings_from_elo_features.
    reference_ratings : pandas.DataFrame
        Output of parse_eloratings_reference.
    selected_teams : tuple[str, ...]
        Teams used for the formal tolerance check.
    tolerance : float
        Maximum absolute rating delta accepted for selected teams.

    Returns
    -------
    pandas.DataFrame
        Merged spot-check rows with deltas and pass/fail flags.
    """
    merged = reference_ratings.merge(final_ratings, on="team", how="inner")  # compare common teams
    merged["rating_delta"] = merged["elo_rating"] - merged["elo_reference_rating"]  # signed delta
    merged["abs_rating_delta"] = merged["rating_delta"].abs()  # absolute delta for thresholds
    selected_set = set(selected_teams)  # fast selected-team membership
    merged["selected_for_check"] = merged["team"].isin(selected_set)  # formal check flag
    merged["within_tolerance"] = merged["abs_rating_delta"] <= tolerance  # tolerance flag
    return merged.sort_values("elo_reference_rank").reset_index(drop=True)  # reference rank order

# ---------------------------------------------------------------------------
# Importance classification: map a tournament name to an eloratings K-factor.
# These rules are a sensible default for the martj42 dataset's `tournament`
# field (D1); verify the exact categorisation against eloratings.net (D2). The
# Stage 1 validation tolerance (+/- 75 points) absorbs minor disagreements.
# ---------------------------------------------------------------------------

# Continental championship FINALS tournaments and major intercontinental events
# map to the K=50 ("continental_finals") bucket.
_CONTINENTAL_FINALS_NAMES: frozenset[str] = frozenset({
    "UEFA Euro",                 # European Championship finals
    "Copa América",              # South American championship finals
    "African Cup of Nations",    # African championship finals
    "AFC Asian Cup",             # Asian championship finals
    "CONCACAF Championship",     # historical North/Central American finals
    "Gold Cup",                  # CONCACAF Gold Cup finals
    "CONCACAF Gold Cup",         # alternative naming in some datasets
    "Oceania Nations Cup",       # Oceania championship finals
    "FIFA Confederations Cup",   # major intercontinental tournament
})


def classify_importance(tournament: str) -> str:
    """Return the eloratings importance label for a tournament name.

    Parameters
    ----------
    tournament : str
        The tournament field from the match record (D1).

    Returns
    -------
    str
        One of the keys of config.ELO_K_BY_IMPORTANCE.
    """
    name = (tournament or "").strip()             # normalise None / whitespace
    if name == "FIFA World Cup":                  # the World Cup final tournament
        return "world_cup_finals"                 # K = 60
    if "qualification" in name.lower():           # any qualifier (WC or continental)
        return "wc_continental_qualifier"         # K = 40
    if name == "UEFA Nations League":             # treated as a major tournament
        return "wc_continental_qualifier"         # K = 40
    if name in _CONTINENTAL_FINALS_NAMES:         # continental finals / intercontinental
        return "continental_finals"               # K = 50
    if name == "Friendly":                        # friendlies
        return "friendly"                         # K = 20
    return "other_tournament"                     # everything else (minor tournaments), K = 30


def k_factor_for_importance(importance: str) -> int:
    """Return the K-factor for an importance label (config.ELO_K_BY_IMPORTANCE)."""
    return ELO_K_BY_IMPORTANCE[importance]        # dict lookup; KeyError if unknown label


def assign_k_factors(matches: pd.DataFrame, tournament_col: str = "tournament") -> pd.Series:
    """Return a Series of K-factors, one per match row, from the tournament name.

    Parameters
    ----------
    matches : pandas.DataFrame
        The match table containing a tournament-name column.
    tournament_col : str
        Name of the tournament column (default "tournament").

    Returns
    -------
    pandas.Series
        Integer K-factor per row, aligned to the input index.
    """
    importance_labels = matches[tournament_col].map(classify_importance)  # label per row
    return importance_labels.map(k_factor_for_importance).astype(int)     # K per row


# ---------------------------------------------------------------------------
# Core Elo maths.
# ---------------------------------------------------------------------------

def goal_difference_index(goal_diff_abs: int) -> float:
    """Return the eloratings goal-difference multiplier G for |goal difference|.

    Parameters
    ----------
    goal_diff_abs : int
        Absolute goal difference of the match (>= 0).

    Returns
    -------
    float
        The multiplier G (D2): 1 for <=1 goal, 1.5 for 2, 1.75 for 3, and
        1.75 + (N-3)/8 for N >= 4.
    """
    if goal_diff_abs <= 1:                         # draws and one-goal wins
        return 1.0                                 # no amplification
    if goal_diff_abs == 2:                         # two-goal wins
        return 1.5                                 # modest amplification
    if goal_diff_abs == 3:                         # three-goal wins
        return 1.75                                # larger amplification
    return 1.75 + (goal_diff_abs - 3) / 8.0        # 4+ goals: keep growing slowly


def expected_score(rating_a: float, rating_b: float,
                   home_advantage: float, divisor: float = ELO_DIVISOR) -> float:
    """Win expectancy W_e for team A vs team B under the eloratings logistic.

    Parameters
    ----------
    rating_a, rating_b : float
        Current Elo ratings of team A (the side being scored) and team B.
    home_advantage : float
        Points added to A's effective rating (config.ELO_HOME_ADVANTAGE at a
        non-neutral venue when A is home, else 0).
    divisor : float
        Logistic divisor (config.ELO_DIVISOR, 400).

    Returns
    -------
    float
        W_e in (0, 1); the expected score (1 = certain win, 0.5 = even).

    Notes
    -----
    dr = (rating_a - rating_b) + home_advantage. By the logistic identity, the
    opponent's expectancy is simply 1 - W_e.
    """
    rating_diff = (rating_a - rating_b) + home_advantage   # adjusted rating gap dr
    return 1.0 / (10.0 ** (-rating_diff / divisor) + 1.0)  # logistic expectancy


def result_to_actual(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Map a final score to actual Elo scores (W_home, W_away).

    A level score (including a game decided on penalties, which is level at the
    end of extra time) is a draw for Elo: (0.5, 0.5).
    """
    if home_goals > away_goals:                    # home win
        return 1.0, 0.0                            # full point home, none away
    if home_goals < away_goals:                    # away win
        return 0.0, 1.0                            # none home, full point away
    return 0.5, 0.5                                # draw (or shootout): half each


def elo_update(rating: float, k_factor: float, goal_diff_index: float,
               actual: float, expected: float) -> float:
    """Apply one eloratings update and return the new (unrounded) rating.

    R_new = R_old + K * G * (W - W_e).
    """
    points_change = k_factor * goal_diff_index * (actual - expected)  # the delta
    return rating + points_change                                    # the new rating


# ---------------------------------------------------------------------------
# Sequential rating tracker and the date-ordered feature walk.
# ---------------------------------------------------------------------------

class EloTracker:
    """Maintains current Elo ratings and applies eloratings updates per match.

    The tracker is stateful: feed it matches in chronological order. Unseen teams
    are lazily initialised to config.ELO_INITIAL_RATING.

    Parameters
    ----------
    round_changes : bool
        If True (default), round each match's points change to the nearest integer
        as eloratings.net does. Rounding stays zero-sum because the away change is
        the exact negative of the (rounded) home change. Set False for a smoother,
        unrounded rating to feed the goal model.
    """

    def __init__(self, round_changes: bool = True) -> None:
        """Initialise an empty rating book."""
        self.ratings: dict[str, float] = {}        # team -> current Elo rating
        self.round_changes = round_changes         # whether to round points changes

    def get(self, team: str) -> float:
        """Return a team's current rating, initialising new teams to 1500."""
        return self.ratings.get(team, ELO_INITIAL_RATING)  # default for unseen teams

    def update_match(self, home_team: str, away_team: str,
                     home_goals: int, away_goals: int,
                     k_factor: float, neutral: bool) -> dict[str, float]:
        """Apply one match update and return the pre/post ratings and expectancy.

        Parameters
        ----------
        home_team, away_team : str
            Team names (must be canonicalised upstream, Stage 1 Task 1.3).
        home_goals, away_goals : int
            Final score (level if the game went to penalties).
        k_factor : float
            Match-importance K (see assign_k_factors).
        neutral : bool
            True if the venue is neutral (then no home advantage is applied).

        Returns
        -------
        dict[str, float]
            Keys: home_elo_pre, away_elo_pre, home_elo_post, away_elo_post,
            home_win_expectancy.
        """
        home_pre = self.get(home_team)             # home rating before the match
        away_pre = self.get(away_team)             # away rating before the match

        # Home advantage applies only at a genuine (non-neutral) home venue.
        home_advantage = 0.0 if neutral else ELO_HOME_ADVANTAGE  # 0 at neutral sites

        # Win expectancy for the home side; the away expectancy is 1 - that.
        home_expected = expected_score(home_pre, away_pre, home_advantage)  # W_e home

        # Actual scores from the result, and the shared goal-difference multiplier.
        home_actual, away_actual = result_to_actual(home_goals, away_goals)  # W home/away
        goal_diff = abs(home_goals - away_goals)    # |goal difference| for this match
        g_index = goal_difference_index(goal_diff)  # the G multiplier (same for both)

        # Compute the home points change; the away change is its exact negative
        # (zero-sum). Round if configured (eloratings rounds the change).
        home_change = k_factor * g_index * (home_actual - home_expected)  # raw delta
        if self.round_changes:                      # match official eloratings rounding
            home_change = float(round(home_change)) # round to nearest integer

        # Apply the zero-sum update: home gains `home_change`, away loses the same.
        self.ratings[home_team] = home_pre + home_change  # new home rating
        self.ratings[away_team] = away_pre - home_change  # new away rating (mirror)

        # Return everything callers need for feature columns and logging.
        return {
            "home_elo_pre": home_pre,                       # home rating before
            "away_elo_pre": away_pre,                       # away rating before
            "home_elo_post": self.ratings[home_team],       # home rating after
            "away_elo_post": self.ratings[away_team],       # away rating after
            "home_win_expectancy": home_expected,           # W_e for the home side
        }


def compute_elo_features(
    matches: pd.DataFrame,
    round_changes: bool = True,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    neutral_col: str = "neutral",
    k_col: str = "k_factor",
    importance_col: str = "importance_weight",
) -> pd.DataFrame:
    """Walk matches in date order and append pre/post Elo and win-expectancy columns.

    Elo is inherently sequential (each rating depends on all prior matches), so
    this is a chronological loop rather than a vectorised operation. For ~49k
    international matches this completes in well under a second.

    Parameters
    ----------
    matches : pandas.DataFrame
        The canonical match table. Must contain the columns named by the *_col
        arguments. A `k_factor` column can be produced with assign_k_factors; if
        it is absent, an upstream `importance_weight` column is used before
        falling back to deriving K from the `tournament` column.
    round_changes : bool
        Passed to EloTracker (round points changes like eloratings, default True).
    date_col, home_col, away_col, home_goals_col, away_goals_col, neutral_col, k_col : str
        Column names in `matches`.
    importance_col : str
        Upstream cleaned K-factor weight column to use when `k_col` is absent.

    Returns
    -------
    pandas.DataFrame
        A copy of `matches` sorted by date with five appended columns:
        home_elo_pre, away_elo_pre, home_elo_post, away_elo_post,
        home_win_expectancy.

    Raises
    ------
    KeyError
        If required columns are missing and cannot be derived.
    """
    # Validate the essential columns are present before doing any work.
    required = [date_col, home_col, away_col, home_goals_col, away_goals_col, neutral_col]
    missing = [c for c in required if c not in matches.columns]  # any missing names
    if missing:                                                  # fail loudly and early
        raise KeyError(f"compute_elo_features missing columns: {missing}")

    # Work on a date-sorted copy so the chronological walk is correct and the
    # caller's DataFrame is not mutated. mergesort is stable (ties keep input order).
    ordered = matches.sort_values(date_col, kind="mergesort").reset_index(drop=True)

    # Ensure a per-row K-factor exists; prefer the cleaned importance weight if present.
    if k_col not in ordered.columns:                            # K not pre-assigned
        if importance_col in ordered.columns:                   # cleaned task 1.3 weight
            ordered[k_col] = ordered[importance_col].astype(int)  # use upstream K weights
        else:                                                   # no cleaned weight available
            ordered[k_col] = assign_k_factors(ordered)          # derive from tournament

    tracker = EloTracker(round_changes=round_changes)           # fresh rating book
    column_positions = {                                        # map columns to tuple slots
        column_name: position                                   # store the integer position
        for position, column_name in enumerate(ordered.columns) # enumerate final columns
    }

    # Pre-allocate output columns as Python lists for speed, then attach at the end.
    home_pre_list: list[float] = []                             # home Elo before each match
    away_pre_list: list[float] = []                             # away Elo before each match
    home_post_list: list[float] = []                            # home Elo after each match
    away_post_list: list[float] = []                            # away Elo after each match
    expectancy_list: list[float] = []                           # home win expectancy

    # itertuples(name=None) is faster and handles arbitrary column names safely.
    for row in ordered.itertuples(index=False, name=None):      # iterate in date order
        result = tracker.update_match(                          # apply one match update
            home_team=row[column_positions[home_col]],          # home team name
            away_team=row[column_positions[away_col]],          # away team name
            home_goals=int(row[column_positions[home_goals_col]]),  # home goals
            away_goals=int(row[column_positions[away_goals_col]]),  # away goals
            k_factor=float(row[column_positions[k_col]]),       # match K-factor
            neutral=bool(row[column_positions[neutral_col]]),   # neutral-venue flag
        )
        home_pre_list.append(result["home_elo_pre"])            # collect pre/post values
        away_pre_list.append(result["away_elo_pre"])           # for vectorised attach
        home_post_list.append(result["home_elo_post"])
        away_post_list.append(result["away_elo_post"])
        expectancy_list.append(result["home_win_expectancy"])

    # Attach the collected columns in one shot (avoids slow per-row DataFrame writes).
    ordered["home_elo_pre"] = home_pre_list                     # home rating before
    ordered["away_elo_pre"] = away_pre_list                     # away rating before
    ordered["home_elo_post"] = home_post_list                   # home rating after
    ordered["away_elo_post"] = away_post_list                   # away rating after
    ordered["home_win_expectancy"] = expectancy_list            # home win expectancy

    return ordered                                              # the enriched table
