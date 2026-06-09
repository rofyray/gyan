"""FIFA 2026 group-stage and best-third-place tiebreakers."""

from __future__ import annotations  # modern type hints

import numpy as np  # random draws for lots
import pandas as pd  # standings and match tables

from gyan.config import N_BEST_THIRDS, POINTS_DRAW, POINTS_LOSS, POINTS_WIN  # constants


def group_standings(matches: pd.DataFrame, teams: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Return a resolved FIFA-order group table from played group matches.

    Parameters
    ----------
    matches : pandas.DataFrame
        Group matches with home, away, home_goals, away_goals.
    teams : list[str]
        Teams in the group.
    rng : numpy.random.Generator
        RNG used only for drawing lots when all previous criteria tie.

    Returns
    -------
    pandas.DataFrame
        Standings sorted from first to fourth with ranking columns.
    """
    table = _base_stats(matches, teams)  # points, goal difference, goals scored
    ordered_parts = _resolve_tied_blocks(table, matches, rng)  # recursively resolve ties
    ordered = pd.concat(ordered_parts, ignore_index=True)  # final resolved ordering
    ordered["group_rank"] = range(1, len(ordered) + 1)  # explicit final rank
    return ordered  # resolved table


def rank_best_thirds(third_rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rank all third-placed teams and mark the top eight qualifiers.

    Parameters
    ----------
    third_rows : pandas.DataFrame
        One third-placed row per group with points, goal_diff, goals_for, fair_play_points.
    rng : numpy.random.Generator
        RNG for drawing lots if all ranking criteria tie.

    Returns
    -------
    pandas.DataFrame
        Ranked third-place table with `third_rank` and `qualified`.
    """
    ranked_blocks: list[pd.DataFrame] = []  # collect resolved tie blocks
    sort_cols = ["points", "goal_diff", "goals_for", "fair_play_points"]  # FIFA best-third order
    sorted_rows = third_rows.sort_values(sort_cols, ascending=[False, False, False, True])  # primary sort
    for _, block in sorted_rows.groupby(sort_cols, sort=False, dropna=False):  # unresolved exact ties
        block = block.copy()  # avoid mutating source
        block["lot_order"] = rng.permutation(len(block))  # lots only within exact ties
        ranked_blocks.append(block.sort_values("lot_order"))  # append resolved block
    ranked = pd.concat(ranked_blocks, ignore_index=True)  # resolved ranking
    ranked["third_rank"] = range(1, len(ranked) + 1)  # overall third-place rank
    ranked["qualified"] = ranked["third_rank"] <= N_BEST_THIRDS  # top eight advance
    return ranked.drop(columns=["lot_order"], errors="ignore")  # output without helper column


def _base_stats(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Compute base group statistics for each team."""
    stats = {  # initialise all teams at zero
        team: {"team": team, "points": 0, "goal_diff": 0, "goals_for": 0, "fair_play_points": 0}
        for team in teams
    }
    for row in matches.itertuples(index=False):  # process each played group match
        home_goals = int(row.home_goals)  # observed home goals
        away_goals = int(row.away_goals)  # observed away goals
        stats[row.home]["goals_for"] += home_goals  # home goals scored
        stats[row.away]["goals_for"] += away_goals  # away goals scored
        stats[row.home]["goal_diff"] += home_goals - away_goals  # home goal difference
        stats[row.away]["goal_diff"] += away_goals - home_goals  # away goal difference
        if home_goals > away_goals:  # home win
            stats[row.home]["points"] += POINTS_WIN  # winner gets three
            stats[row.away]["points"] += POINTS_LOSS  # loser gets zero
        elif home_goals < away_goals:  # away win
            stats[row.away]["points"] += POINTS_WIN  # winner gets three
            stats[row.home]["points"] += POINTS_LOSS  # loser gets zero
        else:  # draw
            stats[row.home]["points"] += POINTS_DRAW  # both get one
            stats[row.away]["points"] += POINTS_DRAW  # both get one
    return pd.DataFrame(stats.values())  # convert to table


def _resolve_tied_blocks(
    table: pd.DataFrame,
    matches: pd.DataFrame,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    """Resolve standings blocks by FIFA criteria, including head-to-head."""
    resolved: list[pd.DataFrame] = []  # collect ordered blocks
    primary_sorted = table.sort_values(  # points, goal difference, goals scored
        ["points", "goal_diff", "goals_for"],
        ascending=[False, False, False],
    )
    for _, block in primary_sorted.groupby(["points", "goal_diff", "goals_for"], sort=False):  # ties
        if len(block) == 1:  # no tie to resolve
            resolved.append(block.copy())  # append singleton
            continue  # next block
        h2h = _head_to_head_stats(matches, block["team"].tolist())  # tied-team mini table
        h2h_joined = block.merge(h2h, on="team", how="left")  # attach h2h criteria
        h2h_sorted = h2h_joined.sort_values(  # h2h points, GD, goals
            ["h2h_points", "h2h_goal_diff", "h2h_goals_for", "fair_play_points"],
            ascending=[False, False, False, True],
        )
        for _, h2h_block in h2h_sorted.groupby(  # exact ties after h2h/fair play
            ["h2h_points", "h2h_goal_diff", "h2h_goals_for", "fair_play_points"],
            sort=False,
        ):
            h2h_block = h2h_block.copy()  # avoid pandas view pitfalls
            if len(h2h_block) > 1:  # drawing of lots required
                h2h_block["lot_order"] = rng.permutation(len(h2h_block))  # random lot order
                h2h_block = h2h_block.sort_values("lot_order")  # resolve exact tie
            resolved.append(h2h_block[table.columns].copy())  # drop helper h2h columns
    return resolved  # ordered block list


def _head_to_head_stats(matches: pd.DataFrame, tied_teams: list[str]) -> pd.DataFrame:
    """Compute mini-table stats among tied teams only."""
    tied_set = set(tied_teams)  # fast membership checks
    h2h_matches = matches[matches["home"].isin(tied_set) & matches["away"].isin(tied_set)]  # tied only
    stats = {  # initialise mini-table
        team: {"team": team, "h2h_points": 0, "h2h_goal_diff": 0, "h2h_goals_for": 0}
        for team in tied_teams
    }
    for row in h2h_matches.itertuples(index=False):  # process tied-team matches
        home_goals = int(row.home_goals)  # home goals
        away_goals = int(row.away_goals)  # away goals
        stats[row.home]["h2h_goals_for"] += home_goals  # home h2h goals
        stats[row.away]["h2h_goals_for"] += away_goals  # away h2h goals
        stats[row.home]["h2h_goal_diff"] += home_goals - away_goals  # home h2h GD
        stats[row.away]["h2h_goal_diff"] += away_goals - home_goals  # away h2h GD
        if home_goals > away_goals:  # home h2h win
            stats[row.home]["h2h_points"] += POINTS_WIN  # three points
        elif home_goals < away_goals:  # away h2h win
            stats[row.away]["h2h_points"] += POINTS_WIN  # three points
        else:  # h2h draw
            stats[row.home]["h2h_points"] += POINTS_DRAW  # one point
            stats[row.away]["h2h_points"] += POINTS_DRAW  # one point
    return pd.DataFrame(stats.values())  # mini-table output
