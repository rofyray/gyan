"""Monte-Carlo tournament simulator for the 2026 World Cup."""

from __future__ import annotations  # modern type hints

import json  # load bracket pairings
from pathlib import Path  # typed paths

import numpy as np  # random generators and array aggregation
import pandas as pd  # schedule/group tables

from gyan.config import DIXON_COLES_MAX_GOALS  # score matrix truncation
from gyan.engine.dixon_coles import DixonColesModel  # Stage 1 fitted engine
from gyan.simulation.sample import sample_knockout_scoreline, sample_scorelines  # score sampling
from gyan.simulation.structure import venue_is_host_home  # home-field venue logic


STAGE_COLUMNS: tuple[str, ...] = (  # advancement output columns
    "R32", "R16", "QF", "SF", "final", "champion",
)


def load_structure(groups_path: Path | str, schedule_path: Path | str, bracket_path: Path | str) -> dict[str, object]:
    """Load processed Stage 2 structure artifacts.

    Parameters
    ----------
    groups_path, schedule_path, bracket_path : Path | str
        Processed structure files from Stage 2.1.

    Returns
    -------
    dict[str, object]
        Structure payload used by the tournament simulator.
    """
    groups = pd.read_parquet(groups_path)  # load group table
    schedule = pd.read_parquet(schedule_path)  # load fixture table
    bracket = json.loads(Path(bracket_path).read_text(encoding="utf-8"))  # load bracket JSON
    return {"groups": groups, "schedule": schedule, "bracket": bracket}  # bundled structure


def prepare_simulation_inputs(
    model: DixonColesModel,
    structure: dict[str, object],
    elo_ratings: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Precompute fixture matrices and lookups shared by tournament simulations."""
    groups = structure["groups"]  # group table
    schedule = structure["schedule"]  # schedule table
    bracket = structure["bracket"]  # bracket mapping
    elo_like = _team_strength_lookup(model, groups["team"].tolist(), elo_ratings)  # Elo for penalties/upsets
    group_matches = schedule[schedule["stage"] == "group"].copy()  # group-stage fixtures
    group_matrices = {  # precompute regulation score matrices for every group match
        int(row.match_id): model.predict_fixture(row.home, row.away, bool(row.neutral))
        for row in group_matches.itertuples(index=False)
    }
    group_lookup = {group: group_frame["team"].tolist() for group, group_frame in groups.groupby("group")}  # teams
    group_simulation = _prepare_group_simulation(group_lookup, group_matches, group_matrices)  # compact group data
    r32_pairings = {int(row["match_id"]): row for row in bracket["round_of_32"]}  # R32 match map
    tree_pairings = {int(row["match_id"]): row for row in bracket["knockout_tree"]}  # later rounds
    schedule_by_match = {int(row.match_id): row for row in schedule.itertuples(index=False)}  # venue lookup
    return {  # bundled precomputed inputs
        "groups": groups,
        "schedule": schedule,
        "bracket": bracket,
        "model": model,
        "elo_like": elo_like,
        "group_matches": group_matches,
        "group_matrices": group_matrices,
        "group_lookup": group_lookup,
        "group_simulation": group_simulation,
        "r32_pairings": r32_pairings,
        "tree_pairings": tree_pairings,
        "schedule_by_match": schedule_by_match,
    }


def simulate_one_tournament(rng: np.random.Generator, prepared: dict[str, object]) -> dict[str, object]:
    """Simulate one full 104-match tournament.

    Parameters
    ----------
    rng : numpy.random.Generator
        Tournament-level random generator.
    prepared : dict[str, object]
        Output of prepare_simulation_inputs.

    Returns
    -------
    dict[str, object]
        Stage reached per team and knockout upset counters.
    """
    group_results = _simulate_group_stage(rng, prepared)  # play 72 group matches
    qualifiers = _resolve_group_stage(rng, prepared, group_results)  # top two plus best thirds
    stage_reached = {team: "" for team in prepared["groups"]["team"].tolist()}  # initialise stages
    for team in qualifiers["R32"]:  # all 32 knockout qualifiers
        stage_reached[team] = "R32"  # reached Round of 32
    knockout = _simulate_knockouts(rng, prepared, qualifiers, stage_reached)  # play matches 73-104
    group_draws = sum(1 for scores in group_results.values() for home, away in scores if home == away)  # group draws
    return {  # tournament outcome payload
        "stage_reached": stage_reached,
        "group_draws": group_draws,
        "knockout_upsets": knockout["knockout_upsets"],
        "knockout_decisive_matches": knockout["knockout_decisive_matches"],
        "knockout_matches": knockout["knockout_matches"],
        "knockout_to_extra_time": knockout["knockout_to_extra_time"],
        "knockout_to_penalties": knockout["knockout_to_penalties"],
    }


def run_tournaments_for_indices(sim_indices: list[int], seed: int, prepared: dict[str, object]) -> dict[str, object]:
    """Run a deterministic chunk of tournament simulations by global indices.

    Parameters
    ----------
    sim_indices : list[int]
        Global simulation indices assigned to this worker.
    seed : int
        Master seed.
    prepared : dict[str, object]
        Precomputed structure/model inputs.

    Returns
    -------
    dict[str, object]
        Stage-count and upset-count aggregates for the chunk.
    """
    teams = prepared["groups"]["team"].tolist()  # stable team order
    counts = {team: {stage: 0 for stage in STAGE_COLUMNS} for team in teams}  # stage counts
    knockout_upsets = 0  # chunk upset numerator
    knockout_decisive = 0  # chunk upset denominator
    group_draws = 0  # chunk group-stage draw count
    knockout_matches = 0  # chunk non-third-place knockout matches
    knockout_to_extra_time = 0  # chunk knockout matches level after regulation
    knockout_to_penalties = 0  # chunk knockout matches level after extra time
    champion_trace: list[str] = []  # champion per simulation for convergence
    for sim_index in sim_indices:  # run each assigned simulation index
        rng = np.random.default_rng(np.random.SeedSequence([seed, sim_index]))  # worker-count-stable RNG
        outcome = simulate_one_tournament(rng, prepared)  # simulate full tournament
        champion = None  # filled when stage reached is champion
        for team, reached in outcome["stage_reached"].items():  # aggregate reached stages
            reached_order = _stage_reach_flags(reached)  # flags through terminal stage
            for stage in reached_order:  # increment each reached stage
                counts[team][stage] += 1  # count one simulated reach
            if reached == "champion":  # champion row
                champion = team  # store champion
        champion_trace.append(champion)  # convergence trace
        group_draws += int(outcome["group_draws"])  # add group draw count
        knockout_upsets += int(outcome["knockout_upsets"])  # add upset numerator
        knockout_decisive += int(outcome["knockout_decisive_matches"])  # add denominator
        knockout_matches += int(outcome["knockout_matches"])  # add knockout denominator
        knockout_to_extra_time += int(outcome["knockout_to_extra_time"])  # add ET count
        knockout_to_penalties += int(outcome["knockout_to_penalties"])  # add penalty count
    return {  # chunk aggregate payload
        "counts": counts,
        "group_draws": group_draws,
        "knockout_upsets": knockout_upsets,
        "knockout_decisive_matches": knockout_decisive,
        "knockout_matches": knockout_matches,
        "knockout_to_extra_time": knockout_to_extra_time,
        "knockout_to_penalties": knockout_to_penalties,
        "champion_trace": champion_trace,
    }


def aggregate_chunks(chunks: list[dict[str, object]], teams: list[str], n_sims: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate worker chunks into advancement probabilities and metrics."""
    counts = {team: {stage: 0 for stage in STAGE_COLUMNS} for team in teams}  # total counts
    champion_trace: list[str] = []  # ordered champion trace by chunk order
    knockout_upsets = 0  # total upset numerator
    knockout_decisive = 0  # total upset denominator
    group_draws = 0  # total group draws
    knockout_matches = 0  # total non-third-place knockout matches
    knockout_to_extra_time = 0  # total matches reaching extra time
    knockout_to_penalties = 0  # total matches reaching penalties
    for chunk in chunks:  # merge each chunk
        for team in teams:  # merge team stage counts
            for stage in STAGE_COLUMNS:  # each output stage
                counts[team][stage] += int(chunk["counts"][team][stage])  # add count
        champion_trace.extend(chunk["champion_trace"])  # append champion sequence
        group_draws += int(chunk["group_draws"])  # add group draws
        knockout_upsets += int(chunk["knockout_upsets"])  # add upsets
        knockout_decisive += int(chunk["knockout_decisive_matches"])  # add denominator
        knockout_matches += int(chunk["knockout_matches"])  # add knockout matches
        knockout_to_extra_time += int(chunk["knockout_to_extra_time"])  # add ET matches
        knockout_to_penalties += int(chunk["knockout_to_penalties"])  # add shootouts
    rows = []  # collect probability rows
    for team in teams:  # one row per team
        rows.append(  # probability row in PRD schema
            {
                "team": team,
                "p_reach_R32": counts[team]["R32"] / n_sims,
                "p_reach_R16": counts[team]["R16"] / n_sims,
                "p_reach_QF": counts[team]["QF"] / n_sims,
                "p_reach_SF": counts[team]["SF"] / n_sims,
                "p_reach_final": counts[team]["final"] / n_sims,
                "p_champion": counts[team]["champion"] / n_sims,
            }
        )
    probabilities = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)  # rank
    metrics = {  # run-level metrics
        "knockout_upsets": knockout_upsets,
        "knockout_decisive_matches": knockout_decisive,
        "knockout_upset_rate": knockout_upsets / knockout_decisive if knockout_decisive else 0.0,
        "group_draws": group_draws,
        "group_draw_rate": group_draws / (72 * n_sims),
        "knockout_matches": knockout_matches,
        "knockout_to_extra_time": knockout_to_extra_time,
        "knockout_to_extra_time_rate": knockout_to_extra_time / knockout_matches if knockout_matches else 0.0,
        "knockout_to_penalties": knockout_to_penalties,
        "knockout_to_penalties_rate": knockout_to_penalties / knockout_matches if knockout_matches else 0.0,
        "champion_trace": champion_trace,
    }
    return probabilities, metrics  # final table and metrics


def validate_probabilities(probabilities: pd.DataFrame) -> dict[str, object]:
    """Validate advancement-probability invariants."""
    stage_cols = ["p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_final", "p_champion"]  # cols
    in_range = bool(((probabilities[stage_cols] >= 0.0) & (probabilities[stage_cols] <= 1.0)).all().all())  # bounds
    monotone = bool((probabilities[stage_cols].diff(axis=1).iloc[:, 1:] <= 1e-12).all().all())  # decreasing
    champion_sum = float(probabilities["p_champion"].sum())  # champion probabilities total
    return {  # invariant metrics
        "probabilities_in_range": in_range,
        "probabilities_monotone": monotone,
        "champion_probability_sum": champion_sum,
    }


def _simulate_group_stage(rng: np.random.Generator, prepared: dict[str, object]) -> dict[str, list[tuple[int, int]]]:
    """Simulate all group-stage matches for one tournament."""
    return {  # group -> six sampled scorelines
        group: sample_scorelines(group_data["matrices"], rng)
        for group, group_data in prepared["group_simulation"].items()
    }


def _resolve_group_stage(
    rng: np.random.Generator,
    prepared: dict[str, object],
    group_results: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    """Resolve group tables, best thirds, and R32 qualifiers."""
    winners: dict[str, str] = {}  # group -> winner
    runners_up: dict[str, str] = {}  # group -> runner-up
    thirds: dict[str, str] = {}  # group -> third-place team
    third_rows: list[dict[str, object]] = []  # third-place stats rows
    for group, group_data in prepared["group_simulation"].items():  # resolve each group
        standings = _rank_group_fast(group_data, group_results[group], rng)  # FIFA tiebreak order
        winners[group] = standings[0]["team"]  # group winner
        runners_up[group] = standings[1]["team"]  # group runner-up
        thirds[group] = standings[2]["team"]  # third-place team
        third = dict(standings[2])  # third-place stats
        third["group"] = group  # include group label
        third_rows.append(third)  # collect for best-third table
    third_table = _rank_best_thirds_fast(third_rows, rng)  # rank twelve third teams
    qualified_third_groups = sorted(row["group"] for row in third_table[:8])  # group labels
    qualified_thirds = {group: thirds[group] for group in qualified_third_groups}  # group -> team
    combination_key = "".join(qualified_third_groups)  # Annex C key for this set of third-place qualifiers
    third_assignments = prepared["bracket"]["third_place_combinations"][combination_key]  # destination -> group
    r32_teams = list(winners.values()) + list(runners_up.values()) + list(qualified_thirds.values())  # 32
    return {  # group-stage resolution payload
        "winners": winners,
        "runners_up": runners_up,
        "thirds": thirds,
        "qualified_third_groups": qualified_third_groups,
        "qualified_thirds": qualified_thirds,
        "third_assignments": third_assignments,
        "R32": r32_teams,
    }


def _simulate_knockouts(
    rng: np.random.Generator,
    prepared: dict[str, object],
    qualifiers: dict[str, object],
    stage_reached: dict[str, str],
) -> dict[str, int]:
    """Simulate the fixed knockout tree from R32 through final."""
    winners: dict[int, str] = {}  # match id -> winner team
    losers: dict[int, str] = {}  # match id -> loser team
    knockout_upsets = 0  # lower-Elo winner count
    knockout_decisive = 0  # non-equal Elo knockout count
    knockout_matches = 0  # non-third-place knockout matches
    knockout_to_extra_time = 0  # non-third-place matches level after regulation
    knockout_to_penalties = 0  # non-third-place matches decided by shootout
    for match_id in range(73, 105):  # all knockout and third-place matches
        pairing = _pairing_for_match(match_id, prepared)  # fixed home/away slots
        home = _resolve_slot(pairing["home_slot"], qualifiers, winners, losers)  # home team
        away = _resolve_slot(pairing["away_slot"], qualifiers, winners, losers)  # away team
        schedule_row = prepared["schedule_by_match"][match_id]  # concrete venue for this match
        neutral = not venue_is_host_home(home, schedule_row.venue)  # host home-field only when listed first
        model = prepared["model"]  # fitted Stage 1 model
        lam, mu = model.fixture_means(home, away, neutral)  # regulation means
        rho = 0.0 if model.selected_engine == "plain_poisson" else model.rho  # T-G3 selected engine
        result = sample_knockout_scoreline(  # play knockout to a winner
            lam,
            mu,
            rho,
            prepared["elo_like"][home],
            prepared["elo_like"][away],
            rng,
            DIXON_COLES_MAX_GOALS,
            model.score_distribution,
            model.score_dispersion,
        )
        winner = home if result["winner"] == "home" else away  # team winner
        loser = away if result["winner"] == "home" else home  # team loser
        winners[match_id] = winner  # store winner
        losers[match_id] = loser  # store loser
        if match_id != 103:  # third-place playoff does not affect advancement outputs
            _mark_stage_reached(match_id, winner, stage_reached)  # mark next reached stage for winner
            knockout_matches += 1  # count advancement knockout matches
            knockout_to_extra_time += int(bool(result["regulation_draw"]))  # level after regulation
            knockout_to_penalties += int(result["decided_by"] == "penalties")  # shootout count
        if prepared["elo_like"][home] != prepared["elo_like"][away] and match_id != 103:  # measurable upset
            knockout_decisive += 1  # denominator
            stronger = home if prepared["elo_like"][home] > prepared["elo_like"][away] else away  # stronger team
            if winner != stronger:  # lower-rated team won
                knockout_upsets += 1  # numerator
    return {  # metrics
        "knockout_upsets": knockout_upsets,
        "knockout_decisive_matches": knockout_decisive,
        "knockout_matches": knockout_matches,
        "knockout_to_extra_time": knockout_to_extra_time,
        "knockout_to_penalties": knockout_to_penalties,
    }


def _pairing_for_match(match_id: int, prepared: dict[str, object]) -> dict[str, object]:
    """Return the fixed pairing record for a knockout match id."""
    if match_id <= 88:  # R32
        return prepared["r32_pairings"][match_id]  # R32 pairing map
    return prepared["tree_pairings"][match_id]  # later knockout tree map


def _resolve_slot(
    slot: str,
    qualifiers: dict[str, object],
    winners: dict[int, str],
    losers: dict[int, str],
) -> str:
    """Resolve a bracket slot string to a concrete team."""
    if slot.startswith("Winner Group "):  # group winner slot
        group = slot.replace("Winner Group ", "").strip()  # group letter
        return qualifiers["winners"][group]  # winning team
    if slot.startswith("Runner-up Group "):  # group runner-up slot
        group = slot.replace("Runner-up Group ", "").strip()  # group letter
        return qualifiers["runners_up"][group]  # runner-up team
    if slot.startswith("3rd Group "):  # third-place assignment slot
        candidate_groups = slot.replace("3rd Group ", "").replace("/", "")  # candidate letters
        destination_group = _destination_group_for_third_slot(slot, qualifiers)  # group
        assert destination_group in candidate_groups  # Annex C assignment must be eligible
        return qualifiers["qualified_thirds"][destination_group]  # concrete third-place team
    if slot.startswith("Winner Match "):  # prior match winner slot
        match_id = int(slot.replace("Winner Match ", ""))  # referenced match id
        return winners[match_id]  # team that won referenced match
    if slot.startswith("Loser Match "):  # prior match loser slot
        match_id = int(slot.replace("Loser Match ", ""))  # referenced match id
        return losers[match_id]  # team that lost referenced match
    raise KeyError(f"Unknown bracket slot: {slot}")  # fail loudly on unknown slot


def _destination_group_for_third_slot(
    slot: str,
    qualifiers: dict[str, object],
) -> str:
    """Return the third-place group assigned to a specific R32 slot."""
    slot_to_winner_group = {  # candidate slot text -> winner group destination
        "3rd Group A/B/C/D/F": "E",
        "3rd Group C/D/F/G/H": "I",
        "3rd Group C/E/F/H/I": "A",
        "3rd Group E/H/I/J/K": "L",
        "3rd Group A/E/H/I/J": "G",
        "3rd Group B/E/F/I/J": "D",
        "3rd Group E/F/G/I/J": "B",
        "3rd Group D/E/I/J/L": "K",
    }
    winner_group = slot_to_winner_group[slot]  # bracket destination group
    return qualifiers["third_assignments"][winner_group]  # Annex C-assigned third-place group


def _mark_stage_reached(match_id: int, winner: str, stage_reached: dict[str, str]) -> None:
    """Mark the stage a knockout winner reached after winning a match."""
    if match_id <= 88:  # R32 win reaches R16
        stage_reached[winner] = "R16"  # reached Round of 16
    elif match_id <= 96:  # R16 win reaches QF
        stage_reached[winner] = "QF"  # reached quarterfinal
    elif match_id <= 100:  # QF win reaches SF
        stage_reached[winner] = "SF"  # reached semifinal
    elif match_id <= 102:  # SF win reaches final
        stage_reached[winner] = "final"  # reached final
    elif match_id == 104:  # final win becomes champion
        stage_reached[winner] = "champion"  # champion


def _stage_reach_flags(reached: str) -> list[str]:
    """Return all output stages reached through a terminal stage label."""
    if not reached:  # eliminated in group
        return []  # no output-stage reach
    order = list(STAGE_COLUMNS)  # ordered output stages
    return order[: order.index(reached) + 1]  # stages reached up to terminal stage


def _team_strength_lookup(
    model: DixonColesModel,
    teams: list[str],
    elo_ratings: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return current Elo ratings for penalty/upset comparisons."""
    if elo_ratings is not None and {"team", "elo_rating"}.issubset(elo_ratings.columns):  # real Elo table
        rating_lookup = dict(zip(elo_ratings["team"], elo_ratings["elo_rating"], strict=False))  # team -> Elo
        missing = [team for team in teams if team not in rating_lookup or pd.isna(rating_lookup[team])]  # gaps
        if missing:  # every tournament team needs a real current Elo value
            raise ValueError(f"Missing current Elo ratings for tournament teams: {missing}")  # fail loudly
        return {team: float(rating_lookup[team]) for team in teams}  # use current Elo
    return {  # combine attack and defense into one scalar only when no Elo table is supplied
        team: float((model.attack.get(team, 0.0) + model.defense.get(team, 0.0)) * 400.0 + 1500.0)
        for team in teams
    }


def _prepare_group_simulation(
    group_lookup: dict[str, list[str]],
    group_matches: pd.DataFrame,
    group_matrices: dict[int, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Return compact per-group arrays for the hot simulation path."""
    prepared: dict[str, dict[str, object]] = {}  # group -> compact fixture payload
    for group, teams in group_lookup.items():  # one group at a time
        team_index = {team: index for index, team in enumerate(teams)}  # local team ids
        fixtures = group_matches[group_matches["group"] == group].sort_values("match_id")  # six fixtures
        prepared[group] = {  # compact data used every tournament draw
            "teams": teams,
            "home_idx": [team_index[row.home] for row in fixtures.itertuples(index=False)],
            "away_idx": [team_index[row.away] for row in fixtures.itertuples(index=False)],
            "matrices": [group_matrices[int(row.match_id)] for row in fixtures.itertuples(index=False)],
        }
    return prepared  # group simulation payload


def _rank_group_fast(
    group_data: dict[str, object],
    scores: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Rank one four-team group without pandas in the simulation hot path."""
    teams = group_data["teams"]  # local team labels
    home_idx = group_data["home_idx"]  # six home local ids
    away_idx = group_data["away_idx"]  # six away local ids
    points = np.zeros(4, dtype=np.int16)  # group points
    goal_diff = np.zeros(4, dtype=np.int16)  # group goal difference
    goals_for = np.zeros(4, dtype=np.int16)  # group goals scored
    for match_index, (home_goals, away_goals) in enumerate(scores):  # six matches
        home = home_idx[match_index]  # local home id
        away = away_idx[match_index]  # local away id
        goals_for[home] += home_goals  # home goals
        goals_for[away] += away_goals  # away goals
        goal_diff[home] += home_goals - away_goals  # home GD
        goal_diff[away] += away_goals - home_goals  # away GD
        if home_goals > away_goals:  # home win
            points[home] += 3  # three points
        elif home_goals < away_goals:  # away win
            points[away] += 3  # three points
        else:  # draw
            points[home] += 1  # one point
            points[away] += 1  # one point
    base_order = sorted(range(4), key=lambda idx: (-points[idx], -goal_diff[idx], -goals_for[idx]))  # primary
    ordered: list[int] = []  # final local ids
    position = 0  # walk base-order tie blocks
    while position < len(base_order):  # each primary block
        block = [base_order[position]]  # first member
        key = (points[block[0]], goal_diff[block[0]], goals_for[block[0]])  # primary key
        position += 1  # advance
        while position < len(base_order):  # collect same primary key
            candidate = base_order[position]  # candidate local id
            if (points[candidate], goal_diff[candidate], goals_for[candidate]) != key:  # key changed
                break  # block complete
            block.append(candidate)  # same primary block
            position += 1  # advance
        ordered.extend(_resolve_group_block_fast(block, home_idx, away_idx, scores, rng))  # h2h/lots
    return [  # convert local ids to stat rows
        {
            "team": teams[idx],
            "points": int(points[idx]),
            "goal_diff": int(goal_diff[idx]),
            "goals_for": int(goals_for[idx]),
            "fair_play_points": 0,
        }
        for idx in ordered
    ]


def _resolve_group_block_fast(
    block: list[int],
    home_idx: list[int],
    away_idx: list[int],
    scores: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[int]:
    """Resolve a tied primary group block by head-to-head, fair play, and lots."""
    if len(block) == 1:  # no tie
        return block  # singleton block
    block_position = {team_idx: index for index, team_idx in enumerate(block)}  # local id -> h2h id
    h2h_points = np.zeros(len(block), dtype=np.int16)  # h2h points
    h2h_goal_diff = np.zeros(len(block), dtype=np.int16)  # h2h GD
    h2h_goals_for = np.zeros(len(block), dtype=np.int16)  # h2h GF
    tied_set = set(block)  # membership check
    for match_index, (home_goals, away_goals) in enumerate(scores):  # group matches
        home = home_idx[match_index]  # local home id
        away = away_idx[match_index]  # local away id
        if home not in tied_set or away not in tied_set:  # only tied-team mini-table
            continue  # skip non-h2h fixture
        hpos = block_position[home]  # h2h home id
        apos = block_position[away]  # h2h away id
        h2h_goals_for[hpos] += home_goals  # h2h home GF
        h2h_goals_for[apos] += away_goals  # h2h away GF
        h2h_goal_diff[hpos] += home_goals - away_goals  # h2h home GD
        h2h_goal_diff[apos] += away_goals - home_goals  # h2h away GD
        if home_goals > away_goals:  # h2h home win
            h2h_points[hpos] += 3  # three points
        elif home_goals < away_goals:  # h2h away win
            h2h_points[apos] += 3  # three points
        else:  # h2h draw
            h2h_points[hpos] += 1  # one point
            h2h_points[apos] += 1  # one point
    h2h_order = sorted(  # sort by h2h criteria; fair-play points are all zero in engine-only simulation
        range(len(block)),
        key=lambda idx: (-h2h_points[idx], -h2h_goal_diff[idx], -h2h_goals_for[idx]),
    )
    ordered: list[int] = []  # final tied block order
    position = 0  # walk exact h2h blocks
    while position < len(h2h_order):  # h2h tie blocks
        block_ids = [h2h_order[position]]  # first h2h member
        key = (h2h_points[block_ids[0]], h2h_goal_diff[block_ids[0]], h2h_goals_for[block_ids[0]])  # key
        position += 1  # advance
        while position < len(h2h_order):  # collect exact h2h ties
            candidate = h2h_order[position]  # candidate h2h id
            if (h2h_points[candidate], h2h_goal_diff[candidate], h2h_goals_for[candidate]) != key:  # key changed
                break  # exact block complete
            block_ids.append(candidate)  # exact tie
            position += 1  # advance
        if len(block_ids) > 1:  # drawing of lots
            permutation = rng.permutation(len(block_ids)).tolist()  # random order within exact tie
            block_ids = [block_ids[index] for index in permutation]  # apply lots
        ordered.extend(block[h2h_id] for h2h_id in block_ids)  # convert h2h id -> local team id
    return ordered  # resolved tied block


def _rank_best_thirds_fast(third_rows: list[dict[str, object]], rng: np.random.Generator) -> list[dict[str, object]]:
    """Rank third-placed teams without pandas in the simulation hot path."""
    ordered_rows = sorted(  # primary best-third sort
        third_rows,
        key=lambda row: (-int(row["points"]), -int(row["goal_diff"]), -int(row["goals_for"]), int(row["fair_play_points"])),
    )
    ranked: list[dict[str, object]] = []  # final ranking
    position = 0  # walk exact tie blocks
    while position < len(ordered_rows):  # rank all rows
        block = [ordered_rows[position]]  # first member
        key = (
            int(block[0]["points"]),
            int(block[0]["goal_diff"]),
            int(block[0]["goals_for"]),
            int(block[0]["fair_play_points"]),
        )
        position += 1  # advance
        while position < len(ordered_rows):  # collect exact ties
            candidate = ordered_rows[position]  # next candidate
            candidate_key = (
                int(candidate["points"]),
                int(candidate["goal_diff"]),
                int(candidate["goals_for"]),
                int(candidate["fair_play_points"]),
            )
            if candidate_key != key:  # key changed
                break  # block complete
            block.append(candidate)  # exact tie
            position += 1  # advance
        if len(block) > 1:  # drawing of lots
            permutation = rng.permutation(len(block)).tolist()  # random order
            block = [block[index] for index in permutation]  # apply lots
        ranked.extend(block)  # append resolved block
    return ranked  # ordered third-place rows
