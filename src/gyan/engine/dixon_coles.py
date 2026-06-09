"""Dixon-Coles bivariate Poisson goal model for the Stage 1 Goal expert."""

from __future__ import annotations  # modern type hints on all supported Python versions

import json  # persist and load fitted parameter files
from dataclasses import dataclass  # compact fitted-model container
from pathlib import Path  # typed filesystem paths

import numpy as np  # vectorised likelihood and matrix maths
import pandas as pd  # input match tables and parameter tables
from scipy.optimize import minimize  # L-BFGS-B optimiser for weighted likelihood
from scipy.special import gammaln  # stable log-factorial term for Poisson likelihood
from scipy.stats import poisson  # scoreline matrix Poisson PMFs

from gyan.config import (  # modelling constants live in config.py
    DEFAULT_SCORE_DISTRIBUTION,
    DIXON_COLES_MAX_GOALS,
    DIXON_COLES_MAXITER,
    DIXON_COLES_RHO_BOUNDS,
    DIXON_COLES_RIDGE,
    DIXON_COLES_XI,
    NEGATIVE_BINOMIAL_DISPERSION,
    SCORE_DISTRIBUTION_CORRELATED_NB,
    SCORE_DISTRIBUTION_POISSON,
)


def dixon_coles_tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """Return the Dixon-Coles low-score correction factor tau(x, y).

    Parameters
    ----------
    home_goals, away_goals : int
        Scoreline cell being corrected.
    lam, mu : float
        Home and away Poisson means.
    rho : float
        Dixon-Coles low-score dependence parameter.

    Returns
    -------
    float
        Multiplicative tau correction for the scoreline cell.
    """
    if home_goals == 0 and away_goals == 0:  # 0-0 cell
        return 1.0 - lam * mu * rho  # Dixon-Coles correction
    if home_goals == 0 and away_goals == 1:  # 0-1 cell
        return 1.0 + lam * rho  # Dixon-Coles correction
    if home_goals == 1 and away_goals == 0:  # 1-0 cell
        return 1.0 + mu * rho  # Dixon-Coles correction
    if home_goals == 1 and away_goals == 1:  # 1-1 cell
        return 1.0 - rho  # Dixon-Coles correction
    return 1.0  # all other cells are unchanged


def time_decay_weight(days_before_ref: np.ndarray, xi: float) -> np.ndarray:
    """Return exponential time-decay weights phi(t) = exp(-xi * t).

    Parameters
    ----------
    days_before_ref : numpy.ndarray
        Days before the fitting reference date.
    xi : float
        Dixon-Coles decay rate.

    Returns
    -------
    numpy.ndarray
        Positive weights, with recent matches largest.
    """
    return np.exp(-xi * days_before_ref)  # vectorised exponential decay


def scoreline_matrix(lam: float, mu: float, rho: float, max_goals: int = DIXON_COLES_MAX_GOALS) -> np.ndarray:
    """Return a normalised scoreline-probability matrix with DC correction.

    Parameters
    ----------
    lam, mu : float
        Home and away Poisson means.
    rho : float
        Dixon-Coles low-score dependence parameter.
    max_goals : int
        Highest goal count retained on each axis.

    Returns
    -------
    numpy.ndarray
        Matrix of shape `(max_goals + 1, max_goals + 1)` that sums to one.
    """
    goal_range = np.arange(max_goals + 1)  # scoreline support from 0 to max_goals
    home_pmf = poisson.pmf(goal_range, lam)  # home marginal PMF
    away_pmf = poisson.pmf(goal_range, mu)  # away marginal PMF
    matrix = np.outer(home_pmf, away_pmf)  # independent Poisson base matrix
    matrix[0, 0] *= dixon_coles_tau(0, 0, lam, mu, rho)  # corrected 0-0 probability
    matrix[0, 1] *= dixon_coles_tau(0, 1, lam, mu, rho)  # corrected 0-1 probability
    matrix[1, 0] *= dixon_coles_tau(1, 0, lam, mu, rho)  # corrected 1-0 probability
    matrix[1, 1] *= dixon_coles_tau(1, 1, lam, mu, rho)  # corrected 1-1 probability
    matrix = np.maximum(matrix, 0.0)  # guard against tiny invalid cells from extreme rho
    return matrix / matrix.sum()  # renormalise after truncation and correction


def correlated_negative_binomial_matrix(
    lam: float,
    mu: float,
    dispersion: float = NEGATIVE_BINOMIAL_DISPERSION,
    max_goals: int = DIXON_COLES_MAX_GOALS,
) -> np.ndarray:
    """Return a shared-frailty negative-binomial scoreline matrix.

    The score means remain the plain-Poisson means. A shared Gamma tempo shock
    adds overdispersion and positive home/away goal correlation.
    """
    if dispersion <= 0.0:  # Gamma shape/rate must be positive
        raise ValueError("negative-binomial dispersion must be positive")
    lam = max(float(lam), 1e-12)  # avoid log(0) while preserving practical zero-goal means
    mu = max(float(mu), 1e-12)  # avoid log(0) while preserving practical zero-goal means
    goal_range = np.arange(max_goals + 1)  # scoreline support from 0 to max_goals
    home_goals = goal_range[:, None]  # home-goal grid
    away_goals = goal_range[None, :]  # away-goal grid
    total = dispersion + lam + mu  # shared denominator after integrating out tempo
    log_matrix = (  # bivariate NB log PMF under shared Gamma frailty
        gammaln(dispersion + home_goals + away_goals)
        - gammaln(dispersion)
        - gammaln(home_goals + 1.0)
        - gammaln(away_goals + 1.0)
        + dispersion * np.log(dispersion / total)
        + home_goals * np.log(lam / total)
        + away_goals * np.log(mu / total)
    )
    matrix = np.exp(log_matrix)  # convert back to probability space
    return matrix / matrix.sum()  # renormalise after truncation


def match_outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Collapse a scoreline matrix to home/draw/away probabilities.

    Parameters
    ----------
    matrix : numpy.ndarray
        Scoreline probability matrix with home goals on rows and away goals on columns.

    Returns
    -------
    tuple[float, float, float]
        `(P_home_win, P_draw, P_away_win)`.
    """
    p_home = float(np.tril(matrix, -1).sum())  # row index > column index means home win
    p_draw = float(np.trace(matrix))  # diagonal cells are draws
    p_away = float(np.triu(matrix, 1).sum())  # column index > row index means away win
    return p_home, p_draw, p_away  # ordered W/D/L probabilities


def ranked_probability_score(probabilities: tuple[float, float, float], outcome_index: int) -> float:
    """Return RPS for ordered W/D/L probabilities and an observed outcome.

    Parameters
    ----------
    probabilities : tuple[float, float, float]
        Ordered probabilities `(home win, draw, away win)`.
    outcome_index : int
        Observed outcome index: 0 home win, 1 draw, 2 away win.

    Returns
    -------
    float
        Ranked Probability Score with denominator K - 1 for K=3 categories.
    """
    forecast_cdf = np.cumsum(np.asarray(probabilities, dtype=float))  # forecast cumulative probs
    observed = np.zeros(3, dtype=float)  # one-hot outcome vector
    observed[outcome_index] = 1.0  # set observed outcome category
    observed_cdf = np.cumsum(observed)  # observed cumulative vector
    return float(np.mean((forecast_cdf[:-1] - observed_cdf[:-1]) ** 2))  # K-1 denominator


def observed_outcome_index(home_goals: int, away_goals: int) -> int:
    """Return ordered W/D/L outcome index from a scoreline.

    Parameters
    ----------
    home_goals, away_goals : int
        Observed final goals.

    Returns
    -------
    int
        0 for home win, 1 for draw, 2 for away win.
    """
    if home_goals > away_goals:  # home scored more
        return 0  # home win category
    if home_goals == away_goals:  # equal goals
        return 1  # draw category
    return 2  # away win category


@dataclass
class DixonColesModel:
    """Fitted Dixon-Coles parameter container with prediction helpers."""

    teams: list[str]  # ordered team labels used by parameter arrays
    attack: dict[str, float]  # attack parameter by team
    defense: dict[str, float]  # defense parameter by team
    home_field: float  # non-neutral home-field log-goal effect
    rho: float  # Dixon-Coles dependence parameter
    xi: float  # time-decay value used when fitting
    max_goals: int = DIXON_COLES_MAX_GOALS  # matrix truncation
    selected_engine: str = "dixon_coles"  # selected engine label after validation
    score_distribution: str = SCORE_DISTRIBUTION_POISSON  # scoreline matrix family
    score_dispersion: float = NEGATIVE_BINOMIAL_DISPERSION  # NB shared-frailty shape

    def fixture_means(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Return `(lambda_home, lambda_away)` for a fixture.

        Parameters
        ----------
        home, away : str
            Team labels in the fitted parameter map.
        neutral : bool
            True for neutral venue, False for genuine home venue.

        Returns
        -------
        tuple[float, float]
            Poisson means for home and away goals.
        """
        home_attack = self.attack.get(home, 0.0)  # unknown teams receive average attack
        away_attack = self.attack.get(away, 0.0)  # unknown teams receive average attack
        home_defense = self.defense.get(home, 0.0)  # unknown teams receive average defense
        away_defense = self.defense.get(away, 0.0)  # unknown teams receive average defense
        home_effect = 0.0 if neutral else self.home_field  # home field only at non-neutral venues
        lam = float(np.exp(np.clip(home_effect + home_attack - away_defense, -4.0, 4.0)))  # home mean
        mu = float(np.exp(np.clip(away_attack - home_defense, -4.0, 4.0)))  # away mean
        return lam, mu  # fixture means

    def predict_fixture(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return a scoreline matrix for a fixture.

        Parameters
        ----------
        home, away : str
            Team labels.
        neutral : bool
            True for neutral venue.

        Returns
        -------
        numpy.ndarray
            Dixon-Coles scoreline matrix.
        """
        lam, mu = self.fixture_means(home, away, neutral)  # compute means first
        if self.score_distribution == SCORE_DISTRIBUTION_CORRELATED_NB:  # calibrated overdispersed matrix
            return correlated_negative_binomial_matrix(lam, mu, self.score_dispersion, self.max_goals)  # full matrix
        if self.score_distribution != SCORE_DISTRIBUTION_POISSON:  # persisted config must be known
            raise ValueError(f"Unknown score_distribution: {self.score_distribution}")  # fail loudly
        rho = 0.0 if self.selected_engine == "plain_poisson" else self.rho  # T-G3 fallback support
        return scoreline_matrix(lam, mu, rho, self.max_goals)  # full scoreline distribution

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of fitted parameters."""
        return {  # compact parameter payload
            "teams": self.teams,
            "attack": self.attack,
            "defense": self.defense,
            "home_field": self.home_field,
            "rho": self.rho,
            "xi": self.xi,
            "max_goals": self.max_goals,
            "selected_engine": self.selected_engine,
            "score_distribution": self.score_distribution,
            "score_dispersion": self.score_dispersion,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DixonColesModel":
        """Build a fitted model from a JSON payload."""
        return cls(  # dataclass constructor from stored fields
            teams=list(payload["teams"]),
            attack={str(key): float(value) for key, value in dict(payload["attack"]).items()},
            defense={str(key): float(value) for key, value in dict(payload["defense"]).items()},
            home_field=float(payload["home_field"]),
            rho=float(payload["rho"]),
            xi=float(payload["xi"]),
            max_goals=int(payload.get("max_goals", DIXON_COLES_MAX_GOALS)),
            selected_engine=str(payload.get("selected_engine", "dixon_coles")),
            score_distribution=str(payload.get("score_distribution", SCORE_DISTRIBUTION_POISSON)),
            score_dispersion=float(payload.get("score_dispersion", NEGATIVE_BINOMIAL_DISPERSION)),
        )

    def save(self, path: Path | str, extra: dict[str, object] | None = None) -> None:
        """Write fitted parameters to JSON.

        Parameters
        ----------
        path : Path | str
            Output parameter path.
        extra : dict[str, object] | None
            Optional metadata to merge into the JSON payload.
        """
        payload = self.to_dict()  # base model payload
        if extra is not None:  # caller supplied metadata
            payload.update(extra)  # merge metadata fields
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")  # write JSON

    @classmethod
    def load(cls, path: Path | str) -> "DixonColesModel":
        """Load fitted parameters from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # read JSON payload
        return cls.from_dict(payload)  # construct fitted model


def _prepare_training_arrays(matches: pd.DataFrame, xi: float) -> dict[str, object]:
    """Prepare indexed arrays for likelihood evaluation.

    Parameters
    ----------
    matches : pandas.DataFrame
        Training match table with teams, goals, neutral flag, date, and optional Elo.
    xi : float
        Time-decay parameter.

    Returns
    -------
    dict[str, object]
        Team list and numeric arrays used by the optimiser.
    """
    ordered = matches.sort_values("date", kind="mergesort").reset_index(drop=True)  # chronological order
    teams = sorted(set(ordered["home_team"]).union(set(ordered["away_team"])))  # stable team list
    team_index = {team: index for index, team in enumerate(teams)}  # team -> integer id
    home_idx = ordered["home_team"].map(team_index).to_numpy(dtype=np.int64)  # home team ids
    away_idx = ordered["away_team"].map(team_index).to_numpy(dtype=np.int64)  # away team ids
    home_goals = ordered["home_goals"].to_numpy(dtype=np.int64)  # observed home goals
    away_goals = ordered["away_goals"].to_numpy(dtype=np.int64)  # observed away goals
    neutral = ordered["neutral"].to_numpy(dtype=bool)  # neutral venue flags
    reference_date = pd.Timestamp(ordered["date"].max())  # latest training date
    days_before = (reference_date - pd.to_datetime(ordered["date"])).dt.days.to_numpy(dtype=float)  # ages
    weights = time_decay_weight(days_before, xi) if xi > 0.0 else np.ones(len(ordered), dtype=float)  # weights
    weights = weights / np.mean(weights)  # normalise for comparable optimiser scale
    elo_lookup = _initial_ability_from_elo(ordered, teams)  # Elo-based initial ability vector
    return {  # bundle arrays and metadata
        "ordered": ordered,
        "teams": teams,
        "team_index": team_index,
        "home_idx": home_idx,
        "away_idx": away_idx,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "neutral": neutral,
        "weights": weights,
        "reference_date": str(reference_date.date()),
        "initial_ability": elo_lookup,
    }


def _initial_ability_from_elo(matches: pd.DataFrame, teams: list[str]) -> np.ndarray:
    """Return a centred Elo-derived ability vector for initial parameters."""
    latest: dict[str, float] = {team: 1500.0 for team in teams}  # default Elo-like ability
    for row in matches.itertuples(index=False):  # walk rows to capture latest pre-match Elo
        latest[row.home_team] = float(getattr(row, "home_elo_pre", 1500.0))  # home latest Elo
        latest[row.away_team] = float(getattr(row, "away_elo_pre", 1500.0))  # away latest Elo
    values = np.asarray([latest[team] for team in teams], dtype=float)  # ordered Elo vector
    return (values - values.mean()) / 900.0  # scaled, centred initial ability


def fit_dixon_coles(
    matches: pd.DataFrame,
    xi: float = DIXON_COLES_XI,
    fit_rho: bool = True,
    maxiter: int = DIXON_COLES_MAXITER,
    ridge: float = DIXON_COLES_RIDGE,
) -> tuple[DixonColesModel, dict[str, object]]:
    """Fit a weighted Dixon-Coles model by maximum likelihood.

    Parameters
    ----------
    matches : pandas.DataFrame
        Completed match table with canonical team names and goals.
    xi : float
        Time-decay rate used for match weights.
    fit_rho : bool
        If False, fix rho to zero for a plain-Poisson baseline.
    maxiter : int
        Optimiser iteration limit.
    ridge : float
        Weak L2 penalty for attack/defense parameters.

    Returns
    -------
    tuple[DixonColesModel, dict[str, object]]
        Fitted model and optimiser diagnostics.
    """
    arrays = _prepare_training_arrays(matches, xi)  # convert match table to arrays
    teams = arrays["teams"]  # ordered team labels
    n_teams = len(teams)  # parameter dimension base
    initial_ability = arrays["initial_ability"]  # Elo-derived starting values
    initial = np.concatenate(  # attack, defense, home_field, rho
        [initial_ability, initial_ability * 0.45, np.asarray([0.18, -0.03 if fit_rho else 0.0])]
    )
    bounds = [(None, None)] * (2 * n_teams) + [(-0.2, 0.8), DIXON_COLES_RHO_BOUNDS if fit_rho else (0.0, 0.0)]  # bounds
    result = minimize(  # optimise weighted negative log likelihood
        _negative_log_likelihood,
        initial,
        args=(arrays, ridge),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "maxfun": 250_000, "ftol": 1e-8, "maxls": 30},
    )
    fitted = _params_to_model(  # model
        result.x,
        teams,
        xi=xi,
        selected_engine="dixon_coles" if fit_rho else "plain_poisson",
        score_distribution=SCORE_DISTRIBUTION_POISSON,
    )
    accepted_fit = bool(result.success or np.isfinite(result.fun))  # validation will judge finite fits
    diagnostics = {  # serialisable optimiser diagnostics
        "success": accepted_fit,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "fun": float(result.fun),
        "n_train_matches": int(len(arrays["ordered"])),
        "n_teams": int(n_teams),
        "reference_date": arrays["reference_date"],
        "fit_rho": bool(fit_rho),
    }
    return fitted, diagnostics  # fitted model and diagnostics


def _params_to_model(
    params: np.ndarray,
    teams: list[str],
    xi: float,
    selected_engine: str,
    score_distribution: str = DEFAULT_SCORE_DISTRIBUTION,
) -> DixonColesModel:
    """Convert a parameter vector into a DixonColesModel."""
    n_teams = len(teams)  # team parameter dimension
    attack_values = params[:n_teams] - np.mean(params[:n_teams])  # enforce mean attack = 0
    defense_values = params[n_teams: 2 * n_teams]  # fitted defense values
    attack = {team: float(value) for team, value in zip(teams, attack_values, strict=True)}  # map
    defense = {team: float(value) for team, value in zip(teams, defense_values, strict=True)}  # map
    return DixonColesModel(  # fitted container
        teams=teams,
        attack=attack,
        defense=defense,
        home_field=float(params[-2]),
        rho=float(params[-1]),
        xi=xi,
        selected_engine=selected_engine,
        score_distribution=score_distribution,
        score_dispersion=NEGATIVE_BINOMIAL_DISPERSION,
    )


def _negative_log_likelihood(params: np.ndarray, arrays: dict[str, object], ridge: float) -> float:
    """Return weighted negative log-likelihood for optimiser calls."""
    n_teams = len(arrays["teams"])  # team parameter dimension
    attack = params[:n_teams] - np.mean(params[:n_teams])  # centred attack constraint
    defense = params[n_teams: 2 * n_teams]  # defense parameters
    home_field = params[-2]  # non-neutral home effect
    rho = params[-1]  # low-score dependence
    home_idx = arrays["home_idx"]  # home team ids
    away_idx = arrays["away_idx"]  # away team ids
    home_goals = arrays["home_goals"]  # observed home goals
    away_goals = arrays["away_goals"]  # observed away goals
    neutral = arrays["neutral"]  # neutral venue flags
    weights = arrays["weights"]  # time-decay weights
    lam = np.exp(np.clip((~neutral).astype(float) * home_field + attack[home_idx] - defense[away_idx], -4.0, 4.0))  # home means
    mu = np.exp(np.clip(attack[away_idx] - defense[home_idx], -4.0, 4.0))  # away means
    tau = np.ones_like(lam)  # default correction factor
    mask_00 = (home_goals == 0) & (away_goals == 0)  # 0-0 cells
    mask_01 = (home_goals == 0) & (away_goals == 1)  # 0-1 cells
    mask_10 = (home_goals == 1) & (away_goals == 0)  # 1-0 cells
    mask_11 = (home_goals == 1) & (away_goals == 1)  # 1-1 cells
    tau[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho  # DC 0-0 correction
    tau[mask_01] = 1.0 + lam[mask_01] * rho  # DC 0-1 correction
    tau[mask_10] = 1.0 + mu[mask_10] * rho  # DC 1-0 correction
    tau[mask_11] = 1.0 - rho  # DC 1-1 correction
    if np.any(tau <= 1e-9):  # invalid correction gives undefined log likelihood
        return 1e12  # large penalty for invalid parameter region
    home_log_prob = home_goals * np.log(lam) - lam - gammaln(home_goals + 1.0)  # Poisson log pmf
    away_log_prob = away_goals * np.log(mu) - mu - gammaln(away_goals + 1.0)  # Poisson log pmf
    log_likelihood = np.log(tau) + home_log_prob + away_log_prob  # corrected log likelihood
    penalty = ridge * (float(np.sum(attack ** 2)) + float(np.sum(defense ** 2)))  # weak ridge
    return float(-np.sum(weights * log_likelihood) + penalty)  # weighted negative log likelihood


def evaluate_model_rps(model: DixonColesModel, matches: pd.DataFrame, force_rho_zero: bool = False) -> pd.DataFrame:
    """Evaluate match-level W/D/L RPS for a fitted model.

    Parameters
    ----------
    model : DixonColesModel
        Fitted model to evaluate.
    matches : pandas.DataFrame
        Heldout match rows.
    force_rho_zero : bool
        If True, evaluate with rho=0 for a plain-Poisson comparison.

    Returns
    -------
    pandas.DataFrame
        Per-match probabilities and RPS values.
    """
    rows: list[dict[str, object]] = []  # collect one evaluation record per match
    for row in matches.itertuples(index=False):  # iterate heldout rows
        lam, mu = model.fixture_means(row.home_team, row.away_team, bool(row.neutral))  # fixture means
        if force_rho_zero:  # optional independent-Poisson baseline override
            matrix = scoreline_matrix(lam, mu, 0.0, model.max_goals)  # scoreline probabilities
        else:
            matrix = model.predict_fixture(row.home_team, row.away_team, bool(row.neutral))  # configured matrix
        p_home, p_draw, p_away = match_outcome_probs(matrix)  # W/D/L probabilities
        outcome = observed_outcome_index(int(row.home_goals), int(row.away_goals))  # actual outcome
        rows.append(  # append per-match validation row
            {
                "row_id": getattr(row, "row_id", len(rows)),
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "rps": ranked_probability_score((p_home, p_draw, p_away), outcome),
            }
        )
    return pd.DataFrame(rows)  # per-match evaluation table
