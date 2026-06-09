"""Socioeconomic feature assembly and OLS fits for the GYAN expert."""

from __future__ import annotations  # modern type-hint syntax on all runtimes

import json  # parse cached World Bank and FIFA API artifacts
import re  # normalise country names and extract temperature values
import unicodedata  # strip accents before name joins
from pathlib import Path  # typed file paths accepted by parsers

import pandas as pd  # tabular feature engineering
import statsmodels.api as sm  # OLS model fitting for Task 1.5

from gyan.config import (  # shared constants and published Hoffmann coefficients
    HOFFMANN_2002_COEFFICIENTS,
    HOST_NATIONS_2026,
    LATIN_FOOTBALL_NATIONS,
    TEMP_OPTIMUM_C,
)


# FIFA names that need a separate football name, macro-data name, or climate name.
TEAM_ALIASES: dict[str, str] = {  # association names -> canonical football names
    "USA": "United States",  # FIFA abbreviation differs from D1/WB name
    "IR Iran": "Iran",  # FIFA label includes political prefix
    "Türkiye": "Turkey",  # D1 and WB still use the ASCII English spelling
    "Korea Republic": "South Korea",  # D1 common name
    "Korea DPR": "North Korea",  # D1 common name
    "Côte d'Ivoire": "Ivory Coast",  # D1 common name
    "Czechia": "Czech Republic",  # D1 historical/common name
    "Congo DR": "Democratic Republic of the Congo",  # disambiguate Congo teams
    "Cabo Verde": "Cape Verde",  # D1/Wikipedia common name
    "Curaçao": "Curacao",  # project labels use ASCII spelling
    "China PR": "China",  # FIFA label includes PR suffix
    "Kyrgyz Republic": "Kyrgyzstan",  # D1/WB common name
    "The Gambia": "Gambia",  # D1/WB omit article
    "St Kitts and Nevis": "Saint Kitts and Nevis",  # expand saint abbreviation
    "St Lucia": "Saint Lucia",  # expand saint abbreviation
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",  # expand saint
    "Chinese Taipei": "Taiwan",  # football association label
    "Brunei Darussalam": "Brunei",  # shortened common name
    "US Virgin Islands": "United States Virgin Islands",  # expand US abbreviation
    "Hong Kong, China": "Hong Kong",  # FIFA geopolitical suffix
}

WB_COUNTRY_ALIASES: dict[str, str] = {  # football names -> World Bank country names
    "England": "United Kingdom",  # home nations use UK macro indicators
    "Scotland": "United Kingdom",  # home nations use UK macro indicators
    "Wales": "United Kingdom",  # home nations use UK macro indicators
    "Northern Ireland": "United Kingdom",  # home nations use UK macro indicators
    "Republic of Ireland": "Ireland",  # FIFA uses football-association name
    "Slovakia": "Slovak Republic",  # World Bank formal country name
    "Iran": "Iran, Islamic Rep.",  # World Bank formal country name
    "Turkey": "Turkiye",  # World Bank spelling
    "South Korea": "Korea, Rep.",  # World Bank formal country name
    "North Korea": "Korea, Dem. People's Rep.",  # World Bank formal country name
    "Egypt": "Egypt, Arab Rep.",  # World Bank formal country name
    "Ivory Coast": "Cote d'Ivoire",  # World Bank ASCII formal name
    "Russia": "Russian Federation",  # World Bank formal country name
    "Czech Republic": "Czechia",  # World Bank current name
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",  # World Bank formal name
    "Republic of the Congo": "Congo, Rep.",  # World Bank formal name
    "Venezuela": "Venezuela, RB",  # World Bank formal country name
    "Cape Verde": "Cabo Verde",  # World Bank spelling
    "Syria": "Syrian Arab Republic",  # World Bank formal country name
    "Palestine": "West Bank and Gaza",  # World Bank territory name
    "West Bank and Gaza": "West Bank and Gaza",  # World Bank territory name
    "Kyrgyzstan": "Kyrgyz Republic",  # World Bank formal country name
    "Gambia": "Gambia, The",  # World Bank formal country name
    "Yemen": "Yemen, Rep.",  # World Bank formal country name
    "Saint Kitts and Nevis": "St. Kitts and Nevis",  # World Bank abbreviation
    "Saint Lucia": "St. Lucia",  # World Bank abbreviation
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",  # WB abbreviation
    "Congo": "Congo, Rep.",  # FIFA short label
    "Hong Kong": "Hong Kong SAR, China",  # World Bank formal territory name
    "Macau": "Macao SAR, China",  # World Bank formal territory name
    "Laos": "Lao PDR",  # World Bank formal country name
    "Brunei": "Brunei Darussalam",  # World Bank formal country name
    "Bahamas": "Bahamas, The",  # World Bank formal country name
}

TEMPERATURE_ALIASES: dict[str, str] = {  # football names -> temperature-table names
    "England": "United Kingdom",  # home nations use UK climate proxy
    "Scotland": "United Kingdom",  # home nations use UK climate proxy
    "Wales": "United Kingdom",  # home nations use UK climate proxy
    "Northern Ireland": "United Kingdom",  # home nations use UK climate proxy
    "Republic of Ireland": "Ireland",  # temperature table name
    "United Kingdom": "United Kingdom",  # home-nation macro proxy
    "United States": "United States",  # D12 fallback table name
    "Iran": "Iran",  # D12 fallback table name
    "South Korea": "South Korea",  # D12 fallback table name
    "North Korea": "North Korea",  # D12 fallback table name
    "Ivory Coast": "Ivory Coast",  # D12 fallback table name
    "Czech Republic": "Czech Republic",  # D12 fallback table name
    "Democratic Republic of the Congo": "Democratic Republic of the Congo",  # D12 name
    "Republic of the Congo": "Republic of the Congo",  # D12 name
    "Congo": "Republic of the Congo",  # FIFA short label
    "Cape Verde": "Cape Verde",  # D12 fallback table name
    "Palestine": "Palestine",  # D12 fallback table name
    "West Bank and Gaza": "Palestine",  # temperature table uses Palestine
    "Hong Kong": "Hong Kong",  # may be absent in fallback data
}


def normalise_name(name: str) -> str:
    """Return an ASCII lowercase key for deterministic name joins.

    Parameters
    ----------
    name : str
        Country or team name to normalise.

    Returns
    -------
    str
        Alphanumeric-only lowercase join key.
    """
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()  # strip accents
    return re.sub(r"[^a-z0-9]+", "", ascii_name.lower())  # keep only stable join chars


def canonical_football_team(team_name: str) -> str:
    """Return the project football name for a FIFA ranking team name.

    Parameters
    ----------
    team_name : str
        Team name as supplied by the FIFA rankings API.

    Returns
    -------
    str
        Canonical football team name used for joins to D1/Elo data.
    """
    return TEAM_ALIASES.get(team_name, team_name)  # explicit alias or identity


def macro_country_name(team_name: str) -> str:
    """Return the World Bank country name used for macroeconomic joins.

    Parameters
    ----------
    team_name : str
        Canonical football team name.

    Returns
    -------
    str
        World Bank country/territory name to use for GDP and population.
    """
    return WB_COUNTRY_ALIASES.get(team_name, team_name)  # use alias where WB differs


def temperature_country_name(team_name: str) -> str:
    """Return the country-temperature table name used for climate joins.

    Parameters
    ----------
    team_name : str
        Canonical football team name or macro country proxy.

    Returns
    -------
    str
        Temperature-table country or region name.
    """
    return TEMPERATURE_ALIASES.get(team_name, team_name)  # use alias where climate names differ


def parse_fifa_rankings(path: Path | str) -> pd.DataFrame:
    """Parse the official FIFA rankings API JSON into a compact table.

    Parameters
    ----------
    path : Path | str
        Cached D5 `api.fifa.com/api/v3/rankings` JSON path.

    Returns
    -------
    pandas.DataFrame
        Ranking rows with FIFA rank, points, confederation, and canonical team.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))  # load cached JSON
    rows: list[dict[str, object]] = []  # collect one row per ranked team
    for item in payload["Results"]:  # iterate official ranking objects
        english_names = [name for name in item["TeamName"] if name["Locale"].startswith("en")]  # labels
        fifa_team = english_names[0]["Description"]  # first English description
        canonical_team = canonical_football_team(fifa_team)  # project football name
        rows.append(  # append the compact ranking record
            {
                "team": canonical_team,  # canonical team name
                "fifa_team": fifa_team,  # original FIFA API team label
                "fifa_country_code": item["IdCountry"],  # FIFA association code
                "confederation": item["ConfederationName"],  # AFC/CAF/etc.
                "fifa_rank": int(item["Rank"]),  # current FIFA rank
                "fifa_points": float(item["DecimalTotalPoints"]),  # current points
                "fifa_pub_date": item["PubDate"],  # ranking publication date
            }
        )
    return pd.DataFrame(rows).sort_values("fifa_rank").reset_index(drop=True)  # rank order


def parse_world_bank_latest(path: Path | str, value_col: str) -> pd.DataFrame:
    """Parse a World Bank indicator JSON and keep latest non-null values.

    Parameters
    ----------
    path : Path | str
        Cached World Bank API JSON path.
    value_col : str
        Name to assign to the indicator value column.

    Returns
    -------
    pandas.DataFrame
        One latest non-null row per World Bank ISO3 code.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))  # load cached WB JSON
    records = [  # flatten each non-null country-year observation
        {
            "world_bank_code": row["countryiso3code"],  # World Bank ISO3 code
            "world_bank_country": row["country"]["value"],  # World Bank country name
            "year": int(row["date"]),  # observation year
            value_col: float(row["value"]),  # indicator value
            "join_key": normalise_name(row["country"]["value"]),  # name join key
        }
        for row in payload[1]  # second item contains observations
        if row.get("countryiso3code") and row.get("value") is not None  # usable observations
    ]
    frame = pd.DataFrame(records)  # convert flattened records to a DataFrame
    latest = frame.sort_values("year").groupby("world_bank_code", as_index=False).tail(1)  # latest
    latest = latest.rename(columns={"year": f"{value_col}_year"})  # preserve indicator year
    return latest.reset_index(drop=True)  # stable compact output


def parse_country_temperatures(path: Path | str) -> pd.DataFrame:
    """Parse the D12 fallback country mean annual temperature table.

    Parameters
    ----------
    path : Path | str
        Cached HTML table path.

    Returns
    -------
    pandas.DataFrame
        Country/region names and mean annual temperature in Celsius.

    Notes
    -----
    The cached World Bank CCKP page supplied metadata but not a simple
    downloadable country table. This fallback table is World-Bank-sourced
    country mean annual temperature and is used directly as the D12 climate
    feature.
    """
    table = pd.read_html(path)[0]  # parse the single HTML table
    temperatures = table.rename(columns={"Country or region": "temperature_country"})  # name
    temperatures["mean_annual_temp_c"] = (  # extract Celsius value from display text
        temperatures["Temperature"].astype(str).str.extract(r"([-\u2212]?[0-9]+\.?[0-9]*)")[0]
        .str.replace("\u2212", "-", regex=False).astype(float)
    )
    temperatures["temperature_join_key"] = temperatures["temperature_country"].map(normalise_name)  # key
    return temperatures[["temperature_country", "mean_annual_temp_c", "temperature_join_key"]]  # cols


def build_feature_table(
    fifa_rankings: pd.DataFrame,
    gdp_latest: pd.DataFrame,
    population_latest: pd.DataFrame,
    temperatures: pd.DataFrame,
    elo_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the Task 1.5 per-team socioeconomic feature table.

    Parameters
    ----------
    fifa_rankings : pandas.DataFrame
        Output of parse_fifa_rankings.
    gdp_latest, population_latest : pandas.DataFrame
        Outputs of parse_world_bank_latest for GDP/capita PPP and population.
    temperatures : pandas.DataFrame
        Output of parse_country_temperatures.
    elo_ratings : pandas.DataFrame
        Current Elo ratings from Task 1.4.

    Returns
    -------
    pandas.DataFrame
        One row per FIFA-ranked team with socioeconomic, FIFA, and Elo fields.
    """
    features = fifa_rankings.copy()  # never mutate caller data
    features["macro_country"] = features["team"].map(macro_country_name)  # macro proxy name
    features["macro_join_key"] = features["macro_country"].map(normalise_name)  # WB join key
    features["temperature_country"] = features["team"].map(temperature_country_name)  # climate name
    features["temperature_join_key"] = features["temperature_country"].map(normalise_name)  # temp key
    features = features.merge(  # add GDP/capita PPP latest values
        gdp_latest[["join_key", "world_bank_code", "world_bank_country", "gdp_per_capita_ppp", "gdp_per_capita_ppp_year"]],
        left_on="macro_join_key",
        right_on="join_key",
        how="left",
    )
    features = features.merge(  # add population latest values
        population_latest[["join_key", "population", "population_year"]],
        left_on="macro_join_key",
        right_on="join_key",
        how="left",
        suffixes=("", "_population"),
    )
    features = features.merge(  # add D12 fallback mean annual temperature
        temperatures[["temperature_join_key", "mean_annual_temp_c"]],
        on="temperature_join_key",
        how="left",
    )
    elo_lookup = (  # Elo key, one row per normalised label after alias expansion
        elo_ratings.assign(elo_join_key=elo_ratings["team"].map(normalise_name))
        .drop_duplicates(subset=["elo_join_key"], keep="last")
    )
    features["elo_join_key"] = features["team"].map(normalise_name)  # team join key
    features = features.merge(  # add current final Elo rating for OLS target
        elo_lookup[["elo_join_key", "elo_rating"]],
        on="elo_join_key",
        how="left",
    )
    features["team_tournament"] = "2026"  # Task 1.5 one team-tournament row
    features["gdp_per_capita_ppp_sq"] = features["gdp_per_capita_ppp"] ** 2  # Hoffmann b2 term
    features["population_share"] = features["population"] / features["population"].sum(skipna=True)  # share
    features["temp_dev_sq"] = (features["mean_annual_temp_c"] - TEMP_OPTIMUM_C) ** 2  # eta term
    features["latin"] = features["team"].isin(LATIN_FOOTBALL_NATIONS).astype("int8")  # Latin dummy
    features["host"] = features["team"].isin(HOST_NATIONS_2026).astype("int8")  # 2026 host dummy
    features["latin_x_population_share"] = features["latin"] * features["population_share"]  # phi term
    features["hoffmann_prior_score"] = hoffmann_prior_score(features)  # published-coefficient score
    output_columns = [  # stable schema for downstream experts
        "team_tournament", "team", "fifa_team", "fifa_country_code", "confederation",
        "fifa_rank", "fifa_points", "fifa_pub_date", "macro_country", "world_bank_code",
        "world_bank_country", "gdp_per_capita_ppp", "gdp_per_capita_ppp_year",
        "gdp_per_capita_ppp_sq", "population", "population_year", "population_share",
        "temperature_country", "mean_annual_temp_c", "temp_dev_sq", "latin", "host",
        "latin_x_population_share", "elo_rating", "hoffmann_prior_score",
    ]
    return features[output_columns].sort_values("fifa_rank").reset_index(drop=True)  # stable order


def hoffmann_prior_score(features: pd.DataFrame) -> pd.Series:
    """Evaluate the published Hoffmann 2002 linear predictor.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature table containing the Hoffmann terms.

    Returns
    -------
    pandas.Series
        Linear predictor using config.HOFFMANN_2002_COEFFICIENTS.
    """
    coefficients = HOFFMANN_2002_COEFFICIENTS  # shorthand for readable formula lines
    return (  # direct transcription of the published specification
        coefficients["constant"]
        + coefficients["gnp_per_capita"] * features["gdp_per_capita_ppp"]
        + coefficients["gnp_per_capita_sq"] * features["gdp_per_capita_ppp_sq"]
        + coefficients["temp_dev_sq"] * features["temp_dev_sq"]
        + coefficients["host_dummy"] * features["host"]
        + coefficients["latin_x_pop_share"] * features["latin_x_population_share"]
    )


def fit_socioeconomic_models(features: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Fit pure Hoffmann and FIFA-augmented OLS specs against current Elo.

    Parameters
    ----------
    features : pandas.DataFrame
        Output of build_feature_table.

    Returns
    -------
    dict[str, dict[str, object]]
        Model coefficients, R-squared values, and observation counts.

    Notes
    -----
    The OLS response is current Elo rating because FIFA points are required as an
    augmented predictor. The PRD thresholds are recorded as diagnostics rather
    than strict pass/fail values under this target interpretation.
    """
    required = [  # columns required for both models
        "elo_rating", "gdp_per_capita_ppp", "gdp_per_capita_ppp_sq",
        "temp_dev_sq", "host", "latin_x_population_share", "fifa_points",
    ]
    modelling = features.dropna(subset=required).copy()  # complete-case OLS input
    pure_cols = [  # Hoffmann-only regressors
        "gdp_per_capita_ppp", "gdp_per_capita_ppp_sq", "temp_dev_sq",
        "host", "latin_x_population_share",
    ]
    augmented_cols = pure_cols + ["fifa_points"]  # Klement-style FIFA-points augmentation
    return {  # fit and serialise both model summaries
        "pure_hoffmann": _fit_ols(modelling, pure_cols),
        "augmented_fifa_points": _fit_ols(modelling, augmented_cols),
    }


def _fit_ols(data: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    """Fit one OLS model and return JSON-serialisable summary fields."""
    design = sm.add_constant(data[columns], has_constant="add")  # include intercept
    model = sm.OLS(data["elo_rating"], design).fit()  # fit least-squares model
    return {  # compact machine-readable model summary
        "n_obs": int(model.nobs),  # complete-case row count
        "r_squared": float(model.rsquared),  # R^2 diagnostic
        "coefficients": {name: float(value) for name, value in model.params.items()},  # betas
    }


def validate_feature_table(features: pd.DataFrame) -> dict[str, object]:
    """Validate the socioeconomic feature table and return run metrics.

    Parameters
    ----------
    features : pandas.DataFrame
        Output of build_feature_table.

    Returns
    -------
    dict[str, object]
        Coverage and missingness metrics for the run record.
    """
    assert features["team"].is_unique  # one row per FIFA-ranked team
    required_feature_cols = ["fifa_points", "gdp_per_capita_ppp", "population", "mean_annual_temp_c"]  # cols
    complete_feature_rows = features.dropna(subset=required_feature_cols)  # complete feature rows
    complete_model_rows = features.dropna(subset=required_feature_cols + ["elo_rating"])  # with target
    return {  # expose validation facts to logs/run records
        "rows": int(len(features)),  # total FIFA-ranked teams
        "complete_feature_rows": int(len(complete_feature_rows)),  # rows with core features
        "complete_model_rows": int(len(complete_model_rows)),  # rows usable for OLS
        "missing_gdp": int(features["gdp_per_capita_ppp"].isna().sum()),  # GDP gaps
        "missing_population": int(features["population"].isna().sum()),  # population gaps
        "missing_temperature": int(features["mean_annual_temp_c"].isna().sum()),  # temp gaps
        "missing_elo": int(features["elo_rating"].isna().sum()),  # Elo target gaps
    }
