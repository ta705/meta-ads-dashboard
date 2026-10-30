# etl.py
# Purpose: This is the "Extract, Transform, Load" script. 
# It handles all the messy data work in the background. It cleans up the raw 
# Meta Ads export, removes ads with $0 spend, calculates the ROAS and CAC, 
# and tries to automatically detect the language and region of each ad.

import re
import sys
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    "date": ["date", "day", "reporting starts"],
    "reporting_start": ["reporting starts", "reporting start"],
    "reporting_end": ["reporting ends", "reporting end"],
    "ad_name": ["ad name"],
    "ad_delivery": ["ad delivery"],
    "adset_name": ["ad set name"],
    "campaign_name": ["campaign name"],
    "attribution": ["attribution setting"],
    "results": ["results"],
    "result_indicator": ["result indicator"],
    "cost_per_result": ["cost per results", "cost per result"],
    "adset_budget": ["ad set budget"],
    "adset_budget_type": ["ad set budget type"],
    "spend": ["amount spent (usd)", "amount spent", "spend"],
    "impressions": ["impressions"],
    "reach": ["reach"],
    "frequency": ["frequency"],
    "clicks": ["link clicks", "clicks (all)", "clicks"],
    "ctr": ["ctr (link click-through rate)", "ctr (all)", "ctr"],
    "cpc": ["cpc (cost per link click) (usd)", "cpc (cost per link click)", "cpc (all) (usd)", "cpc (all)", "cpc"],
    "cpm": ["cpm (cost per 1,000 impressions) (usd)", "cpm (cost per 1,000 impressions)", "cpm"],
    "landing_page_views": ["website landing page views", "landing page views"],
    "cost_per_lpv": ["cost per landing page view (usd)", "cost per landing page view"],
    "adds_to_cart": ["website adds to cart", "adds to cart"],
    "checkouts": ["website checkouts initiated", "checkouts initiated"],
    "purchases": ["website purchases", "purchases"],
    "revenue": ["website purchases conversion value", "purchases conversion value"],
    "purchase_roas": ["purchase roas (return on ad spend)", "purchase roas"],
    "cost_per_purchase": ["cost per purchase (usd)", "cost per purchase"],
    "quality_ranking": ["quality ranking"],
    "engagement_ranking": ["engagement rate ranking"],
    "conversion_ranking": ["conversion rate ranking"],
}

def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    try: s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError): pass
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s).strip().lower()

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm_map = {c: _norm(c) for c in df.columns}
    rename = {}
    used = set()
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            for orig, n in norm_map.items():
                if n == alias and orig not in used:
                    rename[orig] = canonical
                    used.add(orig)
                    break
            if canonical in rename.values(): break
    return df.rename(columns=rename)

LANGUAGE_KEYWORDS = {
    "amharic": ["amharic", "habesha", "ethiopia"], "tigrinya": ["tigrinya"], "yoruba": ["yoruba"],
    "igbo": ["igbo"], "hausa": ["hausa"], "lingala": ["lingala"], "somali": ["somali"],
    "twi": ["twi", "akan"], "ga": ["ga "], "fante": ["fante"], "wolof": ["wolof"],
    "luganda": ["luganda"], "shona": ["shona"], "zulu": ["zulu"], "xhosa": ["xhosa"],
    "kinyarwanda": ["kinyarwanda"], "bemba": ["bemba"], "swahili": ["swahili"],
    "english": ["english"], "nigerian": ["nigerian", "nigeria"], "ghanaian": ["ghanaian", "ghana"],
}

REGION_KEYWORDS = {
    "US": ["us", "usa", "united states", "minneapolis", "minnesota", "texas"],
    "UK": ["uk", "united kingdom", "london"], "CA": ["ca ", "canada"],
    "EU": ["eu", "europe", "france", "germany", "belgium", "netherlands", "nl ", "poland", "czech", "czeh", "italy", "spain", "ireland"],
    "AU": ["au", "australia"], "ZA": ["za", "south africa", "johannesburg", "joburg", "cape town", "durban"],
    "Africa": ["zambia", "ghana", "nigeria", "congo", "kenya", "uganda"],
}

FUNNEL_KEYWORDS = {
    "retargeting": ["retarget", "warm", "visitor", "vc and atc"],
    "lookalike": ["lal", "lookalike", "ll "], "interest": ["interest"],
    "adv_plus": ["adv+", "advantage"], "broad": ["super lal", "broad", "general"],
}

def _match_keyword(text: str, mapping: dict) -> str | None:
    text = " " + text.lower() + " "
    for label, kws in mapping.items():
        for kw in kws:
            if kw in text: return label
    return None

def _parse_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    ad = (df["ad_name"].fillna("").astype(str) + " " + df.get("campaign_name", pd.Series("", index=df.index)).fillna("").astype(str)).str.lower()
    adset = (df.get("adset_name", pd.Series("", index=df.index)).fillna("").astype(str).str.lower())
    df["language"] = ad.map(lambda s: _match_keyword(s, LANGUAGE_KEYWORDS)).fillna("unknown")
    df["region"] = adset.map(lambda s: _match_keyword(s, REGION_KEYWORDS)).fillna("unknown")
    df["funnel_stage"] = adset.map(lambda s: _match_keyword(s, FUNNEL_KEYWORDS)).fillna("prospecting")
    return df

NUMERIC_COLS = ["spend", "impressions", "reach", "frequency", "clicks", "ctr", "cpc", "cpm", "landing_page_views", "cost_per_lpv", "adds_to_cart", "checkouts", "purchases", "revenue", "purchase_roas", "cost_per_purchase", "results", "cost_per_result"]

def transform(df_raw: pd.DataFrame, target_roas: float = 1.5, target_cac: float | None = None, min_spend_for_flags: float = 50.0) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _canonicalize_columns(df)
    
    for d in ("date", "reporting_start", "reporting_end"):
        if d in df.columns: df[d] = pd.to_datetime(df[d], errors="coerce")
    if "date" not in df.columns: df["date"] = df.get("reporting_start")
        
    for c in ("ad_name", "adset_name", "campaign_name", "ad_delivery", "attribution", "result_indicator"):
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
            
    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan
        
    df["spend"] = df["spend"].fillna(0.0)
    for c in ("impressions", "reach", "clicks", "landing_page_views", "purchases", "revenue"):
        df[c] = df[c].fillna(0.0)
        
    df = df[(df["spend"] > 0) | (df["impressions"] > 0)].copy()
    
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], np.nan)
    df["cac"] = np.where(df["purchases"] > 0, df["spend"] / df["purchases"], np.nan)
    df["purchase_rate"] = np.where(df["landing_page_views"] > 0, df["purchases"] / df["landing_page_views"], np.nan)
    df["ctr"] = df["ctr"].where(df["ctr"].notna(), np.where(df["impressions"] > 0, df["clicks"] / df["impressions"] * 100, np.nan))
    df["cpm"] = df["cpm"].where(df["cpm"].notna(), np.where(df["impressions"] > 0, df["spend"] / df["impressions"] * 1000, np.nan))
    df["cpc"] = df["cpc"].where(df["cpc"].notna(), np.where(df["clicks"] > 0, df["spend"] / df["clicks"], np.nan))
    
    df = _parse_dimensions(df)
    
    def _flag(row):
        flags = []
        if row["spend"] >= min_spend_for_flags:
            if pd.notna(row["roas"]) and row["roas"] < 1.0: flags.append("cut")
            elif pd.notna(row["roas"]) and row["roas"] >= target_roas: flags.append("scale")
            if (pd.notna(row.get("frequency")) and row["frequency"] > 2.5 and pd.notna(row["roas"]) and row["roas"] < 1.2): flags.append("fatigue")
            if (target_cac is not None and pd.notna(row["cac"]) and row["cac"] > 2 * target_cac): flags.append("high_cac")
        return ",".join(flags) if flags else ""
        
    df["flags"] = df.apply(_flag, axis=1)
    
    keep = ["date", "reporting_start", "reporting_end", "campaign_name", "adset_name", "ad_name", "language", "region", "funnel_stage", "ad_delivery", "attribution", "spend", "impressions", "reach", "frequency", "clicks", "ctr", "cpc", "cpm", "landing_page_views", "adds_to_cart", "checkouts", "purchases", "revenue", "roas", "cac", "purchase_rate", "purchase_roas", "cost_per_purchase", "flags"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)

def load_and_clean(path: str | Path, target_roas: float = 1.5, target_cac: float | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"): raw = pd.read_excel(path)
    else: raw = pd.read_csv(path)
    return transform(raw, target_roas=target_roas, target_cac=target_cac)