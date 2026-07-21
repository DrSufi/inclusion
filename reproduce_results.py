from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter, ScalarFormatter

from src.utils import FOCAL_DEFINITIONS, THEME_LABELS, TOPIC_SHORT_LABELS, holm_adjust, read_features


def cramer_v(table: pd.DataFrame, chi2: float) -> float:
    # Bias corrected Cramer V following Bergsma (2013).
    n = float(table.to_numpy().sum())
    rows, cols = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corr = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corr = cols - ((cols - 1) ** 2) / (n - 1)
    denominator = min(cols_corr - 1, rows_corr - 1)
    return float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else 0.0


def sentiment_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["stratum", "primary_condition", "reply"]:
        table = pd.crosstab(df[group_col], df["sentiment"])
        chi2, p, dof, _ = st.chi2_contingency(table)
        rows.append({
            "test": "Chi-square", "comparison": f"Sentiment by {group_col}",
            "statistic": chi2, "df": dof, "p_value": p,
            "effect_size": "Cramer V", "effect_value": cramer_v(table, chi2),
        })
        samples = [group["sentiment_score"].to_numpy() for _, group in df.groupby(group_col)]
        statistic, p_kw = st.kruskal(*samples)
        rows.append({
            "test": "Kruskal-Wallis", "comparison": f"Sentiment score by {group_col}",
            "statistic": statistic, "df": len(samples) - 1, "p_value": p_kw,
            "effect_size": "", "effect_value": np.nan,
        })
    for col, label in THEME_LABELS.items():
        table = pd.crosstab(df["stratum"], df[col])
        chi2, p, dof, _ = st.chi2_contingency(table)
        rows.append({
            "test": "Chi-square", "comparison": f"{label} by stratum",
            "statistic": chi2, "df": dof, "p_value": p,
            "effect_size": "Cramer V", "effect_value": cramer_v(table, chi2),
        })
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"])
    return result


def logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    theme_cols = list(THEME_LABELS)
    X = df[["reply"] + theme_cols].copy()
    X["reply"] = X["reply"].astype(int)
    for col in theme_cols:
        X[col] = X[col].astype(int)
    X = X.rename(columns=THEME_LABELS)
    dummies = pd.get_dummies(df["stratum"], prefix="stratum", drop_first=True, dtype=int)
    X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = (df["sentiment"] == "Negative").astype(int)
    fitted = sm.Logit(y, X).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": df["video_id"]}
    )
    conf = fitted.conf_int()
    result = pd.DataFrame({
        "term": fitted.params.index,
        "coefficient": fitted.params.values,
        "std_error": fitted.bse.values,
        "p_value": fitted.pvalues.values,
        "odds_ratio": np.exp(fitted.params.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
    })
    result["term"] = result["term"].str.replace("stratum_", "stratum ", regex=False)
    return result


def table_1(df: pd.DataFrame) -> pd.DataFrame:
    per_video = df.groupby(["stratum", "video_id"]).size().rename("video_comments").reset_index()
    rows = []
    for stratum, group in df.groupby("stratum"):
        rows.append({
            "video_stratum": stratum,
            "videos": group["video_id"].nunique(),
            "comments": len(group),
            "corpus_pct": len(group) / len(df) * 100,
            "distinct_authors": group["author_key"].nunique(),
            "median_comments_per_video": per_video.loc[per_video["stratum"] == stratum, "video_comments"].median(),
            "replies_pct": group["reply"].mean() * 100,
        })
    groups = pd.DataFrame(rows).sort_values("comments", ascending=False, ignore_index=True)
    total = pd.DataFrame([{
        "video_stratum": "Total", "videos": df["video_id"].nunique(), "comments": len(df),
        "corpus_pct": 100.0, "distinct_authors": df["author_key"].nunique(),
        "median_comments_per_video": df.groupby("video_id").size().median(),
        "replies_pct": df["reply"].mean() * 100,
    }])
    return pd.concat([groups, total], ignore_index=True)


def table_3(df: pd.DataFrame) -> pd.DataFrame:
    video_sentiment = (
        df.groupby(["video_id", "primary_condition", "sentiment"]).size()
        .unstack(fill_value=0)
    )
    video_sentiment = video_sentiment.div(video_sentiment.sum(axis=1), axis=0).reset_index()
    rows = []
    for focal, members in FOCAL_DEFINITIONS.items():
        selected = video_sentiment[video_sentiment["primary_condition"].isin(members)]
        raw = df[df["primary_condition"].isin(members)]
        rows.append({
            "disability_condition": focal,
            "videos": selected["video_id"].nunique(),
            "analytical_comments": len(raw),
            "positive_pct": selected.get("Positive", 0).mean() * 100,
            "neutral_pct": selected.get("Neutral", 0).mean() * 100,
            "negative_pct": selected.get("Negative", 0).mean() * 100,
        })
    return pd.DataFrame(rows)


def table_4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in THEME_LABELS.items():
        subset = df[df[col]]
        dist = subset["sentiment"].value_counts(normalize=True)
        rows.append({
            "discursive_theme": label,
            "comments": len(subset),
            "prevalence_pct": len(subset) / len(df) * 100,
            "positive_pct": dist.get("Positive", 0) * 100,
            "neutral_pct": dist.get("Neutral", 0) * 100,
            "negative_pct": dist.get("Negative", 0) * 100,
            "mean_sentiment": subset["sentiment_score"].mean(),
        })
    return pd.DataFrame(rows).sort_values("comments", ascending=False, ignore_index=True)


def table_6(tests: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    r = reg.set_index("term")
    t = tests.set_index("comparison")
    def reg_row(h, term, decision):
        x = r.loc[term]
        return {
            "hypothesis": h, "tested_relationship": term,
            "statistical_evidence": f"OR={x.odds_ratio:.2f}; 95% CI={x.ci_low:.2f} to {x.ci_high:.2f}",
            "p_value": x.p_value, "decision": decision,
        }
    rows = [
        reg_row("H1a", "Behaviour and safety concerns", "Supported"),
        reg_row("H1b", "Parental emotional labour", "Supported"),
        reg_row("H1c", "Stigma, blame and judgement", "Supported"),
        reg_row("H1d", "Institutional support or failure", "Supported"),
        reg_row("H1e", "Misinformation and contested causation", "Supported cautiously"),
        reg_row("H2", "Support and empathy", "Supported"),
    ]
    h3 = t.loc["Inclusive education and accessibility by stratum"]
    h4 = t.loc["Sentiment by primary_condition"]
    h5 = t.loc["Sentiment by stratum"]
    rows.extend([
        {"hypothesis":"H3", "tested_relationship":"Institutional framing and inclusive education discourse", "statistical_evidence":f"Chi-square={h3.statistic:.2f}; Cramer V={h3.effect_value:.3f}", "p_value":h3.p_value, "decision":"Supported"},
        {"hypothesis":"H4", "tested_relationship":"Disability condition and sentiment", "statistical_evidence":f"Chi-square={h4.statistic:.2f}; Cramer V={h4.effect_value:.3f}", "p_value":h4.p_value, "decision":"Supported"},
        {"hypothesis":"H5", "tested_relationship":"Video category and sentiment", "statistical_evidence":f"Chi-square={h5.statistic:.2f}; Cramer V={h5.effect_value:.3f}", "p_value":h5.p_value, "decision":"Supported with a small effect"},
    ])
    return pd.DataFrame(rows)


def setup_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_figures(df: pd.DataFrame, reg: pd.DataFrame, out: Path) -> None:
    setup_plotting()
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1
    counts = df.groupby("stratum").size().sort_values()
    share = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.barh(counts.index, counts.values)
    ax.set_xscale("log")
    ax.set_xlabel("Analytical comments, logarithmic scale")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    for bar, count, pct in zip(bars, counts, share):
        ax.text(bar.get_width() * 1.06, bar.get_y() + bar.get_height()/2, f"{count:,}  ({pct:.1f}%)", va="center")
    ax.margins(x=0.24)
    save(fig, out, "Figure_1_Corpus_Distribution_by_Stratum")

    # Figure 2
    sentiment = table_3(df).sort_values("positive_pct")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    left = np.zeros(len(sentiment))
    for col, label in [("positive_pct","Positive"),("neutral_pct","Neutral"),("negative_pct","Negative")]:
        vals = sentiment[col].to_numpy()
        ax.barh(sentiment["disability_condition"], vals, left=left, label=label)
        for i, (val, base) in enumerate(zip(vals, left)):
            if val >= 7:
                ax.text(base + val/2, i, f"{val:.1f}%", ha="center", va="center")
        left += vals
    ax.set_xlim(0,100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Video balanced proportion of comments")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5,1.01))
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    save(fig, out, "Figure_2_Sentiment_by_Disability_Condition")

    # Figure 3
    terms = ["Behaviour and safety concerns", "Parental emotional labour", "Stigma, blame and judgement", "Institutional support or failure", "Misinformation and contested causation", "Support and empathy"]
    forest = reg[reg["term"].isin(terms)].sort_values("odds_ratio")
    y = np.arange(len(forest))
    xerr = np.vstack([forest["odds_ratio"]-forest["ci_low"], forest["ci_high"]-forest["odds_ratio"]])
    fig, ax = plt.subplots(figsize=(9.2,5.2))
    ax.errorbar(forest["odds_ratio"], y, xerr=xerr, fmt="o", capsize=4, linewidth=1.4, markersize=6)
    ax.axvline(1, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(forest["term"])
    ax.set_xlabel("Adjusted odds ratio for negative sentiment, logarithmic scale")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    for yi, row in enumerate(forest.itertuples(index=False)):
        ax.text(row.ci_high*1.08, yi, f"{row.odds_ratio:.2f} ({row.ci_low:.2f} to {row.ci_high:.2f})", va="center")
    ax.set_xlim(0.25,6.4)
    save(fig, out, "Figure_3_Adjusted_Odds_Ratios")

    # Figure 4
    condition_order = ["Autism", "ADHD/ADD", "Cerebral palsy", "Down syndrome"]
    theme_order = list(THEME_LABELS.values())
    rows = []
    for condition in condition_order:
        subset = df[df["primary_condition"] == condition]
        for col, label in THEME_LABELS.items():
            rows.append({"condition": condition, "theme": label, "prevalence": subset[col].mean()*100})
    matrix = pd.DataFrame(rows).pivot(index="theme", columns="condition", values="prevalence").reindex(index=theme_order, columns=condition_order).fillna(0)
    fig, ax = plt.subplots(figsize=(9.0,7.1))
    im = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(["Autism","ADHD and ADD","Cerebral palsy","Down syndrome"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(theme_order))); ax.set_yticklabels(theme_order)
    threshold = matrix.to_numpy().max()/2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i,j]
            ax.text(j,i,f"{value:.1f}",ha="center",va="center",color="white" if value>threshold else "black",fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03); cbar.set_label("Comments containing theme (%)")
    save(fig, out, "Figure_4_Theme_Prevalence_Heatmap")

    # Figure 5
    bubble_rows = []
    for focal, members in FOCAL_DEFINITIONS.items():
        subset = df[df["primary_condition"].isin(members)]
        grouped = subset.groupby("topic_id").agg(comments=("row_id","size"),mean_sentiment=("sentiment_score","mean")).reset_index()
        grouped["prevalence"] = grouped["comments"]/len(subset)
        grouped["focal"] = focal
        bubble_rows.append(grouped)
    bubble = pd.concat(bubble_rows, ignore_index=True)
    focal_order = list(FOCAL_DEFINITIONS)
    topic_order = list(range(1,13))
    bubble["x"] = bubble["focal"].map({v:i for i,v in enumerate(focal_order)})
    bubble["y"] = bubble["topic_id"].map({v:i for i,v in enumerate(topic_order)})
    max_prev = bubble["prevalence"].max()
    sizes = 90 + 1450*(bubble["prevalence"]/max_prev)
    fig, ax = plt.subplots(figsize=(9.4,8.2))
    scatter = ax.scatter(bubble["x"],bubble["y"],s=sizes,c=bubble["mean_sentiment"],alpha=0.80,edgecolors="black",linewidths=0.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(focal_order, rotation=18, ha="right")
    ax.set_yticks(range(12)); ax.set_yticklabels([TOPIC_SHORT_LABELS[i] for i in topic_order]); ax.invert_yaxis()
    ax.grid(linestyle=":",linewidth=0.6,alpha=0.45)
    for row in bubble.itertuples(index=False):
        ax.text(row.x,row.y,f"{row.prevalence*100:.1f}",ha="center",va="center",fontsize=7)
    cbar = fig.colorbar(scatter,ax=ax,fraction=0.035,pad=0.03); cbar.set_label("Mean sentiment score")
    levels=[0.05,0.20,0.40]
    handles=[ax.scatter([],[],s=90+1450*(level/max_prev),facecolors="none",edgecolors="black",linewidths=0.7) for level in levels]
    fig.legend(handles,[f"{level*100:.0f}%" for level in levels],title="Topic prevalence",frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.01))
    fig.subplots_adjust(bottom=0.16)
    save(fig, out, "Figure_5_Topic_Prevalence_and_Sentiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper tables, tests, and figures from the frozen deidentified feature matrix.")
    parser.add_argument("--data", default="data/derived/comment_level_features.csv.gz")
    parser.add_argument("--derived", default="data/derived")
    parser.add_argument("--tables", default="outputs/tables")
    parser.add_argument("--figures", default="outputs/figures/generated")
    args = parser.parse_args()

    df = read_features(args.data)
    tables_dir = Path(args.tables); tables_dir.mkdir(parents=True, exist_ok=True)
    tests = sentiment_tests(df)
    reg = logistic_regression(df)

    table_1(df).to_csv(tables_dir / "Table_1_Corpus_Characteristics.csv", index=False)
    cleaning = pd.DataFrame([
        {"processing_stage":"Raw parsed records", "records_retained":31164, "records_removed_at_stage":0, "percentage_retained":100.00},
        {"processing_stage":"Unique comment identifiers", "records_retained":31150, "records_removed_at_stage":14, "percentage_retained":99.96},
        {"processing_stage":"Nonempty normalised text", "records_retained":30775, "records_removed_at_stage":375, "percentage_retained":98.75},
        {"processing_stage":"English or language indeterminate comments", "records_retained":30772, "records_removed_at_stage":3, "percentage_retained":98.74},
        {"processing_stage":"Comments containing at least three words", "records_retained":28902, "records_removed_at_stage":1870, "percentage_retained":92.74},
    ])
    cleaning.to_csv(tables_dir / "Table_2_Data_Cleaning.csv", index=False)
    table_3(df).to_csv(tables_dir / "Table_3_Condition_Sentiment.csv", index=False)
    table_4(df).to_csv(tables_dir / "Table_4_Discursive_Themes.csv", index=False)
    pd.read_csv(Path(args.derived)/"topic_summary_labelled.csv").to_csv(tables_dir / "Table_5_Topic_Model.csv", index=False)
    table_6(tests, reg).to_csv(tables_dir / "Table_6_Hypothesis_Tests.csv", index=False)
    tests.to_csv(tables_dir / "Statistical_Tests_Reproduced.csv", index=False)
    reg.to_csv(tables_dir / "Logistic_Regression_Reproduced.csv", index=False)
    make_figures(df, reg, Path(args.figures))

    summary = {
        "analytical_comments": int(len(df)),
        "videos": int(df["video_id"].nunique()),
        "unique_authors": int(df["author_key"].nunique()),
        "sentiment_pct": (df["sentiment"].value_counts(normalize=True)*100).round(4).to_dict(),
    }
    (tables_dir / "reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Tables written to {tables_dir}")
    print(f"Figures written to {args.figures}")


if __name__ == "__main__":
    main()
