# VC Investments Data Analytics

Exploratory and predictive analysis of ~54,000 global startup funding records from Crunchbase.

## Business Questions

1. Which countries and industries attract the most VC capital?
2. How has startup funding volume and deal size evolved over time?
3. What funding patterns characterize startups that get acquired vs. those that close?
4. Can we predict whether a startup will be acquired based on its funding profile?

## Dataset

**Source:** `investments_VC.csv` — Crunchbase startup funding data  
**Size:** ~54,000 companies, 39 columns  
**Key fields:** country, market/industry, funding total, funding rounds (seed through Series F), company status (operating / acquired / closed / IPO)

## Methodology

| Step | Description |
|------|-------------|
| 1. Load & Explore | Shape, column types, missing value audit |
| 2. Clean | Numeric parsing, date coercion, whitespace stripping |
| 3. EDA — Countries | Top 15 by startup count and total funding |
| 4. EDA — Trends | Funding count and volume per year (1990–2015) |
| 5. EDA — Markets | Top industries by count and total dollars raised |
| 6. EDA — Rounds | Distribution across Seed, Angel, Series A–F, etc. |
| 7. EDA — Status | Operating / acquired / closed breakdown + median funding |
| 8. Key Insights | Summary of EDA findings |
| 9. ML Model | Binary classification: acquired vs. closed |

## Key Findings

### Geography
- The **USA dominates** with 28,793 startups — nearly 11x the UK (2nd place, 2,642).
- US startups raised **$464B total** vs. China’s $35.6B (2nd place).

### Trends
- VC activity **peaked in 2013** (~8,972 startups funded in one year).
- Total capital peaked in **2010 at $101.8B**, suggesting deal sizes grew faster than deal counts in early years.

### Industries
- **Software** is the most common market (4,620 companies), but **Biotechnology attracts the most capital** ($73.4B) — reflecting its capital-intensive nature.

### Predicting Acquisition
- Trained a **Random Forest classifier** to predict acquired vs. closed startups.
- **Total funding amount** is the strongest predictor — acquired companies raised a median of **$8.4M vs. $1M** for closed ones.
- Model achieved **ROC-AUC ~0.73**, meaningfully above baseline, demonstrating that funding profile is a real signal for acquisition outcomes.

## Technologies

- **Python 3** — pandas, NumPy, Matplotlib
- **scikit-learn** — Logistic Regression, Random Forest, ROC-AUC evaluation
- **Jupyter Notebook**
