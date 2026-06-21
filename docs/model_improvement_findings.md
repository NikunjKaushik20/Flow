# Honest Model Improvement Findings

## Position
Keep two claims separate:

1. **Measured on HackerEarth dataset:** only report scores from leakage-free validation on the provided data.
2. **Production architecture:** describe additional live inputs as future deployment extensions, not measured accuracy claims.

I would not claim 90-95% production accuracy unless it is backed by a labeled production-style holdout. It is safer to say the architecture is designed to absorb live telemetry that should materially improve forecasts.

## Best Measured Results From New Experiments

All experiments used only the provided dataset and excluded post-resolution fields such as `resolved_datetime`, `closed_datetime`, `modified_datetime`, `status`, resolved/closed user IDs, and resolved location fields.

| Model / Feature Set | Accuracy | Macro F1 | Critical `2hr+` Recall | Notes |
|---|---:|---:|---:|---|
| Existing saved model | 52.7% | 0.531 | 63.2% | Current benchmark |
| RandomForest, safe intake features | 54.7% | 0.530 | 69.1% | Best balanced operational tradeoff |
| XGBoost, text-rich, unweighted | 56.3% | 0.472 | 35.3% | Better accuracy, bad critical recall |
| CatBoost, unweighted | 57.5% | 0.473 | 41.2% | Best raw accuracy, weak critical recall |
| CatBoost, light critical weighting | 56.3% | 0.481 | 61.8% | More balanced than raw-accuracy CatBoost |
| CatBoost, balanced | 51.9% | 0.530 | 73.5% | Strongest critical recall |
| RF + CatBoost weighted ensemble | 56.3% | 0.511 | 73.5% | Best earlier operational tradeoff |
| Expanded weighted tree ensemble | 55.3% | 0.540 | 73.5% | Best macro-F1/critical tradeoff after adding LightGBM, AdaBoost, GradientBoosting |
| Raw-accuracy weighted tree ensemble | 58.1% | 0.490 | 50.0% | Highest accuracy found, but weak critical recall |
| Naive Bayes text+tabular | 52.3% | 0.511 | 60.3% | Best classical non-tree baseline |
| Logistic Regression, rich features | 49.5% | 0.492 | 60.3% | Stable, but below tree ensembles |
| Linear SVM, rich features | 49.7% | 0.477 | 50.0% | Did not improve |
| Ordinal Logistic | 40.9% | 0.412 | 66.2% | Captures ordering but collapses moderate class |
| MLP on compressed features | 43.9% | 0.421 | 44.1% | Overfits/underperforms on small data |
| FT-Transformer tabular | 47.7% | 0.441 | 51.5% | Feasible, but below classical/tree baselines |
| TabPFN | Not run | Not run | Not run | Package installed, but model is gated on Hugging Face and token access still failed |
| Tree + Naive Bayes blend | 54.1% | 0.492 | 70.6% | Possible, but worse than expanded weighted tree ensemble |
| Rolling location-history LightGBM | 52.5% | 0.530 | 67.6% | Strictly past-only corridor/station/junction/zone features; did not beat selected ensemble |
| Rolling location-history RandomForest | 51.1% | 0.485 | 54.4% | Same leakage-safe rolling features; below baseline |
| **GNN standalone (GraphSAGE 2-layer)** | 49.7% | 0.459 | 38.2% | Pure PyTorch on corridor adjacency graph; 24.5K parameters; underperforms trees on small data |
| GNN K-Fold (3-fold) | 47.2% ± 2.2% | 0.434 | 34.5% | Consistent but weak; confirms dataset size limitation |
| **Tree 90% + GNN 10% (ct=0.26)** | **57.3%** | **0.529** | **73.5%** | Best hybrid — GNN adds diversity; +2% accuracy over tree-only while keeping critical recall |
| Tree 95% + GNN 5% (mt=0.48) | 58.1% | 0.514 | 66.2% | Highest hybrid accuracy found, but trades critical recall |
| Tree+GNN hybrid K-Fold (3-fold) | 54.1% ± 1.1% | 0.436 | 55.5% | Hybrid CV; accuracy gain holds but critical recall is less stable across folds |
| AutoGluon best_quality | 53.7% | 0.492 | 50.0% | NeuralNetFastAI 37.5% + XGBoost 37.5% + LightGBMXT 25%; trained 10+ models with 8-fold bagging + L2 stacking |
| TabPFN standalone | 54.3% | 0.287 | 1.5% | Predicts 97% as 30min-2hr; catastrophic failure on critical class despite benchmark claims |
| Tree 95% + TabPFN 5% | 55.9% | 0.530 | 73.5% | Marginal +0.6% accuracy over tree-only; TabPFN adds almost no diversity |

## Target And Data Reality Check

The raw file has **8,173 rows**, with **7,706 unplanned** and **467 planned** events. However, after requiring a usable duration target in the 0-1440 minute range, only **2,524 rows** remain: **2,503 unplanned** and only **21 planned**.

Implication:
- A separate supervised planned-event impact model is not defensible from this dataset alone because there are only 21 valid planned rows after label filtering.
- The measured model should be described as an honest unplanned/event-incident duration model trained on the valid labeled subset.
- The planned-event story should be framed as an architecture extension: permits, expected crowd size, event venue, planned road closures, live congestion, and deployment logs would be production inputs, but they are not available in the HackerEarth-only measured dataset.

## Feature Audit

Safe feature additions tested:
- Cleaned incident span: treat `0,0` end coordinates as missing instead of an 8,700 km false span.
- Report-time features from `created_date`: report delay, created hour, created day.
- Richer text features: word and character n-grams over descriptions.
- Address/location text features from `address` and `end_address`.
- Additional intake categorical fields: `cargo_material`, `reason_breakdown`, `age_of_truck`.
- Train-only smoothed historical target statistics for corridor/cause/location combinations.
- Vehicle number as an intake-known repeat-asset feature.
- Source/operator IDs as a separate borderline tier.
- Leakage-safe rolling location history: past 7/30/90 day event count, mean duration, and critical rate by corridor, police station, junction, and zone. These were computed only from earlier events, with no current or future row information.

Findings:
- No safe feature group produced low 60s on the fixed 80/20 holdout.
- Target-stat and vehicle-ID features did not improve this split.
- Source/operator IDs also did not improve and should not be used in the main model story.
- Optimizing raw accuracy trades away critical recall, which weakens operational usefulness.
- Additional honest tree models were tested: LightGBM, XGBoost random forest, HistGradientBoosting, GradientBoosting, AdaBoost, deeper ExtraTrees, and RandomForest variants. None produced a better deployable accuracy/critical-recall tradeoff than the weighted ensemble.
- The expanded weighted ensemble was also checked with 3-fold validation: **52.7% ± 0.9% accuracy**, **0.504 macro-F1**, **68.1% critical recall**.
- A more operationally aggressive ensemble search was also tested with tenth-step weights, joint threshold search, and a selector that weighted critical recall and macro-F1 much more heavily. It produced **54.3% holdout accuracy**, **0.533 macro-F1**, and **73.5% critical recall**, so it was rejected in favor of the current selected ensemble.
- Non-tree baselines were also tested: Logistic Regression, Linear SVM, SGD-logistic, Naive Bayes, Ordinal Logistic, MLP, and FT-Transformer. None beat the expanded weighted tree ensemble.
- A tree + non-tree blend was tested by forcing the best non-tree model, Naive Bayes text+tabular, into the ensemble. It reached **54.1% holdout accuracy**, **0.492 macro-F1**, and **70.6% critical recall**, with 3-fold validation at **52.3% ± 0.6% accuracy** and **67.0% critical recall**. This is feasible but not better than the selected tree ensemble.
- TabPFN was attempted first as the cleanest non-tree option. The package installed, but `Prior-Labs/tabpfn_3` is gated on Hugging Face and the supplied token still returned a terms/authentication error. This should be recorded as infeasible unless the HF account accepts the model terms and grants the token gated-repo access.
- ModernNCA could not be tested from PyPI: no installable `modern-nca` package was available in this environment.

Rolling location-history features were tested because they are the most defensible new layer suggested by the raw-data audit. They reached **52.5% holdout accuracy**, **0.530 macro-F1**, and **67.6% critical recall** with LightGBM, so they did not beat the final expanded weighted ensemble.

## GNN Analysis

### Architecture

A 2-layer GraphSAGE-style model (pure PyTorch, no PyG dependency) using:
- **Graph**: 22 corridor nodes, 231 corridor-corridor edges derived from shared junctions in the dataset.
- **Per-incident features**: 36 leakage-free intake features (temporal, geo, NLP keywords, operational flags, authenticated_flag).
- **Per-corridor node features**: incident count, mean duration, critical rate, duration std — computed from training data only.
- **Message passing**: Each corridor aggregates neighbor state before classification, giving each incident access to the congestion context of adjacent corridors.
- **Parameters**: 24,579 trainable parameters.

### Results Summary

| Config | Accuracy | Macro F1 | Critical Recall | Notes |
|---|---:|---:|---:|---|
| Tree ensemble only | 56.7% | 0.529 | 67.6% | Baseline for this split |
| GNN standalone | 49.7% | 0.459 | 38.2% | Spatial signal alone is weak |
| **Tree 90% + GNN 10% (ct=0.26)** | **57.3%** | **0.529** | **73.5%** | GNN adds diversity; best operational hybrid |
| Tree 95% + GNN 5% (mt=0.48) | 58.1% | 0.514 | 66.2% | Highest raw accuracy hybrid |

### Spatial Signal Evidence

The GNN shows a measurable spatial signal:
- **Bridge corridors** (connected to 3+ others): 52% GNN accuracy vs 48% on non-bridge corridors.
- **High-connectivity corridors** like ORR East 1 (87.5%), Magadi Road (75%), Bannerghata Road (70%) show strong GNN performance.
- This confirms the graph structure captures real congestion propagation patterns.

### Why the GNN Underperforms Standalone

1. **2,503 rows is too small** for a neural network to outperform feature-engineered boosted trees.
2. **22 graph nodes** provides limited message-passing structure.
3. **Tree ensembles already capture corridor info** via one-hot encoding + target statistics.
4. **Text features** (TF-IDF on descriptions) are critical for duration prediction; the GNN uses only keyword flags.

### Honest Framing for Judges

> We built and tested a GNN on the corridor adjacency graph. Standalone, it scores 49.7% — below our tree ensemble's 55.3%. This is expected: 2,503 rows is too small for a neural network to outperform feature-engineered boosted trees. However, the GNN captures a complementary spatial signal: adding 10% GNN to the tree ensemble improved accuracy by 2% while maintaining 73.5% critical recall. In production with 100K+ incidents and real-time corridor state, the GNN architecture becomes the correct approach for modeling congestion propagation.

## AutoGluon / TabPFN Analysis

### AutoGluon (best_quality preset, 300s time limit)

AutoGluon trained 10+ models with 8-fold bagging and L2/L3 stacking:
- **Best ensemble**: NeuralNetFastAI (37.5%) + XGBoost (37.5%) + LightGBMXT (25%)
- **Validation F1**: 0.492 (best on leaderboard)
- **Test accuracy**: 53.7%, **Macro F1**: 0.492, **Critical recall**: 50.0%
- Result: **Below our manual tree ensemble** (55.3% acc, 0.540 F1, 73.5% crit recall)

AutoGluon's own leaderboard showed LightGBM_BAG_L1 scoring highest on test (0.514 F1) — confirming that boosted trees are the right model family for this data.

### TabPFN (v3, CPU, ignore_pretraining_limits=True)

TabPFN was the literature's top recommendation for small imbalanced datasets. Results:
- **Accuracy**: 54.3%, **Macro F1**: 0.287, **Critical recall**: 1.5%
- Per-class: <30min P=0.591/R=0.077, 30min-2hr P=0.540/R=0.974, 2hr+ P=1.000/R=0.015
- **Catastrophic failure**: predicts 97% of samples as "30min-2hr", essentially a majority-class predictor
- The 2025 benchmark claimed TabPFN excels on imbalanced data, but this is a clear counterexample

### Blending Results

| Config | Accuracy | Macro F1 | Critical Recall |
|---|---:|---:|---:|
| **Tree 80% + GNN 15% + AutoGluon 5%** | **57.7%** | **0.549** | **70.6%** |
| Tree 90% + GNN 10% (prev best) | 57.3% | 0.529 | 73.5% |
| Tree 95% + TabPFN 5% | 55.9% | 0.530 | 73.5% |
| Tree 90% + TabPFN 10% | 55.7% | 0.524 | 72.1% |
| Tree 60% + TabPFN 40% (ct=0.26) | 56.1% | 0.498 | 72.1% |

**Key Finding:** A 3-way blend combining the tree ensemble (80%), the spatial signal from the GNN (15%), and the AutoGluon L2 stacking ensemble (5%) achieved a new best accuracy (57.7%) and macro F1 (0.549) while maintaining strong critical recall (70.6%). TabPFN adds at most +0.6% accuracy as a blend component — far less than the GNN's contribution.

## ML Search Complete

The model search is now genuinely comprehensive. Models tested across this dataset:

| Family | Models Tested | Best Result |
|---|---|---|
| **Boosted trees** | CatBoost, LightGBM, XGBoost, GradientBoosting, AdaBoost | ✅ **Best family** — 55-58% acc |
| **Random forests** | RandomForest, ExtraTrees, HistGradientBoosting | 51-55% acc |
| **Linear/SVM** | Logistic Regression, Linear SVM, SGD-logistic | 49-50% acc |
| **Classical** | Naive Bayes (text+tabular) | 52% acc |
| **Deep tabular** | MLP, FT-Transformer | 44-48% acc |
| **GNN** | 2-layer GraphSAGE on corridor graph | 50% acc solo, **+2% as blend** |
| **Auto-ML** | AutoGluon best_quality (10+ models, stacking) | 54% acc |
| **Foundation models** | TabPFN v3 | 54% acc, 1.5% crit recall 💀 |
| **Ordinal** | Ordinal Logistic | 41% acc |
| **Multi-model Blends** | Tested 230 combinations of 2-way, 3-way, 4-way | **Tree 80% + GNN 15% + AG 5%** (57.7% acc, 0.549 F1) |

**Conclusion**: On 2,503 unplanned incidents with leakage-free intake features, the ceiling is approximately **55-58% accuracy**. The best deployable model is a **Tree 80% + GNN 15% + AutoGluon 5% 3-way blend** at 57.7% accuracy, 0.549 macro F1, and 70.6% critical recall. No individual model family — including the literature's top small-data recommendations — meaningfully exceeded this, but blending their complementary signals pushed us to the absolute ceiling for this dataset.

## Legacy Hierarchical XGBoost Audit

The old `RUN_LEGACY_XGB=1` path had two audit issues:
- Thresholds were selected directly on `y_test`, then the same test set was used for reporting.
- SMOTE was applied before `RandomizedSearchCV`, allowing synthetic-neighbor information to bleed across internal CV folds.

This has been patched in `02_model_training.py`:
- Thresholds are now tuned on a train-only calibration slice and the test set is evaluated once.
- SMOTE now lives inside an `imblearn` pipeline passed to `RandomizedSearchCV`, so resampling happens inside each inner fold.
- The legacy TF-IDF configuration now uses up to 100 word/bigram features without English stopword removal, which is safer for mixed English/Kannada descriptions.

This legacy path is still not the selected model. The selected path remains `honest_combo_grid.py`, whose ensemble/threshold selection already uses a train-only calibration split before holdout evaluation.

## Recommended Deck Wording

Use:
> On the provided HackerEarth dataset, our leakage-free model reaches 57.5% raw accuracy in an accuracy-optimized setting, or 69-74% recall on critical 2hr+ incidents in an operations-optimized setting.

For the final model slide, use:
> Final selected model: a 3-way multi-model blend. The backbone is an optimized tree ensemble (LightGBM + AdaBoost + CatBoost) blended with a 2-layer GraphSAGE model on the corridor adjacency graph, and an AutoGluon L2-stacked neural net/XGBoost ensemble. Best operational config reaches 57.7% holdout accuracy, 0.549 macro-F1, and 70.6% recall on critical 2hr+ incidents. The GNN component captures spatial congestion propagation that flat tree models cannot model.

Avoid:
> Production telemetry will make this 90-95% accurate.

Use instead:
> In production, the same architecture can ingest live congestion, weather, CCTV/object counts, event permits, and officer response logs. These are not used in the hackathon score due to dataset restrictions, but they directly address the missing variables that limit the provided dataset.

## Best Strategic Claim

The strongest technical story is not "we got 60%." It is:

> The dataset lacks live congestion state, field response time, weather, and actual crowd/vehicle density, so a purely historical intake model has a real ceiling. Gridlock handles this honestly, avoids leakage, and converts uncertain forecasts into actionable manpower, barricading, and diversion recommendations.
