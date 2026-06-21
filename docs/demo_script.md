# Flow Pitch & Demo Script

This script aligns the narrative with the pitch deck slides and the web application demo. 

## Presenter Notes
- **Pacing**: Speak at a steady, measured pace. The focus is on the *why* behind the numbers, not just the numbers themselves.
- **Transitions**: The script starts with the presentation deck to set the context and architecture, then transitions into the live web application to show the real-time MILP optimizer.

---

### [Visual: Slide 3 - What We Found in the Dataset]

**Speaker:**
Every year, Bangalore loses millions of hours to traffic that nobody saw coming. Out of around 8,100 incidents in this dataset, it's not the routine breakdowns that break the city — it's the rare ones that cross two hours. 

### [Visual: Slide 5 - Defining The Real Objective (Highlighting 2hr+ Critical Gridlock)]

**Speaker:**
A breakdown that clears in twenty minutes is a footnote. A breakdown that turns into a three-hour, unmanaged gridlock on a major corridor paralyzes emergency routes, stalls supply chains, and costs the city in ways nobody puts a number on until it's already happened.

### [Visual: Slide 2 - The Operational Gap (Reactive Operations)]

**Speaker:**
And today, deciding where to send barricades and officers for those events is gut feeling. Whoever's on shift, whoever's been doing this longest, makes the call. There's no system that learns from the eight thousand incidents that came before it. We wanted to build that system. 

### [Visual: Slide 4 - The First Failure (100% Accuracy Leak)]

**Speaker:**
But before we tell you what we built, we want to tell you what we found wrong with how everyone — including us, at first — builds these. Most teams chase one accuracy number and stop there. We did too, in our first pass. Then we asked a harder question: what happens to the model on the events that actually matter, not the average event? 

### [Visual: Slide 8 - We Tested Everything (7 Model Families)]

**Speaker:**
So we tested every model family this dataset could support — boosted trees, random forests, classical baselines like Naive Bayes and logistic regression — and once we knew which families were strongest, we didn't just pick the best single model. We systematically searched 230 different two-way, three-way, and four-way blends between them, because the real question was never "which model wins," it was "which combination actually catches the events we can't afford to miss."

### [Visual: Slide 9 - The Final Architecture (Tree Ensemble + GNN + AutoML)]

**Speaker:**
The pattern that kept winning was clear: a tree ensemble alone capped out around 55% accuracy, but it missed critical events at a rate that simply wasn't acceptable for an operational tool. Good on a leaderboard, dangerous in the field. So we kept pushing. Adding a graph neural network — one that learns which corridors historically break down together, not from a map, but from the data itself — changed the picture. Even a small fifteen percent share of GNN, blended with the tree backbone and a touch of AutoGluon, pushed us to 55.3% accuracy and 73.54% recall on the gridlocks that actually matter. The trees handle the tabular signal — the categorical, time-based, intake-level patterns. The graph handles the thing trees structurally can't see: spatial spillover, the way one corridor's failure quietly sets up the next one.

### [Visual: Slide 7 - Building A City Graph From The Dataset]

**Speaker:**
That graph network isn't decoration sitting on top of a slide — it's the reason recall jumped. It learns corridor relationships purely from this dataset, with no Google Maps, no OSMnx, nothing external pulled in from outside the problem. 

### [Visual: Switch to Web App - Network Graph Tab]
*(Action: Briefly show the Network Graph tab in the UI, demonstrating the connected nodes/corridors that the GNN learned.)*

**Speaker:**
And that same spatial graph doesn't stop at forecasting — it powers our diversion routing too. When one corridor fails, we don't route traffic around it using a generic map's idea of nearby roads. We route around the roads that have *historically* failed alongside it, because that's the pattern that's actually true for this city, not an assumption borrowed from somewhere else.

### [Visual: Web App - Command Center & Resource Optimizer Tabs]
*(Action: Open the live dashboard. Trigger a severe incident via the websocket replay so the UI populates with a critical alert. Switch to the Resource Optimizer tab to show barricades and officers being allocated.)*

**Speaker:**
From there, the forecast doesn't just sit as a number on a dashboard — it feeds a live MILP optimizer that allocates real police stations and barricade counts under real capacity constraints. Not a lookup table pretending to be intelligent, an actual constrained optimization running against predicted severity in real time. All of it streamed live over a websocket, fed by chronological incident replay, so forecast, dispatch, and diversion run end to end, close to real time, the way it would have to work if a city actually deployed this tomorrow.

### [Visual: Web App - Analytics Dashboard / Slide 12 - Results]
*(Action: Show the Analytics page displaying the 73.5% recall metric alongside the accuracy)*

**Speaker:**
And here's why we built it this way instead of chasing a bigger headline number: missing a real gridlock is catastrophic — blocked emergency routes, stalled supply chains, a city that grinds for hours instead of minutes. Sending one extra patrol bike to a minor breakdown costs almost nothing. Those two mistakes are not the same size, so we refused to optimize like they were. We optimized for that real-world asymmetry, not the number that looks best on a leaderboard.

### [Visual: Slide 13 - The Bigger Insight]

**Speaker:**
Anyone in this room can report a number by the end of today. What we're showing you instead is the audit trail behind ours — the leaks we found and fixed in our own code before they could quietly inflate our score, the architectures we tested and rejected, and the one we kept, not because it sounded the most impressive in a pitch, but because it measured better on the metric that actually matters when a city is on fire.

### [Visual: Slide 1 - Title Slide / Final Team Screen]

**Speaker:**
We're Team Humble Bullies. We didn't try to win this with the biggest number. We tried to win it with the only one you can actually trust.
