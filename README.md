# Foraging Behavior Simulation – Marginal Value Theorem

A computational implementation of the Marginal Value Theorem (MVT) for studying optimal patch-leaving decisions in depleting resource environments.

## Background

The MVT (Charnov, 1976) provides a normative account of patch-leaving behavior: a forager maximizing long-run intake rate should leave a patch when its instantaneous gain rate equals the average rate achievable across the environment, including travel time between patches. This prediction has been tested across a wide range of species and serves as a foundational benchmark in both behavioral ecology and computational models of decision-making.

This project implements the MVT computationally and examines how the optimal leaving time shifts under varying patch quality and travel cost. A secondary motivation was to connect the MVT framework to questions about belief-updating timescales: if a forager's estimate of its current intake rate lags behind the true rate, this produces a directional bias toward overstaying, whose magnitude depends on the ratio between the patch depletion timescale and the agent's internal updating timescale.

## Mathematical Model

Cumulative energy gain in a patch follows a diminishing-returns function:

    g(t) = G_max * (1 - exp(-lambda * t))

where G_max is the maximum available resource and lambda is the depletion rate. The optimal leaving time t* satisfies the marginal condition:

    g'(t*) = g(t*) / (t* + t_travel)

This is equivalent to finding the point on the gain curve where the tangent line drawn from -t_travel on the time axis touches the curve. The slope of this tangent gives the maximum achievable long-run intake rate. The optimal leaving time is computed numerically as the root of:

    g'(t) * (t + t_travel) - g(t) = 0

## Simulations

- Classic MVT diagram with optimal tangent-line construction
- Effect of travel time on optimal leaving threshold: longer travel time shifts t* upward
- Patch quality comparison: richer patches with slower depletion rates warrant longer residence times
- Instantaneous gain rate dynamics and crossing of the environmental average threshold

## Key Results

- Optimal residence time increases monotonically with travel time between patches
- Richer patches do not always imply longer stays; the depletion rate lambda matters equally
- Belief-updating lag produces systematic overstaying relative to the instantaneous MVT benchmark, with magnitude scaling as a/alpha, where a is the depletion rate and alpha is the belief-updating rate

## Usage

    pip install numpy matplotlib scipy
    python mvt.py

## References

Charnov, E. L. (1976). Optimal foraging: the marginal value theorem. Theoretical Population Biology, 9(2), 129-136.

Stephens, D. W., & Krebs, J. R. (1986). Foraging Theory. Princeton University Press.

Constantino, S. M., & Daw, N. D. (2015). Learning the opportunity cost of time in a patch-foraging task. Cognitive, Affective, & Behavioral Neuroscience, 15(4), 837-853.

Niv, Y., Daw, N. D., Joel, D., & Dayan, P. (2007). Tonic dopamine: opportunity costs and the control of response vigor. Psychopharmacology, 191(3), 507-520.
