"""
Foraging behavior simulation based on the Marginal Value Theorem (MVT).

The MVT (Charnov, 1976) predicts that an optimal forager should leave a patch
when the instantaneous rate of energy gain drops to the average rate achievable
across the environment (including travel time between patches).

Model:
    - Gain function: g(t) = G_max * (1 - exp(-lambda * t))
    - Optimal leaving time t* satisfies: g'(t*) = (g(t*)) / (t* + t_travel)
    - This is equivalent to the tangent line condition from the origin shifted
      by t_travel on the time axis.

Author: Esma Dogrusozlu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ── Gain function ─────────────────────────────────────────────────────────────

def gain(t, G_max, lam):
    """
    Cumulative energy gain in a patch over time t.

    g(t) = G_max * (1 - exp(-lambda * t))

    Parameters
    ----------
    t     : array-like, time spent in patch
    G_max : float, maximum energy available in patch
    lam   : float, depletion rate (lambda)
    """
    return G_max * (1 - np.exp(-lam * t))


def gain_rate(t, G_max, lam):
    """Instantaneous gain rate: dg/dt = G_max * lambda * exp(-lambda * t)"""
    return G_max * lam * np.exp(-lam * t)


# ── Optimal leaving time ───────────────────────────────────────────────────────

def optimal_leaving_time(G_max, lam, t_travel, t_max=50.0):
    """
    Find t* such that the instantaneous gain rate equals the average
    environmental rate:

        g'(t*) = g(t*) / (t* + t_travel)

    Rearranged as root-finding problem:
        g'(t) * (t + t_travel) - g(t) = 0
    """
    def equation(t):
        return gain_rate(t, G_max, lam) * (t + t_travel) - gain(t, G_max, lam)

    # Check if a root exists in (0, t_max)
    try:
        t_star = brentq(equation, 1e-6, t_max)
    except ValueError:
        t_star = None
    return t_star


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_mvt_tangent(G_max=10.0, lam=0.3, t_travel=2.0, ax=None):
    """
    Classic MVT diagram: gain curve with tangent line from -t_travel.
    The tangent point gives the optimal leaving time t*.
    """
    t = np.linspace(0, 20, 500)
    g = gain(t, G_max, lam)
    t_star = optimal_leaving_time(G_max, lam, t_travel)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Gain curve
    ax.plot(t, g, color='steelblue', linewidth=2, label='Cumulative gain $g(t)$')

    if t_star is not None:
        g_star = gain(t_star, G_max, lam)
        slope = g_star / (t_star + t_travel)

        # Tangent line from -t_travel
        t_line = np.linspace(-t_travel, t_star + 1, 200)
        ax.plot(t_line, slope * (t_line + t_travel),
                color='tomato', linewidth=1.5, linestyle='--',
                label=f'Optimal tangent (slope = {slope:.3f})')

        # Mark t*
        ax.axvline(t_star, color='gray', linewidth=1, linestyle=':')
        ax.scatter([t_star], [g_star], color='tomato', zorder=5, s=60)
        ax.annotate(f'$t^*$ = {t_star:.2f}',
                    xy=(t_star, g_star),
                    xytext=(t_star + 1, g_star - 1.5),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='gray'))

        # Mark travel time origin
        ax.axvline(-t_travel, color='gray', linewidth=0.8, linestyle=':')
        ax.scatter([-t_travel], [0], color='gray', zorder=5, s=40)
        ax.annotate(f'$-t_{{travel}}$ = {-t_travel}',
                    xy=(-t_travel, 0),
                    xytext=(-t_travel + 0.3, 0.8),
                    fontsize=9)

    ax.set_xlim(-t_travel - 0.5, 20)
    ax.set_ylim(-0.5, G_max + 1)
    ax.set_xlabel('Time in patch $t$', fontsize=11)
    ax.set_ylabel('Cumulative energy gain $g(t)$', fontsize=11)
    ax.set_title('Marginal Value Theorem – Optimal Patch Leaving Time', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    return ax


def plot_travel_time_effect():
    """
    Show how optimal leaving time t* changes with travel time t_travel.
    Longer travel → stay longer in patch (classic MVT prediction).
    """
    G_max, lam = 10.0, 0.3
    travel_times = [0.5, 1.0, 2.0, 4.0, 7.0]
    t = np.linspace(0, 25, 600)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Effect of Travel Time on Optimal Foraging Strategy (MVT)',
                 fontsize=13, fontweight='bold')

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(travel_times)))
    t_stars = []

    for t_travel, color in zip(travel_times, colors):
        g = gain(t, G_max, lam)
        t_star = optimal_leaving_time(G_max, lam, t_travel)
        t_stars.append(t_star)

        ax1.plot(t, g, color=color, linewidth=1.8,
                 label=f'$t_{{travel}}$ = {t_travel}')
        if t_star is not None:
            g_star = gain(t_star, G_max, lam)
            slope = g_star / (t_star + t_travel)
            t_line = np.linspace(-t_travel, t_star, 200)
            ax1.plot(t_line, slope * (t_line + t_travel),
                     color=color, linewidth=1, linestyle='--', alpha=0.7)
            ax1.scatter([t_star], [g_star], color=color, zorder=5, s=50)

    ax1.set_xlim(-8, 25)
    ax1.set_ylim(-0.5, G_max + 1)
    ax1.set_xlabel('Time in patch $t$', fontsize=11)
    ax1.set_ylabel('Cumulative gain $g(t)$', fontsize=11)
    ax1.set_title('Gain curves and optimal tangents', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.axvline(0, color='k', linewidth=0.5)

    # Right: t* vs t_travel
    ax2.plot(travel_times, t_stars, 'o-', color='steelblue',
             linewidth=2, markersize=7)
    ax2.set_xlabel('Travel time $t_{travel}$', fontsize=11)
    ax2.set_ylabel('Optimal leaving time $t^*$', fontsize=11)
    ax2.set_title('$t^*$ increases with travel time\n(MVT prediction)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('travel_time_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: travel_time_effect.png")


def plot_patch_quality_effect():
    """
    Compare optimal leaving times across patches of different quality
    (different G_max and depletion rates).
    """
    t_travel = 2.0
    patches = [
        {'label': 'Poor patch',   'G_max': 4.0,  'lam': 0.5, 'color': '#e07b54'},
        {'label': 'Medium patch', 'G_max': 8.0,  'lam': 0.3, 'color': '#5b8db8'},
        {'label': 'Rich patch',   'G_max': 14.0, 'lam': 0.2, 'color': '#4a9e6b'},
    ]

    t = np.linspace(0, 30, 600)
    fig, ax = plt.subplots(figsize=(9, 5))

    for p in patches:
        g = gain(t, p['G_max'], p['lam'])
        t_star = optimal_leaving_time(p['G_max'], p['lam'], t_travel)
        ax.plot(t, g, color=p['color'], linewidth=2, label=p['label'])
        if t_star is not None:
            g_star = gain(t_star, p['G_max'], p['lam'])
            ax.axvline(t_star, color=p['color'], linestyle=':', linewidth=1.2)
            ax.scatter([t_star], [g_star], color=p['color'], zorder=5, s=60)
            ax.annotate(f"$t^*$={t_star:.1f}",
                        xy=(t_star, g_star),
                        xytext=(t_star + 0.5, g_star - 1.2),
                        fontsize=9, color=p['color'])

    ax.set_xlabel('Time in patch $t$', fontsize=11)
    ax.set_ylabel('Cumulative energy gain $g(t)$', fontsize=11)
    ax.set_title(f'Patch Quality and Optimal Leaving Time ($t_{{travel}}$ = {t_travel})',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('patch_quality_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: patch_quality_effect.png")


def plot_gain_rate_dynamics():
    """
    Show instantaneous gain rate over time for different patches.
    The forager leaves when this rate crosses the environmental average.
    """
    t_travel = 2.0
    patches = [
        {'label': 'Poor patch',   'G_max': 4.0,  'lam': 0.5, 'color': '#e07b54'},
        {'label': 'Medium patch', 'G_max': 8.0,  'lam': 0.3, 'color': '#5b8db8'},
        {'label': 'Rich patch',   'G_max': 14.0, 'lam': 0.2, 'color': '#4a9e6b'},
    ]

    t = np.linspace(0.01, 25, 500)
    fig, ax = plt.subplots(figsize=(9, 5))

    for p in patches:
        gr = gain_rate(t, p['G_max'], p['lam'])
        t_star = optimal_leaving_time(p['G_max'], p['lam'], t_travel)
        ax.plot(t, gr, color=p['color'], linewidth=2, label=p['label'])
        if t_star is not None:
            env_rate = gain(t_star, p['G_max'], p['lam']) / (t_star + t_travel)
            ax.axhline(env_rate, color=p['color'], linestyle='--',
                       linewidth=1, alpha=0.6)
            ax.scatter([t_star], [gain_rate(t_star, p['G_max'], p['lam'])],
                       color=p['color'], zorder=5, s=60)

    ax.set_xlabel('Time in patch $t$', fontsize=11)
    ax.set_ylabel("Instantaneous gain rate $g'(t)$", fontsize=11)
    ax.set_title('Gain Rate Dynamics – Leave When Rate Hits Environmental Average',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig('gain_rate_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gain_rate_dynamics.png")


if __name__ == "__main__":
    print("Running MVT foraging simulation...\n")

    # 1. Classic MVT diagram
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_mvt_tangent(G_max=10.0, lam=0.3, t_travel=2.0, ax=ax)
    plt.tight_layout()
    plt.savefig('mvt_classic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: mvt_classic.png")

    # 2. Travel time effect
    plot_travel_time_effect()

    # 3. Patch quality comparison
    plot_patch_quality_effect()

    # 4. Gain rate dynamics
    plot_gain_rate_dynamics()

    print("\nAll figures saved.")
