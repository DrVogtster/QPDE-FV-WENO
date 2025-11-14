import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 

def place_legend_smart(ax=None, ngrid=80, candidates_per_axis=6, legend_kwargs=None):
    """
    Auto-place an inside-the-axes legend where curve density is lowest.
    - ax: Matplotlib Axes (default: current axes)
    - ngrid: resolution of density grid (higher -> finer)
    - candidates_per_axis: how many candidate positions per axis (>= 4 recommended)
    - legend_kwargs: kwargs passed to ax.legend; handles/labels taken from plotted artists
    """
    if ax is None:
        ax = plt.gca()
    if legend_kwargs is None:
        legend_kwargs = {}

    # 1) Gather line data in axes coordinates
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    def to_axes_xy(x, y):
        # Normalize data (x,y) to [0,1] in axes coords
        xn = (x - xlim[0]) / (xlim[1] - xlim[0])
        yn = (y - ylim[0]) / (ylim[1] - ylim[0])
        return xn, yn

    density = np.zeros((ngrid, ngrid), dtype=float)
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        # Filter finite and within bounds
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            continue
        xn, yn = to_axes_xy(x, y)
        m2 = (xn >= 0) & (xn <= 1) & (yn >= 0) & (yn <= 1)
        xn, yn = xn[m2], yn[m2]
        if xn.size == 0:
            continue
        # Bin into grid
        ix = np.clip((xn * (ngrid - 1)).astype(int), 0, ngrid - 1)
        iy = np.clip(((1 - yn) * (ngrid - 1)).astype(int), 0, ngrid - 1)  # invert y for image row
        # Increment density
        np.add.at(density, (iy, ix), 1.0)

    # 2) Make a temporary legend to measure its size in axes coords
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        # If no legend entries exist yet, try building from lines
        handles = ax.lines
        labels = [l.get_label() for l in handles]
    # Create a temp legend (off-canvas) to measure bbox
    temp_leg = ax.legend(handles, labels, loc="upper left", framealpha=0.7, **{k:v for k,v in (legend_kwargs or {}).items() if k not in ("loc","bbox_to_anchor")})
    fig = ax.figure
    fig.canvas.draw()  # needed to get a valid renderer & bbox
    leg_bbox_disp: Bbox = temp_leg.get_window_extent(fig.canvas.get_renderer())
    # Convert bbox from display coords -> axes coords
    inv = ax.transAxes.inverted()
    leg_bbox_axes = leg_bbox_disp.transformed(inv)
    leg_w = leg_bbox_axes.width
    leg_h = leg_bbox_axes.height
    # Remove temp legend
    temp_leg.remove()

    # Clamp in case fonts are huge/small
    leg_w = float(np.clip(leg_w, 0.1, 0.6))
    leg_h = float(np.clip(leg_h, 0.06, 0.5))

    # 3) Evaluate candidate positions on a grid, staying fully inside [0,1]x[0,1]
    xs = np.linspace(0.02, 0.98 - leg_w, candidates_per_axis)
    ys = np.linspace(0.02, 0.98 - leg_h, candidates_per_axis)
    best = None  # (score, x0, y0)

    # Precompute integral image (summed area table) for O(1) region sum
    sat = density.cumsum(axis=0).cumsum(axis=1)
    def sat_sum(y0i, x0i, y1i, x1i):
        # sums density[y0:y1, x0:x1] inclusive-exclusive with boundaries
        a = sat[y1i-1, x1i-1]
        b = sat[y0i-1, x1i-1] if y0i > 0 else 0
        c = sat[y1i-1, x0i-1] if x0i > 0 else 0
        d = sat[y0i-1, x0i-1] if (y0i > 0 and x0i > 0) else 0
        return a - b - c + d

    for y0 in ys:
        for x0 in xs:
            # Convert legend rect [x0,x0+leg_w]x[y0,y0+leg_h] in axes coords -> density grid indices
            x0i = int(np.clip(np.floor(x0 * ngrid), 0, ngrid - 1))
            x1i = int(np.clip(np.ceil((x0 + leg_w) * ngrid), 1, ngrid))
            # y-axis is inverted in the density image index
            # Axes y0 (bottom) -> image row start
            y_bottom = y0
            y_top = y0 + leg_h
            iy0 = int(np.clip(np.floor((1 - y_top) * ngrid), 0, ngrid - 1))
            iy1 = int(np.clip(np.ceil((1 - y_bottom) * ngrid), 1, ngrid))

            overlap = sat_sum(iy0, x0i, iy1, x1i)
            # Add a slight preference for corners (less distracting)
            corner_bonus = 0.0
            if x0 in (xs[0], xs[-1]) and y0 in (ys[0], ys[-1]):
                corner_bonus = -0.05  # encourage corners if tie
            score = overlap + corner_bonus

            if (best is None) or (score < best[0]):
                best = (score, x0, y0)

    _, best_x, best_y = best

    # 4) Finally place the legend anchored at that spot, inside the axes
    # We use loc='upper left' so the bbox_to_anchor = (x0, y0 + leg_h) aligns the top-left corner.
    # But since we already accounted for height, we can simply use loc='lower left' with (best_x, best_y).
    final_kwargs = dict(legend_kwargs or {})
    # Sensible defaults for readability
    final_kwargs.setdefault("framealpha", 0.7)
    final_kwargs.setdefault("fontsize", 9)

    ax.legend(loc="lower left", bbox_to_anchor=(best_x, best_y), **final_kwargs)
    # Optional: redraw
    ax.figure.canvas.draw_idle()


nx_sizes = [16,32,64,128,256,512,1024]
tests=[1,2,3]
title=["smooth","sod","lax"]

for test in tests:
    for nx_size in nx_sizes:
        with open("classical" + str(nx_size) + "test=" + str(test) + ".pkl", "rb") as f:
            (x_fine_class, uexact1_fine, uexact2_fine, uexact3_fine,
             x_class, uexact1_reg, uexact2_reg, uexact3_reg,
             u1_num_reg, u2_num_reg, u3_num_reg) = pickle.load(f)

        with open("quantum" + str(nx_size) + "instance" + str(test) + ".pkl", "rb") as f:
            (x_quan, uexact1_quan, uexact2_quan, uexact3_quan,
             u1_num_quan, u2_num_quan, u3_num_quan) = pickle.load(f)

        if(nx_size!=16 and test !=1):
            # Plot 1
            plt.plot(x_quan, u1_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u1_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact1_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\rho(x,T)$")
            plt.legend(loc="best", fontsize=9, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum1-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()

            # Plot 2
            plt.plot(x_quan, u2_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u2_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact2_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\rho(x,T)u(x,T)$")
            plt.legend(loc="best", fontsize=9, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum2-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()

            # Plot 3
            plt.plot(x_quan, u3_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u3_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact3_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$E(x,T)$")
            plt.legend(loc="best", fontsize=9, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum3-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()
        else:
                # Plot 1
            plt.plot(x_quan, u1_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u1_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact1_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\rho(x,T)$")
            plt.legend()
            place_legend_smart(plt.gca())
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum1-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()

            # Plot 2
            plt.plot(x_quan, u2_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u2_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact2_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$\rho(x,T)u(x,T)$")
            plt.legend()
            place_legend_smart(plt.gca())
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum2-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()

            # Plot 3
            plt.plot(x_quan, u3_num_quan, 'ro', label="Quantum PDE")
            plt.plot(x_class, u3_num_reg, 'bx', label="Classical RK4")
            plt.plot(x_class, uexact3_reg, 'k', label="Exact")
            plt.xlabel(r"$x$")
            plt.ylabel(r"$E(x,T)$")
            plt.legend()
            place_legend_smart(plt.gca())
            plt.tight_layout()
            plt.savefig(f"{title[test-1]}quantum3-{nx_size}instance{test}.pdf", bbox_inches="tight")
            plt.clf()