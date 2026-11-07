/* ============================================================
   charts.js — thin Chart.js wrappers so page templates stay readable.
   Exposes window.BI.
   ============================================================ */
(function () {
  "use strict";

  const C = {
    accent: "#2dd4bf",
    cyan: "#38bdf8",
    good: "#34d399",
    warn: "#fbbf24",
    danger: "#fb7185",
    violet: "#a78bfa",
    muted: "#8ba0bd",
    grid: "rgba(139, 160, 189, 0.12)",
    text: "#c6d5e8",
  };

  if (window.Chart) {
    Chart.defaults.color = C.text;
    Chart.defaults.font.family =
      'Inter, -apple-system, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.font.size = 11.5;
    Chart.defaults.animation.duration = 500;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 7;
    Chart.defaults.plugins.legend.labels.boxHeight = 7;
    Chart.defaults.plugins.legend.labels.padding = 14;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10, 16, 24, 0.95)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(139,160,189,0.25)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.titleColor = "#e7eefb";
    Chart.defaults.plugins.tooltip.bodyColor = "#c6d5e8";
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hitRadius = 8;
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.line.tension = 0.35;
  }

  const axis = (opts) => Object.assign(
    {
      grid: { color: C.grid, drawTicks: false },
      border: { display: false },
      ticks: { color: C.muted, padding: 8, maxTicksLimit: 8 },
    },
    opts || {}
  );

  /** Fill a canvas with a translucent copy of a colour. */
  const alpha = (hex, a) => {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };

  const gradient = (ctx, color) => {
    const g = ctx.createLinearGradient(0, 0, 0, 260);
    g.addColorStop(0, alpha(color, 0.28));
    g.addColorStop(1, alpha(color, 0.02));
    return g;
  };

  function make(canvas, config) {
    if (!canvas || !window.Chart) return null;
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    return new Chart(canvas, config);
  }

  window.BI = {
    colors: C,
    alpha: alpha,

    /** Multi-series time / category line chart. */
    line(canvas, { labels, series, yTitle, xTitle, fillFirst = false, yMin, yMax }) {
      const ctx = canvas.getContext("2d");
      return make(canvas, {
        type: "line",
        data: {
          labels,
          datasets: series.map((s, i) => ({
            label: s.label,
            data: s.data,
            borderColor: s.color || [C.accent, C.cyan, C.warn, C.violet][i % 4],
            backgroundColor: fillFirst && i === 0 ? gradient(ctx, s.color || C.accent) : "transparent",
            fill: fillFirst && i === 0,
            borderDash: s.dashed ? [5, 4] : undefined,
            borderWidth: s.width || 2,
            pointRadius: 0,
            stepped: s.stepped || false,
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          scales: {
            x: axis({ title: xTitle ? { display: true, text: xTitle, color: C.muted } : undefined }),
            y: axis({
              beginAtZero: false,
              min: yMin,
              max: yMax,
              title: yTitle ? { display: true, text: yTitle, color: C.muted } : undefined,
            }),
          },
        },
      });
    },

    /** Vertical or horizontal bar chart. */
    bar(canvas, { labels, data, color, horizontal = false, yTitle, xTitle, valueSuffix = "" }) {
      const c = color || C.accent;
      return make(canvas, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: Array.isArray(c) ? c : alpha(c, 0.75),
            borderColor: Array.isArray(c) ? c : c,
            borderWidth: 1,
            borderRadius: 4,
            maxBarThickness: 34,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: horizontal ? "y" : "x",
          scales: {
            x: horizontal
              ? axis({ beginAtZero: true, title: { display: !!xTitle, text: xTitle, color: C.muted } })
              : axis({ grid: { display: false } }),
            y: horizontal
              ? axis({ grid: { display: false } })
              : axis({ beginAtZero: true, title: { display: !!yTitle, text: yTitle, color: C.muted } }),
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const v = horizontal ? ctx.parsed.x : ctx.parsed.y;
                  return `${v}${valueSuffix}`;
                },
              },
            },
          },
        },
      });
    },

    /** Pareto scatter with colour-by-comfort. */
    pareto(canvas, { cost, energy, ppd, onPick }) {
      const points = cost.map((c, i) => ({ x: c, y: energy[i], ppd: ppd[i] }));
      return make(canvas, {
        type: "scatter",
        data: {
          datasets: [{
            label: "Pareto-optimal day",
            data: points,
            backgroundColor: points.map((p) =>
              p.ppd <= 5.4 ? alpha(C.good, 0.85) : p.ppd <= 5.7 ? alpha(C.warn, 0.85) : alpha(C.violet, 0.85)
            ),
            pointRadius: 6,
            pointHoverRadius: 9,
            pointBorderColor: "rgba(10,16,24,0.9)",
            pointBorderWidth: 1.5,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          onClick: (evt, els) => {
            if (onPick && els.length) onPick(points[els[0].index]);
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) =>
                  `$${ctx.raw.x.toFixed(3)} · ${ctx.raw.y.toFixed(1)} kWh · PPD ${ctx.raw.ppd}%`,
              },
            },
          },
          scales: {
            x: axis({ title: { display: true, text: "Electricity cost (USD/day)", color: C.muted } }),
            y: axis({ title: { display: true, text: "Energy use (kWh/day)", color: C.muted } }),
          },
        },
      });
    },

    /** Bar chart of a histogram given bin centres and counts. */
    histogram(canvas, { centres, counts }) {
      return make(canvas, {
        type: "bar",
        data: {
          labels: centres.map((v) => v.toFixed(2)),
          datasets: [{
            data: counts,
            backgroundColor: centres.map((v) =>
              Math.abs(v) < 0.05 ? alpha(C.good, 0.8) : alpha(C.cyan, 0.6)
            ),
            borderRadius: 3,
            barPercentage: 1,
            categoryPercentage: 0.98,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: { title: (items) => `error ${items[0].label} kW`, label: (i) => `${i.raw} hours` },
            },
          },
          scales: {
            x: axis({ ticks: { color: C.muted, maxTicksLimit: 8 }, grid: { display: false } }),
            y: axis({ beginAtZero: true, title: { display: true, text: "Hours", color: C.muted } }),
          },
        },
      });
    },
  };
})();
