/* ── dashboard.js — global JS loaded on every authenticated page ── */

/**
 * Polls /api/live/ every 15 seconds and dispatches a 'liveDataUpdate'
 * event so individual page templates can react to new data.
 */
(function startPolling() {
  async function poll() {
    try {
      const resp = await fetch('/api/live/');
      if (!resp.ok) return;
      const data = await resp.json();
      document.dispatchEvent(new CustomEvent('liveDataUpdate', { detail: data }));
      updateLastUpdated();
    } catch (e) {
      console.warn('Live poll failed:', e);
    }
  }

  poll();                          // immediate first poll
  setInterval(poll, 15_000);      // then every 15 s
})();

function updateLastUpdated() {
  const el = document.getElementById('lastUpdated');
  if (el) el.textContent = 'Updated ' + new Date().toLocaleTimeString();
}

/**
 * Congestion index → Bootstrap colour name
 */
function indexToColor(index) {
  if (index <= 25) return 'success';
  if (index <= 50) return 'warning';
  if (index <= 75) return 'orange';
  return 'danger';
}

/**
 * Congestion index → CSS hex colour
 */
function indexToHex(index) {
  if (index <= 25) return '#22c55e';
  if (index <= 50) return '#f59e0b';
  if (index <= 75) return '#f97316';
  return '#ef4444';
}

/**
 * Standard Chart.js dark-theme defaults.
 * Usage: new Chart(ctx, makeChartConfig('line', labels, datasets))
 */
function makeChartConfig(type, labels, datasets, extraOptions = {}) {
  return {
    type,
    data: { labels, datasets },
    options: Object.assign({
      responsive: true,
      animation:  false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#64748b', maxTicksLimit: 10 },
          grid:  { color: '#1e2435' },
        },
        y: {
          ticks: { color: '#64748b' },
          grid:  { color: '#1e2435' },
        },
      },
    }, extraOptions),
  };
}
