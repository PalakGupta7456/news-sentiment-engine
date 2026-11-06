/* The Mood Desk: reads data/dashboard.json, rebuilt hourly by the pipeline. */
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const sign = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);

  function toneColor(t) {           // −1..+1 -> neg..neu..pos
    const hex = (h) => h.match(/\w\w/g).map((x) => parseInt(x, 16));
    const [a, b] = t < 0 ? [hex(css("--neu")), hex(css("--neg"))] : [hex(css("--neu")), hex(css("--pos"))];
    const k = Math.min(1, Math.abs(t));
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;
  }
  const toneWord = (t) => (t <= -0.35 ? "grim" : t <= -0.12 ? "mostly negative" : t < 0.12 ? "mixed" : t < 0.35 ? "mostly positive" : "upbeat");

  fetch("data/dashboard.json", { cache: "no-store" })
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(render)
    .catch(() => { $("meta").textContent = "The first hourly run hasn't published yet."; });

  function render(d) {
    const mins = Math.max(0, Math.round((Date.now() - new Date(d.generated_at)) / 60000));
    $("meta").textContent = `Updated ${mins < 60 ? mins + " min" : Math.round(mins / 60) + " h"} ago from ${d.totals.sources} outlets, every hour`;
    if (d.models) { $("m-sent").textContent = d.models.sentiment; $("m-enc").textContent = d.models.encoder; }

    const top = d.stories[0];
    const mix = d.overall.mix;
    $("headline").textContent = `The news is ${toneWord(d.overall.tone)} today${top ? `, and "${top.name.split(",")[0]}" leads it` : ""}.`;
    $("subline").textContent = `${d.totals.articles_window.toLocaleString()} headlines in the last ${d.window_hours} hours. ` +
      `${Math.round((mix.negative || 0) * 100)}% read as negative and ${Math.round((mix.positive || 0) * 100)}% as positive. ` +
      `${d.quality.n_topics} distinct stories found.`;
    const parts = [["negative", "--neg", "Negative"], ["neutral", "--neu", "Neutral"], ["positive", "--pos", "Positive"]];
    $("mix").innerHTML = parts.map(([k, c]) => `<span style="width:${(mix[k] || 0) * 100}%;background:${css(c)}"></span>`).join("");
    $("mix-legend").innerHTML = parts.map(([k, c, l]) => `<span><i style="background:${css(c)}"></i>${l} <b>${Math.round((mix[k] || 0) * 100)}%</b></span>`).join("");

    $("stories").innerHTML = d.stories.slice(0, 9).map((s) => `
      <article class="story">
        <h3>${esc(s.name)}</h3>
        <p class="facts"><span>${s.n} headlines from ${s.sources} ${s.sources === 1 ? "outlet" : "outlets"}</span>
          <span class="tone-chip" style="color:${toneColor(s.tone)}">tone ${sign(s.tone)}</span></p>
        <ul>${s.headlines.slice(0, 3).map((h) => `<li style="--c:${toneColor(h.tone)}"><a href="${esc(h.u)}" target="_blank" rel="noopener">${esc(h.t)}</a> <small>${esc(h.s)}</small></li>`).join("")}</ul>
      </article>`).join("");

    const q = d.quality;
    $("quality").textContent = `Clustering quality: silhouette ${q.silhouette ?? "n/a"} (cosine), with ${Math.round(q.coverage * 100)}% of headlines assigned to a story.`;
    storyMap(d.points);

    const max = Math.max(0.3, ...d.sources.map((s) => Math.abs(s.tone)));
    $("outlets").innerHTML = d.sources.map((s) => {
      const w = (Math.abs(s.tone) / max) * 50;
      const left = s.tone < 0 ? 50 - w : 50;
      return `<li><span>${esc(s.name)}</span><span class="track"><span style="left:${left}%;width:${w}%;background:${toneColor(s.tone / max)}"></span></span><span class="v">${sign(s.tone)}</span></li>`;
    }).join("");

    const muted = css("--muted");
    new Chart($("daily"), {
      type: "bar",
      data: { labels: d.daily.date.map((x) => x.slice(5)), datasets: [{ data: d.daily.tone, backgroundColor: d.daily.tone.map((t) => toneColor(t * 2)), borderRadius: 2 }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (i) => `tone ${sign(i.raw)}, ${d.daily.n[i.dataIndex]} articles` } } },
        scales: { x: { ticks: { color: muted, maxTicksLimit: 10 }, grid: { display: false } }, y: { ticks: { color: muted }, grid: { color: css("--rule") } } } },
    });
  }

  function storyMap(points) {
    const muted = css("--muted");
    new Chart($("map"), {
      type: "scatter",
      data: { datasets: [{ data: points.map((p) => ({ x: p.x, y: p.y, p })), pointRadius: points.map((p) => (p.k >= 0 && p.k < 9 ? 5 : 3)),
        pointBackgroundColor: points.map((p) => toneColor(p.tone)), pointBorderColor: points.map((p) => (p.k >= 0 && p.k < 9 ? css("--ink") : "transparent")), pointBorderWidth: 0.6 }] },
      options: { maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (i) => `${i.raw.p.t} (${i.raw.p.s})` } } },
        scales: { x: { display: false }, y: { display: false } } },
    });
    void muted;
  }
})();
