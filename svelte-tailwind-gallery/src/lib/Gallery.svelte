<script>
  export let images = [];
  import { onMount } from "svelte";

  let q = "";
  let asc = true;
  let list = [];
  let open = false;
  let i = 0;

  function aspectClass({w, h}) {
    if (!w || !h) return "aspect-[4/3]";
    const r = w / h;
    if (r > 1.35 && r < 1.45) return "aspect-[4/3]";
    if (r > 1.7) return "aspect-[16/9]";
    if (r < 0.85) return "aspect-[3/4]";
    return "aspect-[1/1]";
  }

  function filter() {
    const qq = q.trim().toLowerCase();
    list = images.filter(x => (x.caption||'').toLowerCase().includes(qq) || (x.alt||'').toLowerCase().includes(qq));
  }

  function sortAZ() {
    asc = !asc;
    list = [...list].sort((a,b)=> (a.caption||'').localeCompare(b.caption||''));
    if (!asc) list.reverse();
  }

  function shuffle() {
    const arr = [...list];
    for (let k=arr.length-1; k>0; k--) {
      const j = Math.floor(Math.random()*(k+1));
      [arr[k], arr[j]] = [arr[j], arr[k]];
    }
    list = arr;
  }

  function openAt(idx) {
    i = idx; open = true; updateLightbox();
    document.body.style.overflow = "hidden";
  }
  function closeLb() {
    open = false;
    document.body.style.overflow = "";
  }
  function next() { i = (i + 1) % list.length; updateLightbox(); }
  function prev() { i = (i - 1 + list.length) % list.length; updateLightbox(); }

  let current = { src: "", alt: "", caption: "" };
  function updateLightbox() {
    current = list[i] || { src: "", alt: "", caption: "" };
  }

  onMount(()=>{
    list = [...images];
  });

  function onKey(e) {
    if (!open) return;
    if (e.key === "Escape") closeLb();
    if (e.key === "ArrowRight") next();
    if (e.key === "ArrowLeft") prev();
  }
</script>

<!-- Toolbar -->
<div class="max-w-6xl mx-auto px-4 sm:px-6">
  <div class="flex items-center justify-between gap-2">
    <div class="flex gap-2">
      <button on:click={sortAZ} class="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 ring-1 ring-white/10">{asc ? "Sắp xếp A→Z" : "Sắp xếp Z→A"}</button>
      <button on:click={shuffle} class="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 ring-1 ring-white/10">Trộn</button>
    </div>
    <label class="relative">
      <input bind:value={q} on:input={filter} placeholder="Tìm chú thích..." class="peer w-44 sm:w-56 bg-white/5 ring-1 ring-white/10 focus:ring-white/25 outline-none rounded-xl px-3 py-1.5 placeholder:text-zinc-500" />
      <span class="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500">⌘K</span>
    </label>
  </div>
</div>

<!-- Grid -->
<main class="max-w-6xl mx-auto p-4 sm:p-6 select-none">
  <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 sm:gap-3">
    {#each list as img, idx}
      <button class={"relative group " + aspectClass(img) + " overflow-hidden rounded-2xl ring-1 ring-white/10 bg-zinc-900"}
              aria-label={img.alt || `Ảnh ${idx+1}`} on:click={() => openAt(idx)}>
        <img src={img.src} alt={img.alt || ""} loading="lazy" class="h-full w-full object-cover transition duration-500 ease-out group-hover:scale-105" />
        <div class="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/10 to-transparent opacity-0 group-hover:opacity-100 transition"></div>
        <div class="pointer-events-none absolute bottom-0 left-0 right-0 p-2 text-left text-xs text-white/90 opacity-0 group-hover:opacity-100 transition">{img.caption || ""}</div>
      </button>
    {/each}
  </div>
</main>

<!-- Lightbox -->
{#if open}
  <div class="fixed inset-0 flex items-center justify-center bg-black/80 z-50" on:click={(e)=> e.target===e.currentTarget && closeLb()} on:keydown={onKey} tabindex="0" autofocus>
    <button on:click={closeLb} class="absolute top-4 right-4 sm:top-6 sm:right-6 p-2 rounded-2xl bg-white/10 hover:bg-white/20 ring-1 ring-white/15" aria-label="Đóng">✕</button>
    <button on:click|stopPropagation={prev} class="absolute left-2 sm:left-6 p-2 rounded-2xl bg-white/10 hover:bg-white/20 ring-1 ring-white/15" aria-label="Trước">‹</button>
    <figure class="max-w-[94vw] max-h-[86vh] flex flex-col items-center gap-3">
      <img src={current.src} alt={current.alt || ""} class="max-h-[75vh] w-auto object-contain rounded-2xl shadow-[0_10px_30px_rgba(0,0,0,0.5)]" />
      <figcaption class="text-sm text-zinc-300">{current.caption || ""}</figcaption>
    </figure>
    <button on:click|stopPropagation={next} class="absolute right-2 sm:right-6 p-2 rounded-2xl bg-white/10 hover:bg-white/20 ring-1 ring-white/15" aria-label="Sau">›</button>
  </div>
{/if}
