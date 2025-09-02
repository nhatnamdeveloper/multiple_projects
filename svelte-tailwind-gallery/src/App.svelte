<script>
  import Gallery from "./lib/Gallery.svelte";
  import images from "./data/images.js";
</script>

<div class="max-w-6xl mx-auto p-4 sm:p-6">
  <header class="flex items-center justify-between">
    <div class="flex items-center gap-3">
      <img src="/assets/images/icons/logo.svg" alt="Logo" class="h-8 w-8 rounded-xl ring-1 ring-white/10" />
      <h1 class="text-xl sm:text-2xl font-semibold">Album của tôi</h1>
    </div>
    <a class="text-sm text-zinc-400 hover:text-zinc-100" href="https://github.com/" target="_blank" rel="noreferrer">GitHub Pages</a>
  </header>
</div>

<Gallery {images} />
