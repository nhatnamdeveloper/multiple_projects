# Svelte + Tailwind Photo Gallery

Một gallery ảnh tối giản, chạy bằng **Svelte 4 + Vite + TailwindCSS**. Có tìm kiếm, sắp xếp, trộn, lightbox và điều hướng bằng phím (← → Esc).

## Cấu trúc
```
svelte-tailwind-gallery/
├─ public/
│  └─ assets/
│     └─ images/
│        ├─ gallery/   # ảnh của bạn
│        └─ icons/     # favicon, logo, v.v.
├─ src/
│  ├─ data/images.js   # danh sách ảnh
│  ├─ lib/Gallery.svelte
│  ├─ App.svelte
│  ├─ app.css
│  └─ main.js
├─ index.html
├─ tailwind.config.js
├─ postcss.config.js
├─ svelte.config.js
└─ vite.config.js
```

## Chạy dev
```bash
npm i
npm run dev
```
Mở http://localhost:5173

## Build & deploy (tĩnh)
```bash
npm run build
```
Nội dung sẽ có trong `dist/`. Bạn có thể deploy lên **GitHub Pages** (Pages → Build from branch → trỏ vào nhánh chứa thư mục `dist`), hoặc dùng các nền tảng static hosting khác.

### GitHub Pages (cách nhanh)
- Build: `npm run build`
- Commit & push thư mục `dist` lên nhánh `gh-pages` (hoặc `docs/` trên `main`).
- Settings → Pages → Source: chọn `gh-pages` / `/ (root)`.

## Thêm ảnh của bạn
- Đặt ảnh vào `public/assets/images/gallery/` (hoặc host ở URL khác).
- Sửa `src/data/images.js` để thêm mục:
```js
{ src: "/assets/images/gallery/myphoto.jpg", w: 2048, h: 1365, alt: "Mô tả", caption: "Chú thích" }
```

> **Gợi ý bảo mật:** Xóa EXIF (GPS) trước khi public.
