// 매우 단순한 PWA 서비스 워커 — 오프라인 캐시
const CACHE = 'onpose-v6-v1';
const ASSETS = [
  './index.html',
  './manifest.json',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});
