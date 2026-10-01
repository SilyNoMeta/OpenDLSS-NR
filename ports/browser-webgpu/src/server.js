// The static server behind `serve.mjs` and `tools/headless.mjs`.
//
// There is no build step anywhere in this port. The sources are ES modules and WGSL text, served as they are,
// so what runs in the browser is exactly what is in the repository. The only thing a plain `file://` open
// cannot do is fetch the shaders, the fixtures and the weights, which is the whole reason this exists.

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = resolve(fileURLToPath(new URL('..', import.meta.url)));

const types = new Map(Object.entries({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.glb': 'model/gltf-binary',
  '.ktx2': 'image/ktx2',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
}));

/**
 * Serve the port directory. `onReport` receives whatever a page posts to /report, which is how the self-test
 * and the parity page hand a verdict back to a headless run - a browser has no other channel to the process
 * that launched it.
 */
/**
 * Directories outside the port that the pages need: the weights and a parity fixture. Neither belongs in the
 * repository - one is 141 MiB of model, the other 75 MiB of recorded tensors - so both are mounted from
 * wherever they happen to live, named by NR_WEIGHTS and NR_FIXTURES or by the defaults below.
 */
function mounts() {
  const repository = resolve(root, '..', '..');
  return [
    { prefix: '/weights/', directory: resolve(process.env.NR_WEIGHTS ?? join(repository, 'models', 'nr')) },
    { prefix: '/fixtures/', directory: process.env.NR_FIXTURES ? resolve(process.env.NR_FIXTURES) : null },
    { prefix: '/scenes/', directory: process.env.NR_SCENES ? resolve(process.env.NR_SCENES) : null },
    { prefix: '/three/', directory: join(root, 'node_modules', 'three') },
  ].filter((mount) => mount.directory);
}

export function startServer({ port = 8099, host = '127.0.0.1', onReport, onUpload } = {}) {
  const extra = mounts();
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, `http://${host}`);

      if (request.method === 'POST' && url.pathname === '/report') {
        const chunks = [];
        for await (const chunk of request) chunks.push(chunk);
        const body = Buffer.concat(chunks).toString();
        response.writeHead(204).end();
        (onReport ?? console.log)(body);
        return;
      }

      // tools/capture.mjs: the capture page posts each tensor it produced, binary, to /upload/<relative path>.
      // Only a host that asked for uploads accepts them, and the path stays relative (onUpload checks it).
      if (request.method === 'POST' && url.pathname.startsWith('/upload/') && onUpload) {
        const chunks = [];
        for await (const chunk of request) chunks.push(chunk);
        await onUpload(decodeURIComponent(url.pathname.slice('/upload/'.length)), Buffer.concat(chunks));
        response.writeHead(204).end();
        return;
      }

      let path = decodeURIComponent(url.pathname);
      if (path.endsWith('/')) path += 'index.html';
      // normalize collapses any ..; the resolved path still has to stay under whichever root serves it.
      let base = root;
      let relative = normalize(path);
      for (const mount of extra) {
        if (!path.startsWith(mount.prefix)) continue;
        base = mount.directory;
        relative = normalize(path.slice(mount.prefix.length));
        break;
      }
      const target = join(base, relative);
      if (target !== base && !target.startsWith(base + sep)) {
        response.writeHead(403).end('outside the served directory');
        return;
      }
      const info = await stat(target);
      const body = await readFile(target);
      response.writeHead(200, {
        'content-type': types.get(extname(target)) ?? 'application/octet-stream',
        'content-length': info.size,
        // Weights and fixtures are large and fixed for a session; everything else changes as it is edited,
        // and a stale module is a confusing way to fail.
        'cache-control': extname(target) === '.bin' ? 'public, max-age=3600' : 'no-store',
      });
      response.end(body);
    } catch (error) {
      const missing = error.code === 'ENOENT' || error.code === 'ENOTDIR';
      response.writeHead(missing ? 404 : 500).end(missing ? 'not found' : String(error));
    }
  });
  return new Promise((done) => server.listen(port, host, () => done(server)));
}
