/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');

const express = require('express');
const cors = require('cors');
const http = require('http');
const { Server } = require('socket.io');

function loadJson(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(raw);
}

function popOrLast(queue) {
  if (!Array.isArray(queue) || queue.length === 0) return null;
  if (queue.length === 1) return queue[0];
  return queue.shift();
}

function safeJoin(rootDir, relativePath) {
  const full = path.resolve(rootDir, relativePath);
  if (!full.startsWith(path.resolve(rootDir))) {
    throw new Error('Path traversal blocked');
  }
  return full;
}

function createApp({ cachePath, port }) {
  const demoRoot = path.resolve(__dirname);
  const cacheAbsPath = path.resolve(cachePath);
  const cache = loadJson(cacheAbsPath);

  const queues = {
    generate_initial: Array.isArray(cache.queues?.generate_initial) ? [...cache.queues.generate_initial] : [],
    generate_children: Array.isArray(cache.queues?.generate_children) ? [...cache.queues.generate_children] : [],
    evaluate: Array.isArray(cache.queues?.evaluate) ? [...cache.queues.evaluate] : [],
    modify: Array.isArray(cache.queues?.modify) ? [...cache.queues.modify] : [],
    merge: Array.isArray(cache.queues?.merge) ? [...cache.queues.merge] : [],
    code: Array.isArray(cache.queues?.code) ? [...cache.queues.code] : [],
    write: Array.isArray(cache.queues?.write) ? [...cache.queues.write] : [],
    review: Array.isArray(cache.queues?.review) ? [...cache.queues.review] : [],
  };

  const app = express();
  app.use(express.json({ limit: '10mb' }));
  app.use(
    cors({
      origin: true,
      credentials: true,
    })
  );

  app.post('/api/clear-session', (_req, res) => {
    queues.generate_initial = Array.isArray(cache.queues?.generate_initial) ? [...cache.queues.generate_initial] : [];
    queues.generate_children = Array.isArray(cache.queues?.generate_children) ? [...cache.queues.generate_children] : [];
    queues.evaluate = Array.isArray(cache.queues?.evaluate) ? [...cache.queues.evaluate] : [];
    queues.modify = Array.isArray(cache.queues?.modify) ? [...cache.queues.modify] : [];
    queues.merge = Array.isArray(cache.queues?.merge) ? [...cache.queues.merge] : [];
    queues.code = Array.isArray(cache.queues?.code) ? [...cache.queues.code] : [];
    queues.write = Array.isArray(cache.queues?.write) ? [...cache.queues.write] : [];
    queues.review = Array.isArray(cache.queues?.review) ? [...cache.queues.review] : [];
    res.json({ status: 'cleared' });
  });

  app.post('/api/configure', (_req, res) => {
    res.json({
      status: 'configured',
      model: 'demo-cache',
      budget: null,
      budget_preference: null,
    });
  });

  app.get('/api/get-prompts', (_req, res) => {
    res.json({
      system_prompt: cache.prompts?.system_prompt || '',
      criteria: cache.prompts?.criteria || { novelty: '', feasibility: '', impact: '' },
      defaults: cache.prompts?.defaults || { system_prompt: '', novelty: '', feasibility: '', impact: '' },
    });
  });

  app.post('/api/set-system-prompt', (_req, res) => {
    res.json({ status: 'success', message: 'System prompt updated' });
  });

  app.post('/api/set-criteria', (_req, res) => {
    res.json({ status: 'success', message: 'Criteria updated' });
  });

  app.post('/api/suggest-dimensions', (_req, res) => {
    res.json({ dimension_pairs: cache.dimension_pairs || [] });
  });

  app.post('/api/generate-initial', (_req, res) => {
    const payload = popOrLast(queues.generate_initial);
    if (!payload) return res.status(500).json({ error: 'No cached generate-initial payload' });
    res.json(payload);
  });

  app.post('/api/generate-children', (_req, res) => {
    const payload = popOrLast(queues.generate_children);
    if (!payload) return res.status(500).json({ error: 'No cached generate-children payload' });
    res.json(payload);
  });

  app.post('/api/evaluate', (_req, res) => {
    const payload = popOrLast(queues.evaluate);
    if (!payload) return res.status(500).json({ error: 'No cached evaluate payload' });
    res.json(payload);
  });

  app.post('/api/evaluate-dimension', (req, res) => {
    const evalPayload = queues.evaluate[0] || (Array.isArray(cache.queues?.evaluate) ? cache.queues.evaluate[cache.queues.evaluate.length - 1] : null);
    const dimensionIndex = Number(req.body?.dimension_index ?? 0);
    const dimensionPair = req.body?.dimension_pair;
    if (!evalPayload || !dimensionPair) {
      return res.status(400).json({ error: 'Missing cached evaluation or dimension_pair' });
    }
    const key = `${dimensionPair.dimensionA}-${dimensionPair.dimensionB}`;
    const scores = (evalPayload.ideas || []).map((idea) => {
      const score = idea?.scores?.[key];
      const reasonKey = `Dimension${dimensionIndex + 1}Reason`;
      const reason = idea?.[reasonKey] || idea?.originalData?.[reasonKey] || '';
      return { id: idea.id, score, reason };
    });
    res.json({
      scores,
      dimension_pair: dimensionPair,
      dimension_index: dimensionIndex,
    });
  });

  app.post('/api/modify', (_req, res) => {
    const payload = popOrLast(queues.modify);
    if (!payload) return res.status(500).json({ error: 'No cached modify payload' });
    res.json(payload);
  });

  app.post('/api/merge', (_req, res) => {
    const payload = popOrLast(queues.merge);
    if (!payload) return res.status(500).json({ error: 'No cached merge payload' });
    res.json(payload);
  });

  app.post('/api/code', (_req, res) => {
    const payload = popOrLast(queues.code);
    if (!payload) return res.status(500).json({ error: 'No cached code payload' });
    res.json(payload);
  });

  app.post('/api/write', (_req, res) => {
    const payload = popOrLast(queues.write);
    if (!payload) return res.status(500).json({ error: 'No cached write payload' });
    res.json(payload);
  });

  app.post('/api/review', (_req, res) => {
    const payload = popOrLast(queues.review);
    if (!payload) return res.status(500).json({ error: 'No cached review payload' });
    res.json(payload);
  });

  app.get('/api/files/*', (req, res) => {
    const rel = req.path.replace(/^\/api\/files\//, '');
    if (!rel) return res.status(400).json({ error: 'Missing file path' });

    const txtMap = cache.files?.text || {};
    const binMap = cache.files?.binary || {};

    if (Object.prototype.hasOwnProperty.call(txtMap, rel)) {
      const localRel = txtMap[rel];
      try {
        const full = safeJoin(demoRoot, localRel);
        const content = fs.readFileSync(full, 'utf8');
        return res.json({ content });
      } catch (e) {
        return res.status(404).json({ error: 'File not found' });
      }
    }

    if (Object.prototype.hasOwnProperty.call(binMap, rel)) {
      const localRel = binMap[rel];
      try {
        const full = safeJoin(demoRoot, localRel);
        return res.sendFile(full);
      } catch (e) {
        return res.status(404).json({ error: 'File not found' });
      }
    }

    return res.status(404).json({ error: 'File not found' });
  });

  const buildDir = path.resolve(__dirname, '..', 'build');
  if (fs.existsSync(buildDir)) {
    app.use(express.static(buildDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(buildDir, 'index.html'));
    });
  }

  const server = http.createServer(app);
  const io = new Server(server, {
    cors: {
      origin: true,
      credentials: true,
    },
  });

  io.on('connection', (socket) => {
    const items = Array.isArray(cache.logs) ? cache.logs : [];
    let i = 0;
    const sendNext = () => {
      if (i >= items.length) return;
      const item = items[i];
      i += 1;
      socket.emit('log', {
        message: String(item.message || ''),
        level: String(item.level || 'info'),
        timestamp: Math.floor(Date.now() / 1000),
      });
      setTimeout(sendNext, 250);
    };
    sendNext();
  });

  server.listen(port, () => {
    console.log(`[demo_cache] server running on http://localhost:${port}`);
    console.log(`[demo_cache] cache: ${cacheAbsPath}`);
    if (fs.existsSync(buildDir)) {
      console.log('[demo_cache] serving frontend build from frontend/build');
    } else {
      console.log('[demo_cache] frontend/build not found (run npm run build)');
    }
  });

  return { app, server, io };
}

function main() {
  const port = Number(process.env.DEMO_CACHE_PORT || 3030);
  const cachePath = process.env.DEMO_CACHE_FILE || path.join(__dirname, 'cache.json');
  createApp({ cachePath, port });
}

if (require.main === module) {
  main();
}
