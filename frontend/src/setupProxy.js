const { createProxyMiddleware } = require('http-proxy-middleware');

const LIVE_TARGET = process.env.LIVE_API_TARGET || 'http://localhost:5000';
const DEMO_TARGET = process.env.DEMO_API_TARGET || 'http://localhost:5001';

const resolveTarget = (req) => {
  if (req.headers['x-demo-mode'] === 'true') {
    return DEMO_TARGET;
  }
  if ((req.url || '').startsWith('/demo')) {
    return DEMO_TARGET;
  }
  const referer = req.headers.referer || '';
  if (referer.includes('/demo')) {
    return DEMO_TARGET;
  }
  return LIVE_TARGET;
};

module.exports = function setupProxy(app) {
  app.use(
    ['/api', '/socket.io'],
    createProxyMiddleware({
      target: LIVE_TARGET,
      changeOrigin: true,
      ws: true,
      router: resolveTarget,
      onProxyReq(proxyReq, req) {
        if (req.headers['x-demo-mode'] === 'true') {
          proxyReq.setHeader('x-demo-mode', 'true');
        }
        if ((req.url || '').startsWith('/demo')) {
          proxyReq.setHeader('x-demo-mode', 'true');
        }
      },
    })
  );
};
