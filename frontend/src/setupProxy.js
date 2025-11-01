const { createProxyMiddleware } = require('http-proxy-middleware');

const LIVE_TARGET = process.env.LIVE_API_TARGET || 'http://localhost:5000';
const DEMO_TARGET = process.env.DEMO_API_TARGET || 'http://localhost:5001';

module.exports = function setupProxy(app) {
  app.use(
    ['/demo/api', '/demo/socket.io'],
    createProxyMiddleware({
      target: DEMO_TARGET,
      changeOrigin: true,
      ws: true,
      pathRewrite: {
        '^/demo/api': '/api',
        '^/demo/socket.io': '/socket.io',
      },
    })
  );

  app.use(
    ['/api', '/socket.io'],
    createProxyMiddleware({
      target: LIVE_TARGET,
      changeOrigin: true,
      ws: true,
    })
  );
};
