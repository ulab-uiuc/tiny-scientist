import React, { useState, useEffect, useMemo, useRef } from 'react';
import { io } from 'socket.io-client';

const resolveSocketBaseURL = () => {
  if (typeof window === 'undefined') {
    return 'http://localhost:5000';
  }

  const explicitBase = process.env.REACT_APP_SOCKET_BASE_URL;
  if (explicitBase) {
    return explicitBase;
  }

  const { protocol, hostname, port } = window.location;

  // When running the dev server (port 3000) we need to talk to the backend port.
  if (port === '3000') {
    const backendPort = process.env.REACT_APP_SOCKET_PORT || '5000';
    return `${protocol}//${hostname}:${backendPort}`;
  }

  return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
};

const SOCKET_PATH = process.env.REACT_APP_SOCKET_PATH || '/socket.io';

const LogDisplay = ({ isVisible, onToggle }) => {
  const [logs, setLogs] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);
  const logsEndRef = useRef(null);
  const socketUrl = useMemo(resolveSocketBaseURL, []);

  useEffect(() => {
    socketRef.current = io(socketUrl, {
      path: SOCKET_PATH,
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 2000,
      reconnectionDelayMax: 6000,
      timeout: 10000,
    });

    socketRef.current.on('connect', () => {
      setIsConnected(true);
    });

    socketRef.current.on('disconnect', () => {
      setIsConnected(false);
    });

    socketRef.current.on('log', (data) => {
      const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
      setLogs(prev => [...prev, { ...data, timestamp }]);
    });

    return () => {
      socketRef.current?.disconnect();
      socketRef.current = null;
    };
  }, [socketUrl]);

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        backgroundColor: '#ffffff',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        margin: '16px 0 0 0',
        overflow: 'hidden',
        transition: 'all 0.3s ease-in-out',
        maxHeight: isVisible ? '200px' : '48px',
        width: '800px', // Match the exact SVG plot width
        boxSizing: 'border-box'
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 16px',
          backgroundColor: '#f8fafc',
          borderBottom: isVisible ? '1px solid #e5e7eb' : 'none',
          cursor: 'pointer',
        }}
        onClick={onToggle}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '13px',
            fontWeight: '600',
            color: '#374151'
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="4,17 10,11 4,5"></polyline>
              <line x1="12" y1="19" x2="20" y2="19"></line>
            </svg>
            Backend Logs
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: isConnected ? '#10b981' : '#ef4444',
            }}></div>
            <span style={{
              fontSize: '12px',
              color: '#6b7280',
              fontWeight: '500'
            }}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {logs.length > 0 && (
            <div style={{
              backgroundColor: '#e5e7eb',
              color: '#374151',
              fontSize: '11px',
              fontWeight: '600',
              padding: '2px 6px',
              borderRadius: '10px',
              minWidth: '20px',
              textAlign: 'center'
            }}>
              {logs.length}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isVisible && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearLogs();
              }}
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                fontWeight: '500',
                backgroundColor: '#f3f4f6',
                color: '#6b7280',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = '#e5e7eb';
                e.target.style.color = '#374151';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = '#f3f4f6';
                e.target.style.color = '#6b7280';
              }}
            >
              Clear
            </button>
          )}

          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#6b7280"
            strokeWidth="2"
            style={{
              transform: isVisible ? 'rotate(90deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s ease'
            }}
          >
            <polyline points="9,18 15,12 9,6"></polyline>
          </svg>
        </div>
      </div>

      {/* Log Content */}
      {isVisible && (
        <div style={{
          height: '140px',
          overflow: 'auto',
          padding: '12px',
          backgroundColor: '#1f2937',
          fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
          fontSize: '12px',
          lineHeight: '1.5',
        }}>
          {logs.length === 0 ? (
            <div style={{
              color: '#9ca3af',
              textAlign: 'center',
              padding: '20px 0',
              fontStyle: 'italic'
            }}>
              📡 Waiting for backend messages...
            </div>
          ) : (
            logs.map((log, index) => (
              <div key={index} style={{
                marginBottom: '4px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
                padding: '2px 0',
                borderRadius: '3px'
              }}>
                <span style={{
                  color: '#9ca3af',
                  fontSize: '11px',
                  flexShrink: 0,
                  fontWeight: '400'
                }}>
                  [{log.timestamp}]
                </span>
                <span style={{
                  color: log.level === 'error' ? '#fca5a5' :
                        log.level === 'warning' ? '#fde047' :
                        '#86efac',
                  wordBreak: 'break-word',
                  flex: 1
                }}>
                  {log.message}
                </span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      )}
    </div>
  );
};

export default LogDisplay;
