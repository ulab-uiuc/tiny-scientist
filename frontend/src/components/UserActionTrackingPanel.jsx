import React, { useState, useEffect } from 'react';
import userActionTracker from '../utils/userActionTracker';
import plotStatusTracker from '../utils/plotStatusTracker';

const UserActionTrackingPanel = ({ hideEvaluationView, setHideEvaluationView, currentNodes = [] }) => {
  const [isTracking, setIsTracking] = useState(true);
  const [showPanel, setShowPanel] = useState(false);
  const [actionCount, setActionCount] = useState(0);
  const [plotUpdateCount, setPlotUpdateCount] = useState(0);

  // Update action and plot counts
  useEffect(() => {
    const updateCounts = () => {
      setActionCount(userActionTracker.getActions().length);
      setPlotUpdateCount(plotStatusTracker.getPlotStates().length);
    };

    // Update counts every second
    const interval = setInterval(updateCounts, 1000);
    updateCounts(); // Initial count

    return () => clearInterval(interval);
  }, []);

  const handleToggleTracking = () => {
    if (isTracking) {
      userActionTracker.stopTracking();
      plotStatusTracker.stopTracking();
      // Clear data when turning off (without recording the clear action)
      userActionTracker.clearActions();
      plotStatusTracker.clearPlotStates();
      setActionCount(0);
      setPlotUpdateCount(0);
    } else {
      userActionTracker.startTracking();
      plotStatusTracker.startTracking();
      // Record current plot state when tracking is enabled
      if (currentNodes && currentNodes.length > 0) {
        plotStatusTracker.trackNodesUpdate(currentNodes, 'tracking_enabled');
      }
    }
    setIsTracking(!isTracking);
  };

  const handleDownloadChoice = () => {
    setShowPanel(!showPanel);
  };

  const handleDownloadActions = () => {
    userActionTracker.downloadJSON();
    setShowPanel(false); // Close panel after download
  };

  const handleDownloadPlot = () => {
    plotStatusTracker.downloadJSON();
    setShowPanel(false); // Close panel after download
  };

  return (
    <div
      data-uatrack-suppress="true"
      data-uatrack-suppress-hover="true"
      data-uatrack-suppress-click="true"
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 9999,
      }}
    >
      {/* Main toggle controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
      >
        {/* Hide Plot View Toggle */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}
        >
          <span style={{
            fontSize: '0.75rem',
            color: '#9CA3AF',
            fontWeight: 'bold'
          }}>
            con 0
          </span>

          {/* Toggle Button */}
          <button
            onClick={() => setHideEvaluationView(!hideEvaluationView)}
            style={{
              position: 'relative',
              width: '32px',
              height: '16px',
              backgroundColor: !hideEvaluationView ? '#4C84FF' : '#D1D5DB',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              transition: 'background-color 0.2s ease',
              outline: 'none',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: '1px',
                left: !hideEvaluationView ? '17px' : '1px',
                width: '14px',
                height: '14px',
                backgroundColor: 'white',
                borderRadius: '50%',
                transition: 'left 0.2s ease',
                boxShadow: '0 1px 2px rgba(0, 0, 0, 0.2)'
              }}
            />
          </button>

          <span style={{
            fontSize: '0.75rem',
            color: '#9CA3AF',
            fontWeight: 'bold'
          }}>
            con 1
          </span>
        </div>

        {/* Download Button */}
        <button
          onClick={handleDownloadChoice}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '6px 8px',
            backgroundColor: '#4C84FF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            fontSize: '0.75rem',
            fontWeight: '500',
            cursor: 'pointer',
            transition: 'background-color 0.2s ease',
            outline: 'none',
            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <polyline points="7,10 12,15 17,10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Logs
        </button>
      </div>

      {/* Expandable Panel */}
      {showPanel && (
        <div
          style={{
            position: 'absolute',
            top: '35px',
            right: '0px',
            backgroundColor: 'white',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
            padding: '16px',
            minWidth: '220px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            zIndex: 10000
          }}
        >
          <div style={{ marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <div
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: isTracking ? '#10B981' : '#EF4444'
                }}
              />
              <span style={{ fontSize: '0.875rem', color: '#374151' }}>
                Status: {isTracking ? 'Recording' : 'Stopped'}
              </span>
            </div>
            <div style={{ fontSize: '0.875rem', color: '#6B7280' }}>
              Actions recorded: {actionCount}
            </div>
            <div style={{ fontSize: '0.875rem', color: '#6B7280' }}>
              Plot updates: {plotUpdateCount}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={handleDownloadActions}
              disabled={actionCount === 0}
              style={{
                padding: '8px 12px',
                backgroundColor: actionCount > 0 ? '#4C84FF' : '#D1D5DB',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '0.875rem',
                cursor: actionCount > 0 ? 'pointer' : 'not-allowed',
                fontWeight: '500'
              }}
            >
              Download User Actions
            </button>
            <button
              onClick={handleDownloadPlot}
              disabled={plotUpdateCount === 0}
              style={{
                padding: '8px 12px',
                backgroundColor: plotUpdateCount > 0 ? '#10B981' : '#D1D5DB',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '0.875rem',
                cursor: plotUpdateCount > 0 ? 'pointer' : 'not-allowed',
                fontWeight: '500'
              }}
            >
              Download Diagram
            </button>
          </div>

          <div style={{ marginTop: '12px', fontSize: '0.75rem', color: '#9CA3AF' }}>
            Session ID: {userActionTracker.sessionId?.substring(0, 8)}...
          </div>
        </div>
      )}
    </div>
  );
};

export default UserActionTrackingPanel;
