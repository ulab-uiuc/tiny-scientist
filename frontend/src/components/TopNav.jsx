import React from 'react';

/**
 * 顶部导航栏：切换 Exploration / Evaluation 视图
 * @param {boolean} showTree
 * @param {Function} setShowTree
 */
const TopNav = ({ currentView, setCurrentView, showCodeView = false }) => {
  /* ---------- SVG 图标 ---------- */
  const overviewIcon = (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ marginRight: '8px' }}
    >
      <path
        d="m2.25 12 8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );

  const explorationIcon = (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ marginRight: '8px' }}
    >
      <g clipPath="url(#clip0)">
        <path
          d="M8.525 15.831C12.242 15.831 15.255 12.818 15.255 9.101C15.255 5.385 12.242 2.372 8.525 2.372C4.809 2.372 1.796 5.385 1.796 9.101C1.796 12.818 4.809 15.831 8.525 15.831Z"
          stroke="#A2A2A2"
          strokeWidth="2"
          strokeLinejoin="round"
        />
        <path
          d="M10.765 6.467C10.192 5.894 9.400 5.539 8.525 5.539C7.651 5.539 6.859 5.894 6.286 6.467"
          stroke="#A2A2A2"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M13.363 13.939L16.722 17.298"
          stroke="#A2A2A2"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
      <defs>
        <clipPath id="clip0">
          <rect width="19" height="19" fill="white" transform="translate(0.213 0.789)" />
        </clipPath>
      </defs>
    </svg>
  );

  const evaluationIcon = (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ marginRight: '8px' }}
    >
      <path
        d="M2.479 2.465V16.715H16.729"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="8.021" cy="8.007" r="1.585" fill="currentColor" />
      <circle cx="14.750" cy="4.444" r="1.979" fill="currentColor" />
      <circle cx="6.042" cy="13.153" r="1.188" fill="currentColor" />
      <circle cx="13.167" cy="11.570" r="1.188" fill="currentColor" />
    </svg>
  );

  const codeIcon = (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ marginRight: '8px' }}
    >
      <path
        d="M7 8L3 12L7 16M17 8L21 12L17 16M14 4L10 20"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );

  const paperIcon = (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ marginRight: '8px' }}
    >
      <path
        d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 2V8H20"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16 13H8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16 17H8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10 9H9H8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );

  const githubIcon = (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"
      />
    </svg>
  );

  /* ---------- 样式 ---------- */
  const tabStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: '10px 16px',
    cursor: 'pointer',
    borderRadius: '9999px',
    transition: 'all 0.2s ease',
    fontSize: '0.875rem',
    fontWeight: 500,
  };

  const getActiveStyle = (viewName) =>
    currentView === viewName
      ? { backgroundColor: '#fff', color: '#141414' }
      : { backgroundColor: 'transparent', color: '#A0AEC0' };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#0F172A',
        padding: '10px 20px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center' }}>
      {/* Home View */}
      <div
        style={{ ...tabStyle, ...getActiveStyle('home_view'), marginRight: 6 }}
        onClick={() => setCurrentView('home_view')}
      >
        {overviewIcon}
        Home View
      </div>
      {/* Exploration View */}
      <div
        style={{ ...tabStyle, ...getActiveStyle('exploration'), marginRight: 6 }}
        onClick={() => setCurrentView('exploration')}
      >
        {explorationIcon}
        Exploration View
      </div>

      {/* Evaluation View */}
      <div
        style={{ ...tabStyle, ...getActiveStyle('evaluation'), marginRight: 6 }}
        onClick={() => setCurrentView('evaluation')}
      >
        {evaluationIcon}
        Evaluation View
      </div>

      {/* Code View - only show if code has been generated */}
      {showCodeView && (
        <div
          style={{ ...tabStyle, ...getActiveStyle('code_view'), marginRight: 6 }}
          onClick={() => setCurrentView('code_view')}
        >
          {codeIcon}
          Code View
        </div>
      )}

      {/* Paper View */}
      <div
        style={{ ...tabStyle, ...getActiveStyle('paper_view') }}
        onClick={() => setCurrentView('paper_view')}
      >
        {paperIcon}
        Paper View
      </div>
      </div>

      {/* GitHub Logo */}
      <a
        href="https://github.com/ulab-uiuc/tiny-scientist"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: 'flex',
          alignItems: 'center',
          color: '#A0AEC0',
          textDecoration: 'none',
          transition: 'color 0.2s ease',
        }}
        onMouseEnter={(e) => e.target.style.color = '#fff'}
        onMouseLeave={(e) => e.target.style.color = '#A0AEC0'}
      >
        {githubIcon}
      </a>
    </div>
  );
};

export default TopNav;
