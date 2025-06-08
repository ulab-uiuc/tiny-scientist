import React from 'react';

/**
 * 顶部导航栏：切换 Exploration / Evaluation 视图
 * @param {boolean} showTree
 * @param {Function} setShowTree
 */
const TopNav = ({ currentView, setCurrentView }) => {
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
        backgroundColor: '#0F172A',
        padding: '10px 20px',
      }}
    >
      {/* Overview */}
      <div
        style={{ ...tabStyle, ...getActiveStyle('overview'), marginRight: 6 }}
        onClick={() => setCurrentView('overview')}
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
        style={{ ...tabStyle, ...getActiveStyle('evaluation') }}
        onClick={() => setCurrentView('evaluation')}
      >
        {evaluationIcon}
        Evaluation View
      </div>
    </div>
  );
};

export default TopNav;
