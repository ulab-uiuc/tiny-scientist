import React from 'react';

/**
 * 单个因子（Novelty / Feasibility / Impact）分数与理由
 */
const FactorBlock = ({
  label,
  color,
  score,
  reason,
  beforeVal,
  afterVal,
  showAfter,
}) => {
  const hasBefore = beforeVal !== null && beforeVal !== undefined;
  const round = (v) => Math.round(parseFloat(v) || 0);

  /* 箭头 */
  const Arrow = () => {
    if (!hasBefore) return null;
    const b = round(beforeVal);
    const a = round(afterVal);
    if (a > b) return <span style={{ margin: '0 6px' }}>▲</span>;
    if (a < b) return <span style={{ margin: '0 6px' }}>▼</span>;
    return <span style={{ margin: '0 6px' }}>─</span>;
  };

  return (
    <div
      style={{
        backgroundColor: '#F9FAFB',
        border: '1px solid #E5E7EB',
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
        <span
          style={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            backgroundColor: color,
            marginRight: 8,
          }}
        />
        <span
          style={{ fontSize: '1rem', fontWeight: 600, paddingRight: '1em', color }}
        >
          {label}
        </span>
        {hasBefore ? (
          <>
            {round(beforeVal)}
            <Arrow />
            {round(afterVal)}
          </>
        ) : (
          <span>Score: {round(score)}</span>
        )}
      </div>

      <div
        style={{
          fontSize: '0.875rem',
          color: '#111827',
          lineHeight: 1.4,
          whiteSpace: 'pre-wrap',
        }}
      >
        {reason || '(No reason provided)'}
      </div>
    </div>
  );
};

export default FactorBlock;
