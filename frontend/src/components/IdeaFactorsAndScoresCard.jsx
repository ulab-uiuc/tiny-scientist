import React from 'react';
import FactorBlock from './FactorBlock';

/**
 * 组合 3 个因子块的卡片
 */
const IdeaFactorsAndScoresCard = ({ node, isEvaluating, showAfter }) => {
  const hasPrev = !!node.previousState;

  if (isEvaluating) {
    return (
      <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: 16 }}>
        Scores will be displayed once evaluation is complete.
      </p>
    );
  }

  const pick = (field) =>
    showAfter || !hasPrev ? node[field] : node.previousState[field];

  return (
    <>
      <FactorBlock
        label="Novelty"
        color="#618685"
        score={pick('noveltyScore')}
        reason={pick('noveltyReason')}
        beforeVal={hasPrev ? node.previousState.noveltyScore : null}
        afterVal={node.noveltyScore}
        showAfter={showAfter}
      />

      <FactorBlock
        label="Feasibility"
        color="#50394c"
        score={pick('feasibilityScore')}
        reason={pick('feasibilityReason')}
        beforeVal={hasPrev ? node.previousState.feasibilityScore : null}
        afterVal={node.feasibilityScore}
        showAfter={showAfter}
      />

      <FactorBlock
        label="Impact"
        color="#4040a1"
        score={pick('impactScore')}
        reason={pick('impactReason')}
        beforeVal={hasPrev ? node.previousState.impactScore : null}
        afterVal={node.impactScore}
        showAfter={showAfter}
      />
    </>
  );
};

export default IdeaFactorsAndScoresCard;
