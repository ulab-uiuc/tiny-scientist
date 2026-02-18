import React, { useState } from 'react';
import { extractContentSections } from '../utils/contentParser';
import './IdeaCard.css';

const escapeRegExp = (str = '') => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const ExperimentTable = ({ src }) => {
  if (!src || typeof src !== 'string') return null;
  const lines = src.trim().split('\n').filter(l => l.trim());
  if (lines.length < 2) return <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>{src}</pre>;
  const parseRow = (line) =>
    line.split('|').map(c => c.trim()).filter((_, i, arr) => i !== 0 && i !== arr.length - 1);
  const headers = parseRow(lines[0]);
  const dataRows = lines.slice(2).map(parseRow);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', lineHeight: 1.4 }}>
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i} style={{ padding: '6px 10px', backgroundColor: '#F3F4F6', border: '1px solid #E5E7EB', textAlign: 'left', fontWeight: 600, color: '#374151', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, ri) => (
            <tr key={ri} style={{ backgroundColor: ri % 2 === 0 ? '#fff' : '#F9FAFB' }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{ padding: '6px 10px', border: '1px solid #E5E7EB', color: '#4B5563', verticalAlign: 'top' }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const renderWithHighlights = (text, highlights, onClickHighlight) => {
  if (!text || !highlights || highlights.length === 0) return text;
  const sanitized = highlights
    .map(h => (h || '').trim())
    .filter(Boolean);

  if (sanitized.length === 0) return text;

  const regex = new RegExp(`(${sanitized.map(h => escapeRegExp(h)).join('|')})`, 'gi');
  return text.split(regex).map((part, idx) => {
    if (!part) return null;
    const isHit = sanitized.some(h => h.toLowerCase() === part.toLowerCase());
    if (!isHit) return part;
    const payload = part.trim();
    return (
      <span
        key={`hl-${idx}-${payload}`}
        className="problem-highlight"
        onClick={(e) => {
          e.stopPropagation();
          if (payload && onClickHighlight) {
            onClickHighlight(payload);
          }
        }}
      >
        {part}
      </span>
    );
  });
};

/**
 * 显示单条假设，可在 Before / After 之间切换
 * 现在支持动态维度系统：
 * - Tabs 显示 Importance/Difficulty/NoveltyComparison（文本内容）
 * - 如果有 selectedDimensionPairs，则显示动态维度分数
 * - 向后兼容旧的三维度系统（Feasibility/Novelty/Impact）
 */
const IdeaCard = ({
  node,
  showAfter,
  setShowAfter,
  onEditCriteria,
  onModifyScore, // Function to handle score modifications
  showModifyButton = false, // Whether to show modify buttons (for tree view)
  pendingChanges = null, // Pending changes from parent component
  currentView = 'evaluation', // Current view context (exploration/evaluation)
  selectedDimensionPairs = null, // 新增：选中的维度对，用于显示动态分数
  activeDimensions = null, // 当前显示的维度面，用于高亮
  onShowFragmentMenu = null, // 新增:显示 Fragment 菜单的回调
  activeDimensionIndices = [0, 1, 2], // Active dimension indices
  onToggleDimensionIndex = null, // Callback to toggle dimension index
  onCreateFragmentFromHighlight = null, // 新增：点击重点直接生成 fragment
  onSwapDimension = null, // 交换维度方向
  onEditDimension = null, // 编辑维度名称 (pairIndex, anchorRect) => void
}) => {
  const [expandedSections, setExpandedSections] = useState({ experiment: false });

  const toggleSection = (sectionName) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionName]: !prev[sectionName]
    }));
  };

  // Handle root node differently
  if (node.type === 'root') {
    return (
      <div
        style={{
          border: '1px solid #e5e7eb',
          borderRadius: 8,
          padding: '16px',
          marginBottom: 16,
          backgroundColor: 'transparent',
        }}
      >
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>
          {'Research Intent'}
        </h2>

        <div style={{
          fontSize: '0.95rem',
          lineHeight: 1.6,
          padding: '10px',
          backgroundColor: '#f9fafb',
          borderRadius: '4px',
          borderLeft: `3px solid #4C84FF`,
        }}>
          {node.content}
        </div>
      </div>
    );
  }

  // Handle fragment node
  if (node.type === 'fragment') {
    return (
      <div
        style={{
          border: '2px solid #f59e0b',
          borderRadius: 8,
          padding: '16px',
          marginBottom: 16,
          backgroundColor: '#fffbeb',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            backgroundColor: '#fef08a',
            border: '2px solid #f59e0b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '0.875rem',
            fontWeight: '600',
            color: '#92400e',
            marginRight: '12px'
          }}>
            {node.id.split('-S')[1]}
          </div>
          <div>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 700, color: '#92400e', margin: 0 }}>
              Fragment
            </h2>
            <p style={{ fontSize: '0.75rem', color: '#78716c', margin: '2px 0 0 0' }}>
              from {node.parentId}
            </p>
          </div>
        </div>

        <div style={{
          fontSize: '0.95rem',
          lineHeight: 1.6,
          padding: '12px',
          backgroundColor: 'white',
          borderRadius: '6px',
          border: '1px solid #fbbf24',
          whiteSpace: 'pre-wrap'
        }}>
          {node.content}
        </div>

        <div style={{
          marginTop: '12px',
          padding: '8px 12px',
          backgroundColor: '#fef3c7',
          borderRadius: '6px',
          fontSize: '0.75rem',
          color: '#92400e',
          fontStyle: 'italic'
        }}>
          💡 This fragment has no evaluation scores yet. Merge it with other ideas to evaluate.
        </div>
      </div>
    );
  }

  // Regular idea node handling
  const hasPrevious = !!node.previousState;
  const hasNext = !hasPrevious && !!node.modifiedState;
  const alternateNode = hasPrevious
    ? node.previousState
    : (hasNext ? node.modifiedState : null);
  const isShowingAlternate = !!alternateNode && !showAfter;
  const displayNode = isShowingAlternate ? alternateNode : node;
  const content = displayNode?.content || '';

  // TODO: Edit icon for future score editing UI
  // const editIcon = (<svg>...</svg>);

  // 解析内容中的各个章节
  const sections = extractContentSections(content);

  // 找到问题部分（作为主要内容显示）
  const problemSection = sections.find(section => section.title === 'Problem');
  const problemHighlights = node.problemHighlights || node.originalData?.problem_highlights || [];

  // 处理文本选择事件
  const handleMouseUp = (e) => {
    if (!onShowFragmentMenu) {
      console.log('[Fragment Debug] onShowFragmentMenu not provided');
      return;
    }

    // 保存 currentTarget 引用（React 合成事件在异步回调中会被清空）
    const cardElement = e.currentTarget;

    // 延迟获取 selection，确保选择完成
    setTimeout(() => {
      const selection = window.getSelection();
      const selectedText = selection.toString().trim();

      console.log('[Fragment Debug] Text selected:', selectedText);

      if (selectedText && selectedText.length > 0 && selection.rangeCount > 0) {
        // 检查选中的文本是否在当前 IdeaCard 内
        const range = selection.getRangeAt(0);
        if (cardElement && range.commonAncestorContainer &&
          cardElement.contains(range.commonAncestorContainer)) {

          // 检查选中的文本是否在Problem部分内
          let currentNode = range.commonAncestorContainer;
          let inProblemSection = false;

          // 向上遍历DOM树,查找data-section="problem"的元素
          console.log('[Fragment Debug] Starting node:', currentNode);
          let debugPath = [];
          while (currentNode && currentNode !== cardElement) {
            if (currentNode.nodeType === 1) {
              const sectionAttr = currentNode.getAttribute ? currentNode.getAttribute('data-section') : null;
              debugPath.push(`${currentNode.nodeName}[data-section="${sectionAttr}"]`);
              if (sectionAttr === 'problem') {
                inProblemSection = true;
                break;
              }
            }
            currentNode = currentNode.parentNode;
          }

          console.log('[Fragment Debug] DOM path:', debugPath.join(' > '));
          console.log('[Fragment Debug] In Problem section:', inProblemSection);

          // 只有在Problem部分选中文字才显示Fragment菜单
          if (inProblemSection) {
            // 获取选择的屏幕位置
            const rect = range.getBoundingClientRect();
            const x = rect.left + rect.width / 2;
            const y = rect.bottom + 5; // 菜单显示在选中文本下方

            console.log('[Fragment Debug] Showing fragment menu at', x, y);
            onShowFragmentMenu(x, y, selectedText, node.id);
          }
        }
      }
    }, 10);
  };

  // 获取维度分数(如果使用新系统)
  const getDimensionScores = (targetNode) => {
    if (!selectedDimensionPairs || selectedDimensionPairs.length < 2) {
      return null; // 不显示动态维度分数
    }

    if (!targetNode || !targetNode.scores) {
      return null; // 节点还没有动态维度分数
    }

    return selectedDimensionPairs.map((pair, index) => {
      const primaryKey = `${pair.dimensionA}-${pair.dimensionB}`;
      const reverseKey = `${pair.dimensionB}-${pair.dimensionA}`;

      const extractScoreValue = (entry) => {
        if (entry === null || entry === undefined) return null;
        if (typeof entry === 'number') return entry;
        if (typeof entry === 'object' && entry.value !== undefined) return entry.value;
        return null;
      };

      const orientScoreValue = (value, flipped) => {
        if (!flipped || typeof value !== 'number') return value;
        if (value >= -50 && value <= 50) return -value;
        if (value >= 0 && value <= 100) return 100 - value;
        return -value;
      };

      const resolveScore = () => {
        const getScoreEntry = (k) => {
          if (!k) return undefined;
          if (targetNode.scores && targetNode.scores[k] !== undefined) return targetNode.scores[k];
          if (targetNode.originalData && targetNode.originalData.scores && targetNode.originalData.scores[k] !== undefined) {
            return targetNode.originalData.scores[k];
          }
          return undefined;
        };

        const primaryEntry = getScoreEntry(primaryKey);
        const primaryValue = extractScoreValue(primaryEntry);
        if (primaryValue !== null) {
          return { value: primaryValue, reasonKey: primaryKey };
        }

        const reverseEntry = getScoreEntry(reverseKey);
        const reverseValue = extractScoreValue(reverseEntry);
        if (reverseValue !== null) {
          return { value: orientScoreValue(reverseValue, true), reasonKey: reverseKey };
        }

        return { value: null, reasonKey: primaryKey };
      };

      const { value: rawScore, reasonKey } = resolveScore();

      // Helper: extract a reason for this dimension pair from multiple possible locations
      const extractReasonForPair = () => {
        try {
          // 0) PRIORITY: Check for Dimension1Reason / Dimension2Reason directly on node or originalData
          const dimensionReasonKey = `Dimension${index + 1}Reason`;
          if (targetNode[dimensionReasonKey]) return targetNode[dimensionReasonKey];
          if (targetNode.originalData && targetNode.originalData[dimensionReasonKey]) {
            return targetNode.originalData[dimensionReasonKey];
          }          // 1) originalData.scores may contain structured info
          if (targetNode.originalData && targetNode.originalData.scores && targetNode.originalData.scores[reasonKey]) {
            const s = targetNode.originalData.scores[reasonKey];
            if (s && typeof s === 'object') {
              if (s.reason) return s.reason;
              if (s.reasons && Array.isArray(s.reasons)) return s.reasons.join('\n');
            }
          }

          // 2) node.scores entry might be an object with reason
          if (targetNode.scores && targetNode.scores[reasonKey] && typeof targetNode.scores[reasonKey] === 'object') {
            const s = targetNode.scores[reasonKey];
            if (s.reason) return s.reason;
            if (s.reasons && Array.isArray(s.reasons)) return s.reasons.join('\n');
            if (s.value !== undefined) return null; // structured but no reason
          }

          // 3) explicit reason keys on originalData (e.g. "ImpactReason" or "<dim>AReason")
          if (targetNode.originalData) {
            const keys = Object.keys(targetNode.originalData);
            for (const k of keys) {
              if (k.toLowerCase().includes('reason')) {
                const lower = k.toLowerCase();
                const pairKey = reasonKey.toLowerCase();
                // if the reason key references either dimension name or the full pair, prefer it
                if (lower.includes(pairKey) || lower.includes(pair.dimensionA?.toLowerCase()) || lower.includes(pair.dimensionB?.toLowerCase())) {
                  const v = targetNode.originalData[k];
                  if (v) return v;
                }
              }
            }
            // fallback: any Reason field
            for (const k of keys) {
              if (k.toLowerCase().includes('reason')) {
                const v = targetNode.originalData[k];
                if (v) return v;
              }
            }
          }

          // 4) evaluationReasoning fallback (legacy impact/feasibility/novelty)
          if (targetNode.evaluationReasoning) {
            if (targetNode.evaluationReasoning.impactReasoning) return targetNode.evaluationReasoning.impactReasoning;
            if (targetNode.evaluationReasoning.feasibilityReasoning) return targetNode.evaluationReasoning.feasibilityReasoning;
            if (targetNode.evaluationReasoning.noveltyReasoning) return targetNode.evaluationReasoning.noveltyReasoning;
          }
        } catch (err) {
          // ignore and return null
          console.error('Error extracting reason:', err);
        }
        return null;
      };

      const reason = extractReasonForPair();

      return {
        label: `${pair.dimensionA} ←→ ${pair.dimensionB}`,
        score: rawScore !== undefined ? rawScore : null,
        reason,
        pair
      };
    });
  };

  // TODO: Old three-dimension score getter - keep for backward compatibility if needed
  // const getScore = (sectionName) => {
  //   const scoreMap = { 'Impact': 'impactScore', 'Feasibility': 'feasibilityScore', 'Novelty': 'noveltyScore' };
  //   const scoreField = scoreMap[sectionName];
  //   if (!scoreField) return null;
  //   // ... (check pending changes, etc.)
  //   return node[scoreField];
  // };

  // 获取动态维度分数
  const dimensionScores = getDimensionScores(displayNode);

  return (
    <div
      onMouseUp={handleMouseUp}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
        position: 'relative',
        userSelect: 'text', // 允许文本选择
      }}
    >
      {/* Toggle 按钮 */}
      {(hasPrevious || hasNext) && (
        <button
          onClick={() => {
            setShowAfter((prev) => !prev);
          }}
          style={{
            position: 'absolute',
            top: 12,
            right: 16,
            padding: '4px 8px',
            borderRadius: 9999,
            border: 'none',
            cursor: 'pointer',
            backgroundColor: '#E5E7EB',
            fontSize: '0.75rem',
            fontWeight: 500,
          }}
        >
          {hasPrevious
            ? (showAfter ? 'Check Original Idea' : 'Back to Modified Idea')
            : (hasNext
              ? (showAfter ? 'Check Modified Idea' : 'Back to Original Idea')
              : '')}
        </button>
      )}

      {/* 标题 + ID Badge */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '0.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, flex: 1 }}>
          {displayNode?.title || node.title || 'Untitled'}
        </h2>
        {node.id && node.id !== 'root' && (
          <span style={{
            fontSize: '0.7rem',
            fontWeight: 600,
            color: '#6B7280',
            backgroundColor: '#F3F4F6',
            border: '1px solid #E5E7EB',
            borderRadius: '4px',
            padding: '2px 6px',
            whiteSpace: 'nowrap',
            flexShrink: 0,
            marginTop: '4px',
            fontFamily: 'monospace'
          }}>
            #{node.id}
          </span>
        )}
      </div>

      {/* 问题部分 - 直接显示在标题下方，添加下横线 */}
      {problemSection && problemSection.content ? (
        <div
          data-section="problem"
          className="problem-content"
          style={{
            fontSize: '0.95rem',
            lineHeight: 1.5,
            color: '#374151',
            padding: '10px 0',
            marginBottom: '15px',
            borderBottom: '2px solid #e5e7eb',
            paddingBottom: '15px'
          }}>
          {renderWithHighlights(problemSection.content, problemHighlights, (text) => {
            if (onCreateFragmentFromHighlight) {
              onCreateFragmentFromHighlight(text, node.id);
            }
          })}
        </div>
      ) : content && content.trim() && !sections.length ? (
        // Fallback: Show full content if no structured sections
        <div className="problem-content" style={{
          fontSize: '0.95rem',
          lineHeight: 1.5,
          color: '#374151',
          padding: '10px',
          marginBottom: '15px',
          backgroundColor: '#f9fafb',
          borderRadius: '4px',
          borderLeft: '3px solid #4C84FF',
          whiteSpace: 'pre-wrap',
          borderBottom: '2px solid #e5e7eb',
          paddingBottom: '15px'
        }}>
          {renderWithHighlights(content, problemHighlights, (text) => {
            if (onCreateFragmentFromHighlight) {
              onCreateFragmentFromHighlight(text, node.id);
            }
          })}
        </div>
      ) : null}

      {/* 动态维度分数显示（如果有）- 每个dimension一整行 */}
      {dimensionScores && dimensionScores.length > 0 && (
        <div style={{
          marginBottom: '15px',
          borderBottom: '1px solid #e5e7eb',
          paddingBottom: '15px'
        }}>
          {dimensionScores.map((dim, idx) => {
            // Determine signed display (-50..+50)
            let signed = null;
            if (dim.score !== null && dim.score !== undefined) {
              const v = Number(dim.score);
              if (!Number.isNaN(v)) {
                if (v >= -50 && v <= 50) {
                  signed = Math.round(v);
                } else if (v >= 0 && v <= 100) {
                  signed = Math.round(v) - 50;
                } else {
                  signed = Math.max(-50, Math.min(50, Math.round(v)));
                }
              }
            }

            // Color mapping per axis role
            const xPosStyle = { background: '#dbeafe', color: '#1e3a8a' };
            const xNegStyle = { background: '#fef3c7', color: '#92400e' };
            const yPosStyle = { background: '#dcfce7', color: '#14532d' };
            const yNegStyle = { background: '#fce7f3', color: '#831843' };
            let scoreColor = '#374151';
            let scoreBg = 'transparent';

            const activeX = activeDimensions?.xDimension;
            const activeY = activeDimensions?.yDimension;
            const isX = activeX && (`${activeX.dimensionA}-${activeX.dimensionB}`) === `${dim.pair.dimensionA}-${dim.pair.dimensionB}`;
            const isY = activeY && (`${activeY.dimensionA}-${activeY.dimensionB}`) === `${dim.pair.dimensionA}-${dim.pair.dimensionB}`;

            if (signed !== null) {
              if (isX) {
                if (signed >= 0) {
                  scoreColor = xPosStyle.color;
                  scoreBg = xPosStyle.background;
                } else {
                  scoreColor = xNegStyle.color;
                  scoreBg = xNegStyle.background;
                }
              } else if (isY) {
                if (signed >= 0) {
                  scoreColor = yPosStyle.color;
                  scoreBg = yPosStyle.background;
                } else {
                  scoreColor = yNegStyle.color;
                  scoreBg = yNegStyle.background;
                }
              } else {
                if (signed >= 0) {
                  scoreColor = xPosStyle.color;
                  scoreBg = xPosStyle.background;
                } else {
                  scoreColor = xNegStyle.color;
                  scoreBg = xNegStyle.background;
                }
              }
            }

            // Parse dimension label to get left and right parts
            const labelParts = dim.label.split('←→').map(s => s.trim());
            const leftDim = labelParts[0] || '';
            const rightDim = labelParts[1] || '';
            const isActiveDimension = activeDimensionIndices.includes(idx);
            const canSwap = isActiveDimension && !!onSwapDimension;

            return (
              <div key={idx} style={{
                marginBottom: idx < dimensionScores.length - 1 ? '12px' : '0',
                fontSize: '0.95rem',
                lineHeight: 1.5,
                color: '#374151'
              }}>
                {/* Dimension label and score on same line */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  marginBottom: dim.reason ? '6px' : '0'
                }}>
                  {/* Checkbox for toggling dimension */}
                  <input
                    type="checkbox"
                    checked={isActiveDimension}
                    onChange={(e) => {
                      e.stopPropagation(); // Prevent card selection if any
                      if (onToggleDimensionIndex) onToggleDimensionIndex(idx);
                    }}
                    style={{ cursor: 'pointer', width: '16px', height: '16px', accentColor: '#2563EB' }}
                  />

                  {/* Left dimension - highlight if score is negative, clickable to edit */}
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onEditDimension) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        onEditDimension(idx, { top: rect.bottom + 5, left: rect.left });
                      }
                    }}
                    style={{
                      fontWeight: 500,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: signed !== null && signed < 0 ? scoreBg : 'transparent',
                      color: signed !== null && signed < 0 ? scoreColor : '#374151',
                      cursor: onEditDimension ? 'pointer' : 'default',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                    title={onEditDimension ? 'Click to edit dimension' : undefined}
                  >
                    {leftDim}
                    {onEditDimension && (
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5, marginLeft: '4px' }}>
                        <path d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.468-.325z" />
                      </svg>
                    )}
                  </span>

                  {/* Score in the middle - clickable to swap */}
                  {dim.score !== null ? (
                    <span
                      onClick={(e) => {
                        e.stopPropagation();
                        if (canSwap) onSwapDimension(idx);
                      }}
                      style={{
                        padding: '2px 10px',
                        borderRadius: '6px',
                        fontSize: '0.9rem',
                        fontWeight: 700,
                        backgroundColor: scoreBg,
                        color: scoreColor,
                        cursor: canSwap ? 'pointer' : 'default',
                        opacity: canSwap ? 1 : 0.35,
                        border: canSwap ? '1px solid rgba(55,65,81,0.15)' : '1px solid transparent',
                        userSelect: 'none',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                      title={canSwap ? 'Click to swap dimension direction' : 'Enable this dimension to swap'}
                    >
                      {signed !== null ? (signed > 0 ? `+${signed}` : signed) : 'N/A'}
                      {canSwap && (
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.6, marginLeft: '4px' }}>
                          <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z" />
                          <path fillRule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z" />
                        </svg>
                      )}
                    </span>
                  ) : (
                    <span style={{
                      fontSize: '0.875rem',
                      color: '#9CA3AF',
                      fontStyle: 'italic'
                    }}>
                      N/A
                    </span>
                  )}

                  {/* Right dimension - highlight if score is positive, clickable to edit */}
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onEditDimension) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        onEditDimension(idx, { top: rect.bottom + 5, left: rect.left });
                      }
                    }}
                    style={{
                      fontWeight: 500,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: signed !== null && signed > 0 ? scoreBg : 'transparent',
                      color: signed !== null && signed > 0 ? scoreColor : '#374151',
                      cursor: onEditDimension ? 'pointer' : 'default',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                    title={onEditDimension ? 'Click to edit dimension' : undefined}
                  >
                    {rightDim}
                    {onEditDimension && (
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5, marginLeft: '4px' }}>
                        <path d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.468-.325z" />
                      </svg>
                    )}
                  </span>
                </div>

                {/* Reason text below */}
                {dim.reason && (
                  <div style={{
                    fontSize: '0.95rem',
                    lineHeight: 1.5,
                    color: '#6B7280',
                    paddingLeft: '0',
                    whiteSpace: 'pre-wrap'
                  }}>
                    <span className="dimension-reasoning">
                      {renderWithHighlights(dim.reason, problemHighlights, (text) => {
                        if (onCreateFragmentFromHighlight) {
                          onCreateFragmentFromHighlight(text, node.id);
                        }
                      })}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Collapsible Detail Sections */}
      {displayNode?.originalData && (
        <div style={{ marginTop: '15px', borderTop: '1px solid #e5e7eb', paddingTop: '15px' }}>
          {/* Approach Section */}
          {displayNode.originalData.Approach && (
            <div style={{ marginBottom: '10px' }}>
              <button
                onClick={() => toggleSection('approach')}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#374151',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
              >
                <span>🔬 Approach</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  style={{
                    transform: expandedSections.approach ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}
                >
                  <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z" />
                </svg>
              </button>
              {expandedSections.approach && (
                <div style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                  color: '#4B5563',
                  whiteSpace: 'pre-wrap'
                }}>
                  {displayNode.originalData.Approach}
                </div>
              )}
            </div>
          )}

          {/* Experiment Section */}
          {(displayNode.originalData.Experiment || displayNode.originalData.ExperimentTable) && (
            <div style={{ marginBottom: '10px' }}>
              <button
                onClick={() => toggleSection('experiment')}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#374151',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
              >
                <span>🧪 Experiment Plan</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  style={{
                    transform: expandedSections.experiment ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}
                >
                  <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z" />
                </svg>
              </button>
              {expandedSections.experiment && (
                <div style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                  color: '#4B5563'
                }}>
                  {displayNode.originalData.ExperimentTable ? (
                    <ExperimentTable src={displayNode.originalData.ExperimentTable} />
                  ) : typeof displayNode.originalData.Experiment === 'object' ? (
                    <div>
                      {displayNode.originalData.Experiment.Model && (
                        <div style={{ marginBottom: '8px' }}>
                          <strong>Model:</strong> {displayNode.originalData.Experiment.Model}
                        </div>
                      )}
                      {displayNode.originalData.Experiment.Dataset && (
                        <div style={{ marginBottom: '8px' }}>
                          <strong>Dataset:</strong>{' '}
                          {typeof displayNode.originalData.Experiment.Dataset === 'object'
                            ? JSON.stringify(displayNode.originalData.Experiment.Dataset, null, 2)
                            : displayNode.originalData.Experiment.Dataset}
                        </div>
                      )}
                      {displayNode.originalData.Experiment.Metric && (
                        <div>
                          <strong>Metric:</strong> {displayNode.originalData.Experiment.Metric}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      {displayNode.originalData.Experiment}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Importance Section */}
          {displayNode.originalData.Importance && (
            <div style={{ marginBottom: '10px' }}>
              <button
                onClick={() => toggleSection('importance')}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#374151',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
              >
                <span>⭐ Importance</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  style={{
                    transform: expandedSections.importance ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}
                >
                  <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z" />
                </svg>
              </button>
              {expandedSections.importance && (
                <div style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                  color: '#4B5563',
                  whiteSpace: 'pre-wrap'
                }}>
                  {displayNode.originalData.Importance}
                </div>
              )}
            </div>
          )}

          {/* Difficulty Section */}
          {displayNode.originalData.Difficulty && (
            <div style={{ marginBottom: '10px' }}>
              <button
                onClick={() => toggleSection('difficulty')}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#374151',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
              >
                <span>⚠️ Difficulty</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  style={{
                    transform: expandedSections.difficulty ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}
                >
                  <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z" />
                </svg>
              </button>
              {expandedSections.difficulty && (
                <div style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                  color: '#4B5563',
                  whiteSpace: 'pre-wrap'
                }}>
                  {displayNode.originalData.Difficulty}
                </div>
              )}
            </div>
          )}

          {/* NoveltyComparison Section */}
          {displayNode.originalData.NoveltyComparison && (
            <div style={{ marginBottom: '10px' }}>
              <button
                onClick={() => toggleSection('noveltyComparison')}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#374151',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
              >
                <span>🆕 Novelty Comparison</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  style={{
                    transform: expandedSections.noveltyComparison ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}
                >
                  <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z" />
                </svg>
              </button>
              {expandedSections.noveltyComparison && (
                <div style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  lineHeight: 1.6,
                  color: '#4B5563',
                  whiteSpace: 'pre-wrap'
                }}>
                  {displayNode.originalData.NoveltyComparison}
                </div>
              )}
            </div>
          )}

        </div>
      )}

    </div>
  );
};

export default IdeaCard;
