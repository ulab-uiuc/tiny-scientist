import React, { useState, useRef } from 'react';

/**
 * 显示单条假设，可在 Before / After 之间切换
 * 问题部分直接显示在标题下方，其余三个部分（Importance/Feasibility/Novelty）可通过标签切换
 */
const HypothesisCard = ({
  node,
  showAfter,
  setShowAfter,
  onEditCriteria,
  activeSection, // Receive state from parent
  setActiveSection, // Receive setter from parent
}) => {
  const [hoveredTab, setHoveredTab] = useState(null); // This state remains local


  // Create a ref to hold an object of refs for each edit button
  const buttonRefs = useRef({});
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
  const hasPrevious = !!node.previousState;
  const content =
    hasPrevious && !showAfter ? node.previousState.content : node.content;

  // Edit icon SVG
  const editIcon = (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );

  // 解析内容中的各个章节
  const parseContent = (content) => {
    // 定义需要识别的章节标题 (支持多种格式)
    const sections = [
      { title: "Description", regex: /\*\*Description:\*\*|Description:|Description\s*\*\*/ },
      { title: "Impact", regex: /\*\*Impact:\*\*|Impact:|Impact\s*\*\*/ },
      { title: "Feasibility", regex: /\*\*Feasibility:\*\*|Feasibility:|Feasibility\s*\*\*/ },
      { title: "Novelty", regex: /\*\*Novelty Comparison:\*\*|Novelty Comparison:|Novelty:|Novelty\s*\*\*/ }
    ];

    // 如果内容为空，返回空数组
    if (!content) return [];

    // 提取各个章节
    const parsedSections = [];

    // 找出所有匹配的章节位置
    const allMatches = [];
    sections.forEach(section => {
      const displayTitle = section.displayTitle || section.title;
      let match;
      let tempContent = content;
      let offset = 0;

      // Find all occurrences of the section
      while ((match = tempContent.match(section.regex)) !== null) {
        const startPos = match.index + offset;
        const matchLength = match[0].length;

        allMatches.push({
          title: displayTitle,
          position: startPos,
          length: matchLength
        });

        // Move past this match for the next iteration
        offset += match.index + matchLength;
        tempContent = content.substring(offset);
      }
    });

    // Sort matches by their position in the content
    allMatches.sort((a, b) => a.position - b.position);

    // Extract content between section headings
    for (let i = 0; i < allMatches.length; i++) {
      const currentMatch = allMatches[i];
      const startPos = currentMatch.position + currentMatch.length;

      // Find the end position (either the next section or the end of content)
      let endPos = content.length;
      if (i < allMatches.length - 1) {
        endPos = allMatches[i + 1].position;
      }

      // Extract the content
      const sectionContent = content.substring(startPos, endPos).trim();

      // Add to parsed sections if there's actual content
      if (sectionContent) {
        parsedSections.push({
          title: currentMatch.title,
          content: sectionContent
        });
      }
    }

    return parsedSections;
  };

  // 从内容中获取所有部分
  const sections = parseContent(content);

  // 找到问题部分
  const descriptionSection = sections.find(section => section.title === 'Description');

  // 过滤出三个主要评分部分
  const scoreSections = sections.filter(section =>
    ['Impact', 'Feasibility', 'Novelty'].includes(section.title)
  );

  // 获取当前激活部分的内容
  const activeContent = scoreSections.find(section => section.title === activeSection)?.content || '';

  // 获取评分 (从节点或前一状态)
  const getScore = (sectionName) => {
    const scoreMap = {
      'Impact': 'impactScore',
      'Feasibility': 'feasibilityScore',
      'Novelty': 'noveltyScore'
    };

    const scoreField = scoreMap[sectionName];
    if (!scoreField) return null;

    return hasPrevious && !showAfter
      ? node.previousState[scoreField]
      : node[scoreField];
  };

  // 各部分的颜色
  const sectionColors = {
    "Impact": "#4040a1",    // Blue
    "Feasibility": "#50394c",   // Purple
    "Novelty": "#618685",       // Green
  };

  // 处理标签切换
  const handleSectionChange = (newSection) => {
    setActiveSection(newSection);
  };

  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
        position: 'relative',
      }}
    >
      {/* Toggle 按钮 */}
      {hasPrevious && (
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
          {showAfter ? 'Check Original Idea' : 'Check Modified Idea'}
        </button>
      )}

      {/* 标题 */}
      <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>
        {node.title || 'Untitled'}
      </h2>

      {/* 问题部分 - 直接显示在标题下方 */}
      {descriptionSection && (
        <div style={{
          fontSize: '0.95rem',
          lineHeight: 1.5,
          color: '#374151',
          padding: '10px 0',
          marginBottom: '15px',
          borderBottom: '1px solid #e5e7eb'
        }}>
          {descriptionSection.content}
        </div>
      )}

      {/* 部分选择标签 */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid #e5e7eb',
        marginBottom: '15px'
      }}>
        {scoreSections.map(section => {
          const isActive = activeSection === section.title;
          const score = getScore(section.title);
          const color = sectionColors[section.title];
          const sectionKey = section.title.toLowerCase();

          // Ensure a ref object exists for the current section's button
          if (!buttonRefs.current[sectionKey]) {
            buttonRefs.current[sectionKey] = React.createRef();
          }

          return (
            <div
              key={section.title}
              onClick={() => handleSectionChange(section.title)}
              onMouseEnter={() => setHoveredTab(section.title)}
              onMouseLeave={() => setHoveredTab(null)}
              style={{
                padding: '8px 12px',
                marginRight: '10px',
                cursor: 'pointer',
                position: 'relative',
                fontWeight: isActive ? 600 : 400,
                color: isActive ? color : '#6b7280',
                borderBottom: isActive ? `2px solid ${color}` : 'none',
                display: 'flex',
                alignItems: 'center',
                transition: 'color 0.1s ease, border-bottom 0.1s ease',
                opacity: activeSection === section.title ? 1 : 0.8,
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: color,
                  marginRight: 8,
                  display: isActive ? 'block' : 'none'
                }}
              />
              {section.title}

              {/* Edit icon */}
              <button
                ref={buttonRefs.current[sectionKey]} // Assign the ref to the button
                onClick={(e) => {
                  e.stopPropagation();
                  onEditCriteria(buttonRefs.current[sectionKey], sectionKey);
                }}
                style={{
                  marginLeft: '6px',
                  padding: '2px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  opacity: 0.6,
                  color: '#6b7280',
                }}
                title={`Edit ${section.title} criteria`}
              >
                {editIcon}
              </button>

              {score !== null && (
                <span style={{
                  marginLeft: '8px',
                  fontSize: '0.85rem',
                  backgroundColor: isActive ? `${color}20` : '#f3f4f6',
                  padding: '2px 5px',
                  borderRadius: '4px',
                  color: isActive ? color : '#6b7280',
                }}>
                  {Math.round(score)}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* 所选部分的内容 */}
      <div
        style={{
          fontSize: '0.95rem',
          lineHeight: 1.6,
          color: '#374151',
          padding: '10px',
          backgroundColor: '#f9fafb',
          borderRadius: '4px',
          borderLeft: `3px solid ${sectionColors[activeSection]}`,
        }}
      >
        {activeContent}
      </div>
    </div>
  );
};

export default HypothesisCard;
