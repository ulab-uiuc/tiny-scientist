import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Helper function to parse the specific markdown table format
const parseMarkdownTable = (markdown) => {
  if (!markdown || typeof markdown !== 'string') return [];
  const lines = markdown.trim().split('\n');
  if (lines.length < 3) return []; // Header, separator, at least one data row

  // Assumes fixed headers: | Component | Specification | Justification / Rationale | Status |
  const dataLines = lines.slice(2); // Skip header and separator

  return dataLines.map(line => {
    const cells = line.split('|').map(cell => cell.trim());
    // cells[0] is empty because of the leading '|'
    if (cells.length < 5) return null; // Handle malformed rows
    return {
      Component: cells[1] || '',
      Specification: cells[2] || '',
      'Justification / Rationale': cells[3] || '',
      Status: cells[4] || ''
    };
  }).filter(Boolean); // Filter out any null rows
};

// Helper function to serialize data back to a markdown table string
const serializeToMarkdown = (data) => {
  const header = '| Component | Specification | Justification / Rationale | Status |';
  const separator = '|---|---|---|---|';
  const rows = data.map(row =>
    `| ${row.Component} | ${row.Specification.replace(/\n/g, '<br />')} | ${row['Justification / Rationale'].replace(/\n/g, '<br />')} | ${row.Status} |`
  );
  return [header, separator, ...rows].join('\n');
};

/**
 * A new component for interactively editing the experiment table.
 */
const EditableTable = ({ content, onSave, onCancel }) => {
  const [tableData, setTableData] = useState(() => parseMarkdownTable(content));
  const originalData = useRef(parseMarkdownTable(content)); // To store for reset

  const handleCellChange = (index, field, value) => {
    const newData = [...tableData];
    newData[index][field] = value;
    setTableData(newData);
  };

  const handleSave = () => {
    const newMarkdown = serializeToMarkdown(tableData);
    onSave(newMarkdown);
  };

  const handleReset = () => {
    setTableData(originalData.current);
  };

  return (
    <div>
      <div style={{ maxHeight: '55vh', overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: '4px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, backgroundColor: 'white' }}>
            <tr>
              {['Component', 'Specification', 'Justification / Rationale', 'Status'].map(header => (
                <th key={header} style={{ border: '1px solid #ddd', padding: '12px', textAlign: 'left', backgroundColor: '#f6f8fa', fontWeight: 600 }}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableData.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <td style={{ border: '1px solid #ddd', padding: '8px', verticalAlign: 'top', fontWeight: 500 }}>{row.Component}</td>
                <td style={{ border: '1px solid #ddd', padding: '0' }}>
                  <textarea
                    value={row.Specification}
                    onChange={(e) => handleCellChange(rowIndex, 'Specification', e.target.value)}
                    style={{ width: '100%', height: '100%', minHeight: '80px', border: 'none', padding: '8px', resize: 'vertical', boxSizing: 'border-box' }}
                  />
                </td>
                <td style={{ border: '1px solid #ddd', padding: '0' }}>
                  <textarea
                    value={row['Justification / Rationale']}
                    onChange={(e) => handleCellChange(rowIndex, 'Justification / Rationale', e.target.value)}
                    style={{ width: '100%', height: '100%', minHeight: '80px', border: 'none', padding: '8px', resize: 'vertical', boxSizing: 'border-box' }}
                  />
                </td>
                <td style={{ border: '1px solid #ddd', padding: '8px', verticalAlign: 'top' }}>{row.Status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px', gap: '10px' }}>
         <button onClick={onCancel} style={{ padding: '8px 16px', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer', backgroundColor: '#fff' }}>Cancel</button>
         <button onClick={handleReset} style={{ padding: '8px 16px', border: '1px solid #e5e7eb', borderRadius: '4px', cursor: 'pointer', backgroundColor: '#f9fafb', color: '#dc2626' }}>Reset</button>
         <button onClick={handleSave} style={{ padding: '8px 16px', border: 'none', borderRadius: '4px', cursor: 'pointer', backgroundColor: '#4C84FF', color: 'white' }}>Save</button>
      </div>
    </div>
  );
};


/**
 * A reusable modal component, now with an optional onEdit handler.
 */
const Modal = ({ isOpen, onClose, children, title, onEdit }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#fff',
          borderRadius: '8px',
          padding: '24px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          border: '1px solid #e5e7eb',
          width: '80%',
          maxWidth: '900px',
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #e5e7eb',
          paddingBottom: '12px',
          marginBottom: '16px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600, color: '#111827' }}>
              {title || 'Details'}
            </h3>
            {/* Edit button only appears if onEdit function is provided */}
            {onEdit && (
              <button
                onClick={onEdit}
                style={{
                  padding: '4px 10px',
                  backgroundColor: '#f3f4f6',
                  border: '1px solid #d1d5db',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.8rem',
                  fontWeight: 500,
                }}
              >
                ✏️ Edit
              </button>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              color: '#6b7280',
            }}
          >
            ×
          </button>
        </div>
        <div style={{ overflowY: 'auto', flexGrow: 1 }}>
          {children}
        </div>
      </div>
    </div>
  );
};


/**
 * A styled component to render Markdown content.
 */
const MarkdownRenderer = ({ content }) => {
  const tableStyles = `
    .markdown-table-container table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }
    .markdown-table-container th, .markdown-table-container td { border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }
    .markdown-table-container th { background-color: #f6f8fa; font-weight: 600; }
    .markdown-table-container tr:nth-child(even) { background-color: #f9fafb; }
  `;
  const safeContent = typeof content === 'string' ? content : '';
  return (
    <div className="markdown-table-container">
      <style>{tableStyles}</style>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {safeContent || "No content available."}
      </ReactMarkdown>
    </div>
  );
};


/**
 * The main card component with editable experiment plans.
 */
const IdeaCard = ({
  node,
  showAfter,
  setShowAfter,
  onEditCriteria,
  activeSection,
  setActiveSection,
  onUpdateTable, // Receive the update handler
}) => {
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [showExperimentModal, setShowExperimentModal] = useState(false);
  const [isEditingExperiment, setIsEditingExperiment] = useState(false);

  const buttonRefs = useRef({});

  if (node.type === 'root') {
    return (
      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: '16px', marginBottom: 16, backgroundColor: 'transparent' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>Research Intent</h2>
        <div style={{ fontSize: '0.95rem', lineHeight: 1.6, padding: '10px', backgroundColor: '#f9fafb', borderRadius: '4px', borderLeft: `3px solid #4C84FF` }}>
          {node.content}
        </div>
      </div>
    );
  }

  const hasPrevious = !!node.previousState;
  const displayedNode = hasPrevious && !showAfter ? node.previousState : node;

  const { originalData } = displayedNode;
  const description = originalData?.Description || '';
  const importance = originalData?.Importance || '';
  const difficulty = originalData?.Difficulty || '';
  const noveltyComparison = originalData?.NoveltyComparison || '';
  const comparisonTable = originalData?.ComparisonTable || '';
  const experimentTable = originalData?.ExperimentTable || '';

  let activeContent = '';
  if (activeSection === 'Impact') activeContent = importance;
  else if (activeSection === 'Feasibility') activeContent = difficulty;
  else if (activeSection === 'Novelty') activeContent = noveltyComparison;

  const editIcon = (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );

  const getScore = (sectionName) => {
    const scoreMap = { 'Impact': 'impactScore', 'Feasibility': 'feasibilityScore', 'Novelty': 'noveltyScore' };
    const scoreField = scoreMap[sectionName];
    if (!scoreField) return null;
    return hasPrevious && !showAfter ? node.previousState[scoreField] : node[scoreField];
  };

  const sectionColors = {
    "Impact": "#4040a1",
    "Feasibility": "#50394c",
    "Novelty": "#618685",
  };

  const tabs = ['Impact', 'Feasibility', 'Novelty'];

  return (
    <>
      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16, position: 'relative' }}>
        {hasPrevious && (
          <button onClick={() => setShowAfter((prev) => !prev)} style={{ position: 'absolute', top: 12, right: 16, padding: '4px 8px', borderRadius: 9999, border: 'none', cursor: 'pointer', backgroundColor: '#E5E7EB', fontSize: '0.75rem', fontWeight: 500 }}>
            {showAfter ? 'Check Original Idea' : 'Check Modified Idea'}
          </button>
        )}

        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>
          {displayedNode.title || 'Untitled'}
        </h2>

        <div style={{ fontSize: '0.95rem', lineHeight: 1.5, color: '#374151', padding: '10px 0', borderBottom: '1px solid #e5e7eb' }}>
          {description}
        </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: '15px', marginBottom: '15px' }}>
          {comparisonTable && (
            <button onClick={() => setShowComparisonModal(true)} style={{ flex: 1, padding: '10px', backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '6px', cursor: 'pointer', textAlign: 'left', fontSize: '0.9rem', fontWeight: 600, color: '#374151' }}>
              📊 View Comparison Table
            </button>
          )}
          {experimentTable && (
            <button onClick={() => setShowExperimentModal(true)} style={{ flex: 1, padding: '10px', backgroundColor: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '6px', cursor: 'pointer', textAlign: 'left', fontSize: '0.9rem', fontWeight: 600, color: '#374151' }}>
              🧪 View Experiment Plan
            </button>
          )}
        </div>

        <div style={{ display: 'flex', borderBottom: '1px solid #e5e7eb', marginBottom: '15px' }}>
          {tabs.map(tab => {
            const isActive = activeSection === tab;
            const score = getScore(tab);
            const color = sectionColors[tab];
            const sectionKey = tab.toLowerCase().replace(' ', '_');
            if (!buttonRefs.current[sectionKey]) buttonRefs.current[sectionKey] = React.createRef();
            return (
              <div key={tab} onClick={() => setActiveSection(tab)} style={{ padding: '8px 12px', marginRight: '10px', cursor: 'pointer', position: 'relative', fontWeight: isActive ? 600 : 400, color: isActive ? color : '#6b7280', borderBottom: isActive ? `2px solid ${color}` : 'none', display: 'flex', alignItems: 'center' }}>
                {isActive && <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color, marginRight: 8, display: 'block' }} />}
                {tab}
                <button ref={buttonRefs.current[sectionKey]} onClick={(e) => { e.stopPropagation(); onEditCriteria(buttonRefs.current[sectionKey], sectionKey); }} style={{ marginLeft: '6px', padding: '2px', backgroundColor: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', opacity: 0.6, color: '#6b7280' }} title={`Edit ${tab} criteria`}>
                  {editIcon}
                </button>
                {score !== null && (
                  <span style={{ marginLeft: '8px', fontSize: '0.85rem', backgroundColor: isActive ? `${color}20` : '#f3f4f6', padding: '2px 5px', borderRadius: '4px', color: isActive ? color : '#6b7280' }}>
                    {Math.round(score)}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <div style={{ fontSize: '0.95rem', lineHeight: 1.6, color: '#374151', padding: '10px', backgroundColor: '#f9fafb', borderRadius: '4px', borderLeft: `3px solid ${sectionColors[activeSection] || '#ccc'}` }}>
          <MarkdownRenderer content={activeContent} />
        </div>
      </div>

      <Modal isOpen={showComparisonModal} onClose={() => setShowComparisonModal(false)} title="Novelty Comparison">
        <MarkdownRenderer content={comparisonTable} />
      </Modal>

      <Modal
        isOpen={showExperimentModal}
        onClose={() => {
          setShowExperimentModal(false);
          setIsEditingExperiment(false); // Ensure edit mode is off when closing
        }}
        title={isEditingExperiment ? "Edit Experiment Plan" : "Experiment Plan"}
        onEdit={!isEditingExperiment ? () => setIsEditingExperiment(true) : undefined}
      >
        {isEditingExperiment ? (
          <EditableTable
            content={experimentTable}
            onSave={(newMarkdown) => {
              onUpdateTable(displayedNode.id, 'ExperimentTable', newMarkdown);
              setIsEditingExperiment(false);
            }}
            onCancel={() => setIsEditingExperiment(false)}
          />
        ) : (
          <MarkdownRenderer content={experimentTable} />
        )}
      </Modal>
    </>
  );
};

export default IdeaCard;
