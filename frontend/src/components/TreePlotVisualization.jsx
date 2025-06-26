// ============== 段落1：导入依赖 ==============
import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import Editor from '@monaco-editor/react';

import TopNav from './TopNav';
import HypothesisCard from './HypothesisCard';
import HypothesisFactorsAndScoresCard from './HypothesisFactorsAndScoresCard';


// Helper components defined outside the main component to preserve state
const ContextAndGenerateCard = ({
  isAddingCustom,
  userInput,
  setUserInput,
  generateChildNodes,
  selectedNode,
  isGenerating,
  isEvaluating,
  isAnalysisSubmitted,
  setIsAddingCustom,
  handleAddCustomHypothesis,
  customHypothesis,
  setCustomHypothesis,
  setIsEditingSystemPrompt,
  setModalAnchorEl,
  editIcon,
  handleProceedWithSelectedIdea
}) => {
  const newEditButtonRef = useRef(null);
  const generateButtonRef = useRef(null);
  const addCustomButtonRef = useRef(null);
  const contextInputRef = useRef(null);
  const customFormRef = useRef(null);

  // Removed user action tracking

  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        backgroundColor: '#FAFAFA',
        padding: '16px',
        marginBottom: '16px',
      }}
    >
      {!isAddingCustom ? (
        <div onClick={(e) => e.stopPropagation()}>
          <label htmlFor="context-input" style={{ marginBottom: '8px', fontSize: '0.875rem', color: '#6b7280', display: 'block' }}>
            Add context for new hypotheses (optional)
          </label>
          <textarea
            ref={contextInputRef}
            id="context-input"
            rows={2}
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              fontSize: '0.875rem',
              padding: '8px',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              resize: 'none',
              marginBottom: '10px',
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center' }}>
            {/* --- Button Group Start --- */}
            <div style={{ display: 'flex', alignItems: 'center', marginRight: '8px' }}>
              <button
                ref={generateButtonRef}
                onClick={(e) => {
                  e.stopPropagation();
                  generateChildNodes();
                }}
                disabled={!selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted}
                style={{
                  padding: '0 16px',
                  backgroundColor: '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px 0 0 6px',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  cursor: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 'not-allowed' : 'pointer',
                  opacity: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 0.6 : 1,
                  height: '38px',
                  boxSizing: 'border-box',
                }}
              >
                Generate New Hypotheses
              </button>

              <button
                ref={newEditButtonRef}
                type="button"
                aria-label="Edit system prompt"
                title="Edit system prompt"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setModalAnchorEl(newEditButtonRef);
                  setIsEditingSystemPrompt(true);
                }}
                style={{
                  padding: '0 12px',
                  backgroundColor: '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderLeft: '1px solid #6D97FF',
                  borderRadius: '0 6px 6px 0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '38px',
                  boxSizing: 'border-box',
                  cursor: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 'not-allowed' : 'pointer',
                  opacity: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 0.6 : 1,
                }}
              >
                {editIcon}
              </button>
            </div>
            {/* --- Button Group End --- */}

            <button
              ref={addCustomButtonRef}
              onClick={(e) => {
                e.stopPropagation();
                setIsAddingCustom(true);
              }}
              style={{
                padding: '0 16px',
                backgroundColor: '#fff',
                color: '#4C84FF',
                border: '1px solid #4C84FF',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: 500,
                cursor: 'pointer',
                height: '38px',
                boxSizing: 'border-box',
                marginRight: '8px',
              }}
            >
              Add Custom Hypothesis
            </button>

            <button
              onClick={(e) => {
                console.log("=== BUTTON CLICK DETECTED ===");
                console.log("Event:", e);
                console.log("selectedNode:", selectedNode);
                console.log("isGenerating:", isGenerating);
                console.log("isEvaluating:", isEvaluating);
                console.log("Button disabled:", !selectedNode || isGenerating || isEvaluating);

                e.stopPropagation();
                handleProceedWithSelectedIdea();
              }}
              disabled={!selectedNode || isGenerating || isEvaluating}
              style={{
                padding: '0 16px',
                backgroundColor: selectedNode && !isGenerating && !isEvaluating ? '#4C84FF' : '#9CA3AF',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: 500,
                cursor: selectedNode && !isGenerating && !isEvaluating ? 'pointer' : 'not-allowed',
                height: '38px',
                boxSizing: 'border-box',
              }}
            >
              Proceed
            </button>
          </div>
        </div>
      ) : (
        <form
          ref={customFormRef}
          onSubmit={(e) => {
            e.stopPropagation();
            handleAddCustomHypothesis(e);
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ marginBottom: '12px' }}>
            <label htmlFor="custom-hypothesis-title" style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
              Title (2-3 words)
            </label>
            <input
              id="custom-hypothesis-title"
              type="text"
              value={customHypothesis.title}
              onChange={(e) => setCustomHypothesis(prev => ({ ...prev, title: e.target.value }))}
              onClick={(e) => e.stopPropagation()}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                fontSize: '0.875rem',
              }}
              required
            />
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label htmlFor="custom-hypothesis-content" style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
              Hypothesis Content
            </label>
            <textarea
              id="custom-hypothesis-content"
              value={customHypothesis.content}
              onChange={(e) => setCustomHypothesis(prev => ({ ...prev, content: e.target.value }))}
              onClick={(e) => e.stopPropagation()}
              rows={4}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                fontSize: '0.875rem',
                resize: 'vertical',
              }}
              required
            />
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="submit"
              disabled={isGenerating || isEvaluating}
              onClick={(e) => e.stopPropagation()}
              style={{
                padding: '8px 16px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '0.875rem',
                cursor: isGenerating || isEvaluating ? 'not-allowed' : 'pointer',
                opacity: isGenerating || isEvaluating ? 0.6 : 1,
              }}
            >
              Add Hypothesis
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsAddingCustom(false);
                setCustomHypothesis({ title: '', content: '' });
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
};


const Dashboard = ({
  nodeId,
  nodes,
  isEvaluating,
  showTree,
  setModalAnchorEl,
  setEditingCriteria,
  // Props for ContextAndGenerateCard
  isAddingCustom,
  userInput,
  setUserInput,
  generateChildNodes,
  selectedNode,
  isGenerating,
  isAnalysisSubmitted,
  setIsAddingCustom,
  handleAddCustomHypothesis,
  customHypothesis,
  setCustomHypothesis,
  setIsEditingSystemPrompt,
  editIcon,
  handleProceedWithSelectedIdea
}) => {
  const [showAfter, setShowAfter] = useState(true);
  const [activeSection, setActiveSection] = useState('Impact');

  const node = nodeId ? nodes.find(n => n.id === nodeId) : null;

  useEffect(() => {
    // When the node being viewed changes, reset the tab to Impact.
    setActiveSection('Impact');
  }, [nodeId]);

  if (!node) {
    return (
      <div
        style={{
          padding: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '160px',
          color: '#9ca3af',
        }}
      >
        <p>Click on a node to view details</p>
      </div>
    );
  }

  return (
    <div
      style={{
        border: '1px solid #E5E7EB',
        borderRadius: '8px',
        backgroundColor: '#FFFFFF',
        padding: '16px',
      }}
    >
      <HypothesisCard
        node={node}
        showAfter={showAfter}
        setShowAfter={setShowAfter}
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        onEditCriteria={(buttonRef, dimension) => {
          setModalAnchorEl(buttonRef);
          setEditingCriteria(dimension);
        }}
      />
      {showTree ? (
        <ContextAndGenerateCard
          isAddingCustom={isAddingCustom}
          userInput={userInput}
          setUserInput={setUserInput}
          generateChildNodes={generateChildNodes}
          selectedNode={selectedNode}
          isGenerating={isGenerating}
          isEvaluating={isEvaluating}
          isAnalysisSubmitted={isAnalysisSubmitted}
          setIsAddingCustom={setIsAddingCustom}
          handleAddCustomHypothesis={handleAddCustomHypothesis}
          customHypothesis={customHypothesis}
          setCustomHypothesis={setCustomHypothesis}
          setIsEditingSystemPrompt={setIsEditingSystemPrompt}
          setModalAnchorEl={setModalAnchorEl}
          editIcon={editIcon}
          handleProceedWithSelectedIdea={handleProceedWithSelectedIdea}
        />
      ) : null}
    </div>
  );
};


// ============== 段落2：定义 TreePlotVisualization 组件 ==============
const TreePlotVisualization = () => {
  // ... (all your existing state hooks remain here)
  const [currentView, setCurrentView] = useState('home_view'); // Start with home_view
  const [nodes, setNodes] = useState([]);
  const [links, setLinks] = useState([]);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [userInput, setUserInput] = useState('');
  const [operationStatus, setOperationStatus] = useState('');
  const [analysisIntent, setAnalysisIntent] = useState('');
  const [isAnalysisSubmitted, setIsAnalysisSubmitted] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [error, setError] = useState(null);
  const svgRef = useRef(null);
  const [hypothesesList, setHypothesesList] = useState([]);
  const [pendingChange, setPendingChange] = useState(null);
  const [pendingMerge, setPendingMerge] = useState(null);
  // *** 新增：用于放大被拖拽覆盖的目标节点 ***
  const [mergeTargetId, setMergeTargetId] = useState(null);
  const [isAddingCustom, setIsAddingCustom] = useState(false);
  const [customHypothesis, setCustomHypothesis] = useState({ title: '', content: '' });
  // *** 新增：用于主界面模型选择和api-key输入
  const [selectedModel, setSelectedModel] = useState('deepseek-chat');
  const [apiKey, setApiKey] = useState('');
  const [isConfigured, setIsConfigured] = useState(false);
  const [configError, setConfigError] = useState('');
  // *** 新增：用于用户自定义prompts
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isEditingSystemPrompt, setIsEditingSystemPrompt] = useState(false);

  const [impactCriteria, setImpactCriteria] = useState('');
  const [feasibilityCriteria, setFeasibilityCriteria] = useState('');
  const [noveltyCriteria, setNoveltyCriteria] = useState('');
  const [editingCriteria, setEditingCriteria] = useState(null); // 'impact' | 'feasibility' | 'novelty' | null
  const [defaultPrompts, setDefaultPrompts] = useState({
    system_prompt: '',
    novelty: '',
    feasibility: '',
    impact: '',
  });
  // Add refs for positioning modals
  const systemPromptButtonRef = useRef(null);
  const [modalAnchorEl, setModalAnchorEl] = useState(null);
  const [dashboardWidth, setDashboardWidth] = useState(0);
  const dashboardContainerRef = useRef(null);
  const analysisFormRef = useRef(null);
  const svgContainerRef = useRef(null);

  // New state for Code View and Paper View
  const [showProceedConfirm, setShowProceedConfirm] = useState(false);
  const [isGeneratingCode, setIsGeneratingCode] = useState(false);
  const [isGeneratingPaper, setIsGeneratingPaper] = useState(false);
  const [codeResult, setCodeResult] = useState(null);
  const [paperResult, setPaperResult] = useState(null);
  const [proceedError, setProceedError] = useState(null);
  const [codeContent, setCodeContent] = useState('');
  const [codeFileName, setCodeFileName] = useState('experiment.py');
  const [activeCodeTab, setActiveCodeTab] = useState('experiment.py');
  const [experimentFiles, setExperimentFiles] = useState({});
  const [consoleOutput, setConsoleOutput] = useState('');
  const [experimentRuns, setExperimentRuns] = useState([]);
  const [isRunningExperiment, setIsRunningExperiment] = useState(false);
  const [pdfComments, setPdfComments] = useState([]);
  const [hasGeneratedCode, setHasGeneratedCode] = useState(false);
  const [newComment, setNewComment] = useState('');
  const [s2ApiKey, setS2ApiKey] = useState('');

  // Removed tracking hooks

  // Track view changes
  const previousViewRef = useRef(currentView); // Initialize with current view
  const isInitialRender = useRef(true);

  useEffect(() => {
    // Skip tracking on initial render
    if (isInitialRender.current) {
      isInitialRender.current = false;
      return;
    }

    // Only track actual view changes (not initial render)
    if (previousViewRef.current !== currentView) {
      previousViewRef.current = currentView;
    }
  }, [currentView]);

  // This effect will measure the dashboard container and update the width state
  useEffect(() => {
    if (dashboardContainerRef.current) {
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          setDashboardWidth(entry.contentRect.width);
        }
      });
      resizeObserver.observe(dashboardContainerRef.current);
      return () => resizeObserver.disconnect();
    }
  }, []); // Run only once on mount
  // X/Y 轴指标（仅在 Evaluation View 用）
  const [xAxisMetric, setXAxisMetric] = useState('feasibilityScore');
  const [yAxisMetric, setYAxisMetric] = useState('noveltyScore');

  // ID 计数器
  const idCounterRef = useRef(0);
  const generateUniqueId = () => {
    idCounterRef.current += 1;
    return `node-${idCounterRef.current}`;
  };

  // 配色映射
  const colorMap = {
    root: '#4C84FF',
    simple: '#45B649',
    complex: '#FF6B6B',
  };
  // ============== 配置模型和API Key ==============
  const modelOptions = [
    { value: 'deepseek-chat', label: 'DeepSeek Chat' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner' },
    { value: 'gpt-4o', label: 'GPT-4o' },
    { value: 'o1-2024-12-17', label: 'GPT-o1' },
    { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
  ];
  useEffect(() => {
    const fetchPrompts = async () => {
      if (isConfigured) {
        try {
          const response = await fetch('/api/get-prompts', { credentials: 'include' });
          if (!response.ok) {
            throw new Error('Failed to fetch prompts'); // Corrected spelling
          }
          const data = await response.json();
          // Set the current prompts
          setSystemPrompt(data.system_prompt);
          setImpactCriteria(data.criteria.impact);
          setFeasibilityCriteria(data.criteria.feasibility);
          setNoveltyCriteria(data.criteria.novelty);
          // Set the default prompts
          setDefaultPrompts(data.defaults);
        } catch (err) {
          console.error('Error fetching prompts:', err);
          setError(err.message);
        }
      }
    };

    fetchPrompts();
  }, [isConfigured]);
  const handleConfigSubmit = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) {
      setConfigError('Please enter an API key');
      return;
    }


    setConfigError('');
    setOperationStatus('Configuring model...');

    try {
      const response = await fetch('/api/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          model: selectedModel,
          api_key: apiKey,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to configure model');
      }

      setIsConfigured(true);
      setOperationStatus('');
      setCurrentView('exploration'); // Auto-switch to exploration view
    } catch (err) {
      console.error('Configuration error:', err);
      setConfigError(err.message);
      setOperationStatus('');
    }
  };

  // Overview Component
  const OverviewPage = () => (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 'calc(100vh - 60px)', // Adjust for nav height
        backgroundColor: '#f9fafb',
      }}
    >
      <div
        style={{
          backgroundColor: '#fff',
          padding: '48px',
          borderRadius: '12px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          width: '100%',
          maxWidth: '500px',
        }}
      >
        <h2
          style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            marginBottom: '32px',
            textAlign: 'center',
            color: '#111827',
          }}
        >
          Model Configuration
        </h2>

        <form onSubmit={handleConfigSubmit}>
          {/* Model Selection */}
          <div style={{ marginBottom: '24px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '0.875rem',
                fontWeight: 500,
                color: '#374151',
              }}
            >
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
              }}
              style={{
                width: '100%',
                padding: '10px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '0.875rem',
                backgroundColor: '#fff',
                cursor: 'pointer',
              }}
            >
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* API Key Input */}
          <div style={{ marginBottom: '24px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '0.875rem',
                fontWeight: 500,
                color: '#374151',
              }}
            >
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              style={{
                width: '100%',
                padding: '10px 12px',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '0.875rem',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Error Message */}
          {configError && (
            <div
              style={{
                marginBottom: '16px',
                padding: '12px',
                backgroundColor: '#fee2e2',
                border: '1px solid #fecaca',
                borderRadius: '6px',
                color: '#dc2626',
                fontSize: '0.875rem',
              }}
            >
              {configError}
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={operationStatus === 'Configuring model...'}
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: '#4C84FF',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: operationStatus === 'Configuring model...' ? 'not-allowed' : 'pointer',
              opacity: operationStatus === 'Configuring model...' ? 0.7 : 1,
            }}
          >
            {operationStatus === 'Configuring model...' ? 'Configuring...' : 'Start Session'}
          </button>

          {/* Status for already configured */}
          {isConfigured && (
            <div
              style={{
                marginTop: '16px',
                textAlign: 'center',
                color: '#059669',
                fontSize: '0.875rem',
              }}
            >
              ✓ Model configured successfully
            </div>
          )}
        </form>
      </div>
    </div>
  );

  // ============== 自定义prompts==============
  const updateSystemPrompt = async (prompt) => {
    try {
      const response = await fetch('/api/set-system-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ system_prompt: prompt }),
      });
      if (!response.ok) throw new Error('Failed to update system prompt');
      return true;
    } catch (err) {
      console.error('Error updating system prompt:', err);
      setError(err.message);
      return false;
    }
  };

  const updateCriteria = async (dimension, criteria) => {
    try {
      const response = await fetch('/api/set-criteria', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ dimension, criteria }),
      });
      if (!response.ok) throw new Error('Failed to update criteria');
      return true;
    } catch (err) {
      console.error('Error updating criteria:', err);
      setError(err.message);
      return false;
    }
  };
  // Modal component for editing prompts
  const EditModal = ({ isOpen, onClose, title, defaultText, initialValue, onSave, height = '270px', anchorRef, width }) => {
    // Manages its own state while open, initialized by the parent's value
    const [currentValue, setCurrentValue] = useState(initialValue || '');
    const [position, setPosition] = useState({ top: 0, left: 0, transform: undefined });

    useEffect(() => {
      // This ensures if the modal is re-opened for a different item, it shows the new value
      setCurrentValue(initialValue || '');
    }, [initialValue]);

    useEffect(() => {
      // Calculate position only when modal opens or anchor changes
      if (isOpen && anchorRef?.current) {
        const anchorElement = anchorRef.current;
        const anchorRect = anchorElement.getBoundingClientRect();
        const modalWidth = width || 510;
        const viewportWidth = window.innerWidth;
        const screenEdgeMargin = 16;

        // Calculate the ideal centered 'left' position to be exactly below the icon
        let left = anchorRect.left + (anchorRect.width / 2) - (modalWidth / 2);

        // Adjust if it overflows the right edge of the viewport
        if (left + modalWidth > viewportWidth - screenEdgeMargin) {
          left = viewportWidth - modalWidth - screenEdgeMargin;
        }

        if (left < screenEdgeMargin) {
          left = screenEdgeMargin;
        }

        setPosition({
          top: anchorRect.bottom + 8,
          left: left,
          transform: undefined
        });
      } else if (isOpen) {
        // Fallback position if no anchor
        setPosition({ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' });
      }
    }, [isOpen, anchorRef, width]);

    if (!isOpen) return null;

    // The handler for the save button. It now passes the modal's current text back to the parent.
    const handleSave = () => {
      onSave(currentValue);
    };

    return (
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 1000,
        }}
        onClick={onClose}
      >
        <div
          style={{
            position: 'absolute',
            top: position.top,
            left: position.left,
            transform: position.transform,
            backgroundColor: '#fff',
            borderRadius: '8px',
            padding: '24px',
            width: width ? `${width}px` : '510px',
            maxWidth: '90vw',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            border: '1px solid rgba(0, 0, 0, 0.05)',
            display: 'flex',
            flexDirection: 'column',
            height: height,
            boxSizing: 'border-box',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <h3 style={{ marginTop: 0, fontSize: '1.125rem', fontWeight: 600, color: '#111827', flexShrink: 0 }}>
            {title}
          </h3>

          <textarea
            value={currentValue}
            // This now only updates the modal's internal state, preventing the parent from re-rendering
            onChange={(e) => setCurrentValue(e.target.value)}
            placeholder={defaultText}
            style={{
              flexGrow: 1,
              minHeight: '40px',
              marginTop: '8px',
              marginBottom: '16px',
              width: '100%',
              padding: '12px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '0.875rem',
              resize: 'none',
              fontFamily: 'inherit',
              lineHeight: '1.5',
              boxSizing: 'border-box',
            }}
          />

          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', flexShrink: 0 }}>
            <button
              onClick={() => {
                setCurrentValue('');
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              Reset
            </button>
            <button
              onClick={() => {
                onClose();
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              style={{
                padding: '8px 16px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              Save
            </button>
          </div>
        </div>
      </div>
    );
  };
  // Add this with other SVG icon definitions
  const editIcon = (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V13"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18.5 2.50001C18.8978 2.10219 19.4374 1.87869 20 1.87869C20.5626 1.87869 21.1022 2.10219 21.5 2.50001C21.8978 2.89784 22.1213 3.4374 22.1213 4.00001C22.1213 4.56262 21.8978 5.10219 21.5 5.50001L12 15L8 16L9 12L18.5 2.50001Z"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
  // ============== 段落3：评估假设（evaluateHypotheses） ==============
  const evaluateHypotheses = async (hypotheses) => {

    setIsEvaluating(true);
    setOperationStatus('Evaluating hypotheses...');
    setError(null);

    try {
      const requestBody = {
        ideas: hypotheses.map(h => ({
          ...(h.originalData || {}),
          id: h.id,  // Always include the frontend node ID
          title: h.title,
          content: h.content
        })),
        intent: analysisIntent
      };

      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to evaluate ideas');
      }

      const evaluatedHypotheses = await response.json();

      setNodes((prevNodes) => {

        const updatedNodes = prevNodes.map((node) => {
          const evalHypo = evaluatedHypotheses.find((h) => h.id === node.id);
          if (evalHypo) {
            const noveltyScore = evalHypo.noveltyScore;
            const feasibilityScore = evalHypo.feasibilityScore;
            const impactScore = evalHypo.impactScore;

            return {
              ...node,
              noveltyScore,
              feasibilityScore,
              impactScore,
              noveltyReason: evalHypo.noveltyReason || '(No reason provided)',
              feasibilityReason: evalHypo.feasibilityReason || '(No reason provided)',
              impactReason: evalHypo.impactReason || '(No reason provided)',
            };
          }
          return node;
        });

        return updatedNodes;
      });
    } catch (err) {
      console.error('Error evaluating hypotheses:', err);
      setError(err.message);
    } finally {
      setIsEvaluating(false);
      setOperationStatus('');
    }
  };

  // ============== 段落4：分析意图提交处理 (handleAnalysisIntentSubmit) ==============
  const handleAnalysisIntentSubmit = async (e) => {
    e.preventDefault();
    if (!analysisIntent.trim()) return;


    if (!isConfigured) {
      setError('Please configure the model first');
      return;
    }

    setIsGenerating(true);
    setOperationStatus('Generating initial hypotheses...');
    setError(null);

    try {
      // Call Flask backend
      const response = await fetch('/api/generate-initial', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          intent: analysisIntent,
          num_ideas: 3
        }),
      });
      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        throw new Error(`Failed to generate ideas: ${errorText}`);
      }

      const data = await response.json();
      console.log("Received data from API:", data);

      const ideas = data.ideas;

      // Create IDs using the existing method
      const ideasWithId = ideas.map((idea) => {
        const id = generateUniqueId();
        return { id, ...idea };
      });

      const updatedHypothesesList = [...hypothesesList, ...ideasWithId];
      setHypothesesList(updatedHypothesesList);

      // Root node
      const rootNode = {
        id: 'root',
        level: 0,
        title: analysisIntent,
        content: analysisIntent,
        type: 'root',
        x: 0,
        y: 0,
      };

      // Child nodes
      const childSpacing = 200;
      const totalWidth = (ideas.length - 1) * childSpacing;
      const startX = rootNode.x - totalWidth / 2;

      const childNodes = ideasWithId.map((hyp, i) => ({
        id: hyp.id,
        level: 1,
        title: hyp.title.trim(),
        content: hyp.content.trim(),
        type: 'complex',
        x: startX + i * childSpacing,
        y: rootNode.y + 150,
        originalData: hyp.originalData, // Preserve original data for coder/writer
      }));

      const newNodes = [rootNode, ...childNodes];
      const newLinks = childNodes.map((nd) => ({ source: rootNode.id, target: nd.id }));
      setNodes(newNodes);
      setLinks(newLinks);

      setAnalysisIntent('');
      setIsAnalysisSubmitted(true);

      // 评估
      await evaluateHypotheses(updatedHypothesesList);
      setIsGenerating(false);
      setOperationStatus('');
    } catch (err) {
      console.error('Error generating initial hypotheses:', err);
      setError(err.message);
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // ============== 段落5：生成子节点 (generateChildNodes) ==============
  const generateChildNodes = async () => {
    if (!selectedNode) return;
    setIsGenerating(true);
    setOperationStatus('Generating child ideas...');
    setError(null);

    try {
      const response = await fetch('/api/generate-children', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          parent_content: selectedNode.content,
          context: userInput
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate child ideas');
      }

      const data = await response.json();
      const ideas = data.ideas;

      const newHypothesesWithId = ideas.map((hyp) => {
        const id = generateUniqueId();
        return { id, ...hyp };
      });

      const updatedHypothesesList = [...hypothesesList, ...newHypothesesWithId];
      setHypothesesList(updatedHypothesesList);

      // 布局
      const childSpacing = 200;
      const totalWidth = (ideas.length - 1) * childSpacing;
      const startX = selectedNode.x - totalWidth / 2;

      const newNodes = newHypothesesWithId.map((hyp, i) => ({
        id: hyp.id,
        level: selectedNode.level + 1,
        title: hyp.title.trim(),
        content: hyp.content.trim(),
        type: 'complex',
        x: startX + i * childSpacing + Math.random() * 20 - 10,
        y: selectedNode.y + 150 + Math.random() * 20 - 10,
        originalData: hyp.originalData, // Preserve original data for coder/writer
      }));

      const newLinks = newNodes.map((nd) => ({ source: selectedNode.id, target: nd.id }));
      setNodes((prev) => [...prev, ...newNodes]);
      setLinks((prev) => [...prev, ...newLinks]);

      // 评估
      await evaluateHypotheses(updatedHypothesesList);

      setIsGenerating(false);
      setOperationStatus('');
      setUserInput('');
    } catch (err) {
      console.error('Error generating hypotheses:', err);
      setError(err.message);
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // ============== 段落6：根据拖拽修改假设 (modifyHypothesisBasedOnModifications) ==============
  const modifyHypothesisBasedOnModifications = async (
    originalNode,
    ghostNode,
    modifications,
    behindNode
  ) => {
    setError(null);

    try {
      setIsGenerating(true);
      setOperationStatus('Modifying hypothesis...');
      const response = await fetch('/api/modify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          original_idea: originalNode.originalData || {
            id: originalNode.id,
            title: originalNode.title,
            content: originalNode.content
          },
          modifications: modifications,
          behind_idea: behindNode ? (behindNode.originalData || {
            id: behindNode.id,
            title: behindNode.title,
            content: behindNode.content
          }) : null
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to modify idea');
      }

      const data = await response.json();

      ghostNode.content = data.content;
      ghostNode.title = data.title;
      ghostNode.isModified = true;
      ghostNode.previousState = originalNode;
      ghostNode.isGhost = false;
      ghostNode.originalData = data.originalData; // Preserve original data for coder/writer

      const newHypothesis = {
        id: ghostNode.id,
        title: ghostNode.title,
        content: data.content,
      };
      setHypothesesList((prevList) => [...prevList, newHypothesis]);

      setNodes((prevNodes) => prevNodes.map((n) => (n.id === ghostNode.id ? ghostNode : n)));
      setIsGenerating(false);
      setIsEvaluating(true);

      await evaluateHypotheses([...hypothesesList, newHypothesis]);
    } catch (err) {
      console.error('Error modifying hypothesis:', err);
      setError(err.message);
      setIsGenerating(false);
      setOperationStatus('');
    } finally {
      setIsGenerating(false);
      setIsEvaluating(false);
      setOperationStatus('');
    }
  };

  const mergeHypotheses = async (nodeA, nodeB) => {
    /* ------- ① 先打动画标记：后节点放大 ------- */
    setNodes((prev) =>
      prev.map((n) => {
        if (n.id === nodeA.id) {
          // Make nodeA disappear
          return { ...n, evaluationOpacity: 0 };
        } else if (n.id === nodeB.id) {
          // Make nodeB bigger
          return { ...n, isBeingMerged: true };
        }
        return n;
      })
    );

    setIsGenerating(true);
    setOperationStatus('Merging hypotheses...');
    setError(null);
    try {
      /* ---------- ② 调用 Flask API ---------- */
      const response = await fetch('/api/merge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          idea_a: nodeA.originalData || {
            id: nodeA.id,
            title: nodeA.title,
            content: nodeA.content
          },
          idea_b: nodeB.originalData || {
            id: nodeB.id,
            title: nodeB.title,
            content: nodeB.content
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to merge ideas');
      }

      const data = await response.json();

      /* ---------- ③ 生成新节点（深红）并连线 ---------- */
      const newId = generateUniqueId();
      const newHypothesis = {
        id: newId,
        title: data.title,
        content: data.content
      };
      const newNode = {
        id: newId,
        level: Math.min(nodeA.level, nodeB.level) + 1,
        title: data.title.trim(),
        content: data.content.trim(),
        type: 'complex',
        x: (nodeA.x + nodeB.x) / 2,
        y: (nodeA.y + nodeB.y) / 2,
        isMergedResult: true,
        originalData: data.originalData, // Preserve original data for coder/writer
      };

      setHypothesesList((p) => [...p, newHypothesis]);
      // Animation Step 2: Make nodeA reappear and nodeB normal size again
      setNodes((prev) => [
        ...prev.map((n) => {
          if (n.id === nodeA.id) {
            // Make nodeA reappear
            return { ...n, evaluationOpacity: 1 };
          } else if (n.id === nodeB.id) {
            // Reset nodeB to normal size
            return { ...n, isBeingMerged: false };
          }
          return n;
        }),
        newNode, // Add the new merged node
      ]);
      setLinks((prev) => [
        ...prev,
        { source: newId, target: nodeA.id },
        { source: newId, target: nodeB.id },
      ]);

      setMergeTargetId(null);
      setIsGenerating(false);
      setOperationStatus('');
      setIsEvaluating(true);
      await evaluateHypotheses([...hypothesesList, newHypothesis]);
    } catch (err) {
      console.error('[merge] Error merging hypotheses:', err);
      setError(err.message);
    } finally {
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // ============== 段落7：节点点击事件处理 ==============
  // Note: Hover events are now handled by unified tracker to prevent duplicates

  const zoomTransformRef = useRef(null);

  // Cleanup is handled by D3's selectAll('*').remove() in the main useEffect

  // ============== 段落8：D3 渲染逻辑 useEffect ==============
  useEffect(() => {
    if (!svgRef.current || currentView === 'home_view') return;

    const width = 800;
    const height = 600;

    // 清空 SVG
    d3.select(svgRef.current).selectAll('*').remove();

    if (currentView === 'exploration') {
      /* ------------------- Exploration (Tree) View ------------------- */
      const svg = d3.select(svgRef.current).attr('width', width).attr('height', height);

      /* 背景卡片 */
      svg
        .append('rect')
        .attr('x', 2)
        .attr('y', 2)
        .attr('width', width - 4)
        .attr('height', height - 4)
        .attr('rx', 8)
        .attr('ry', 8)
        .style('fill', '#fff')
        .style('stroke', '#e5e7eb')
        .style('stroke-width', 1)
        .style('filter', 'drop-shadow(0 4px 6px rgba(0,0,0,0.1))');

      // Create a group for zooming and panning
      const zoomGroup = svg.append('g').attr('class', 'zoom-group');

      // Initial transform to center the tree
      const initialTransform = d3.zoomIdentity.translate(width / 2, 100);

      // Create zoom behavior
      const zoom = d3.zoom()
        .scaleExtent([0.3, 3]) // Min and max zoom levels
        .on('zoom', (event) => {
          zoomGroup.attr('transform', event.transform);
          zoomTransformRef.current = event.transform;
        });

      // Apply zoom behavior to SVG
      svg.call(zoom);

      // Set initial transform
      const transformToUse = zoomTransformRef.current || initialTransform;
      svg.call(zoom.transform, transformToUse);

      // Add invisible rect for better drag interaction
      svg
        .append('rect')
        .attr('class', 'zoom-rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', width)
        .attr('height', height)
        .style('fill', 'none')
        .style('pointer-events', 'all')
        .lower(); // Put it behind everything

      // The main group for tree content (was 'g', now 'zoomGroup')
      const g = zoomGroup.append('g');

      // New improved layout logic
      const layoutNodes = [...nodes];
      const nodesByLevel = {};

      // Group nodes by level
      layoutNodes.forEach(node => {
        if (!nodesByLevel[node.level]) {
          nodesByLevel[node.level] = [];
        }
        nodesByLevel[node.level].push(node);
      });

      // Process each level
      Object.keys(nodesByLevel).forEach(levelStr => {
        const level = parseInt(levelStr);
        const nodesAtLevel = nodesByLevel[level];

        if (level === 0) {
          // Root node stays at center
          nodesAtLevel[0].x = 0;
          nodesAtLevel[0].y = 0;
          return;
        }

        // Sort nodes by parent order
        nodesAtLevel.sort((a, b) => {
          // Get parent X positions for sorting
          const getParentX = (node) => {
            if (node.isMergedResult) {
              // For merged nodes, find average of parent positions
              const parentLinks = links.filter(link => link.source === node.id);
              const parents = parentLinks.map(link =>
                layoutNodes.find(n => n.id === link.target)
              ).filter(p => p);

              if (parents.length > 0) {
                return parents.reduce((sum, p) => sum + p.x, 0) / parents.length;
              }
            } else {
              // For regular nodes, find single parent
              const parentLink = links.find(link => link.target === node.id);
              if (parentLink) {
                const parent = layoutNodes.find(n => n.id === parentLink.source);
                if (parent) return parent.x;
              }
            }
            return 0;
          };

          const aParentX = getParentX(a);
          const bParentX = getParentX(b);

          // If same parent X, sort by node ID for consistency
          if (aParentX === bParentX) {
            // First, prioritize non-modified nodes
            const aIsModified = a.isModified || a.previousState;
            const bIsModified = b.isModified || b.previousState;

            if (aIsModified !== bIsModified) {
              return aIsModified ? 1 : -1;
            }

            return a.id.localeCompare(b.id);
          }

          return aParentX - bParentX;
        });

        // Apply even spacing
        const nodeSpacing = 200; // Consistent spacing between all nodes
        const totalWidth = (nodesAtLevel.length - 1) * nodeSpacing;
        const startX = -totalWidth / 2;

        nodesAtLevel.forEach((node, index) => {
          node.x = startX + index * nodeSpacing;
          node.y = level * 150;
        });
      });

      /* 连线 */
      links.forEach((lk) => {
        const s = layoutNodes.find((n) => n.id === lk.source);
        const t = layoutNodes.find((n) => n.id === lk.target);
        if (!s || !t) return;
        g.append('path')
          .attr(
            'd',
            `M${s.x},${s.y} C${s.x},${(s.y + t.y) / 2} ${t.x},${(s.y + t.y) / 2} ${t.x},${t.y}`
          )
          .style('fill', 'none')
          .style('stroke', '#ccc')
          .style('stroke-width', 2);
      });

      /* 节点 */
      const nodeG = g
        .selectAll('.node')
        .data(layoutNodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', (d) => `translate(${d.x},${d.y})`);

      // Add event listeners with unique namespaces to prevent duplicates
      nodeG
        .on('mouseenter.exploration-tracking', function (event, d) {
          // Don't track hover if any node is being dragged or events are suppressed after drag
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag) {
          }
          // UI functionality
          setHoveredNode(d);
          d3.select(this).raise();
        })
        .on('mouseleave.exploration-tracking', function (event, d) {
          // Don't track hover if any node is being dragged or events are suppressed after drag
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag) {
          }
          // UI functionality
          setHoveredNode(null);
        })
        .on('click.tracking', (_, d) => {
          // Don't track click if events are suppressed after drag
          if (!d._suppressEventsAfterDrag) {
          }
          // UI functionality
          setSelectedNode(d);
        });

      nodeG
        .append('circle')
        .attr('r', 25)
        .style('fill', (d) => {
          if (d.isMergedResult) return '#B22222';
          if (d.isNewlyGenerated) return '#FFD700';
          return colorMap[d.type] || '#FF6B6B';
        })
        .style('opacity', (d) => {
          if (d.opacity !== undefined) return d.opacity;
          return d.isGhost ? 0.5 : 1;
        })
        .style('stroke', (d) =>
          selectedNode?.id === d.id ? '#000' : hoveredNode?.id === d.id ? '#555' : '#fff'
        )
        .style('stroke-width', (d) =>
          selectedNode?.id === d.id ? 4 : hoveredNode?.id === d.id ? 3 : 2
        )
        .style('cursor', 'pointer');

      // Add text label with a styled rectangular background
      const label = nodeG.append('g')
        .attr('class', 'label-group')
        .style('pointer-events', 'none'); // The label should not capture mouse events

      // Add text first to measure it
      const text = label.append('text')
        .attr('y', -40) // Position above the circle node (radius is 25)
        .attr('text-anchor', 'middle')
        .style('font-family', 'Arial, sans-serif')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('fill', '#1f2937') // A dark gray for good contrast
        .text(d => d.title);

      // Add a backing rectangle for each text element
      label.each(function (d) {
        const textNode = d3.select(this).select('text').node();
        if (textNode) {
          const bbox = textNode.getBBox();
          const padding = { x: 8, y: 4 };

          d3.select(this).insert('rect', 'text') // Insert rect before text
            .attr('x', bbox.x - padding.x)
            .attr('y', bbox.y - padding.y)
            .attr('width', bbox.width + padding.x * 2)
            .attr('height', bbox.height + padding.y * 2)
            .attr('rx', 4) // Rounded corners
            .attr('ry', 4)
            .style('fill', '#f3f4f6') // A light, neutral background
            .style('stroke', '#d1d5db') // A soft border color
            .style('stroke-width', 1.5);
        }
      });


      // Add zoom controls (vertical layout)
      const controls = svg.append('g')
        .attr('class', 'zoom-controls')
        .attr('transform', `translate(${width - 40}, 20)`);

      // Zoom in button
      controls.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('cursor', 'pointer')
        .on('click', () => {
          svg.transition().call(zoom.scaleBy, 1.3);
        });

      controls.append('text')
        .attr('x', 14)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .text('+');

      // Zoom out button
      controls.append('rect')
        .attr('x', 0)
        .attr('y', 32)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('cursor', 'pointer')
        .on('click', () => {
          svg.transition().call(zoom.scaleBy, 0.7);
        });

      controls.append('text')
        .attr('x', 14)
        .attr('y', 48)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .text('−');

      // Reset zoom button
      controls.append('rect')
        .attr('x', 0)
        .attr('y', 64)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('cursor', 'pointer')
        .on('click', () => {
          svg.transition().call(zoom.transform, initialTransform);
          zoomTransformRef.current = initialTransform;
        });

      controls.append('text')
        .attr('x', 13)
        .attr('y', 78)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .text('⟲');

      if (operationStatus && operationStatus.toLowerCase().includes('generating')) {
        svg
          .append('rect')
          .attr('x', 2)
          .attr('y', 2)
          .attr('width', width - 4)
          .attr('height', height - 4)
          .attr('rx', 8)
          .attr('ry', 8)
          .style('fill', 'rgba(255, 255, 255, 0.7)')
          .style('pointer-events', 'none');
        svg
          .append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-size', '1.2rem')
          .style('font-weight', '600')
          .style('fill', '#374151')
          .text(operationStatus);
      }
    } else {
      /* ------------------- Evaluation (Scatter) View ------------------- */
      const svg = d3.select(svgRef.current).attr('width', width).attr('height', height);

      /* 背景卡片 */
      svg
        .append('rect')
        .attr('x', 2)
        .attr('y', 2)
        .attr('width', width - 4)
        .attr('height', height - 4)
        .attr('rx', 8)
        .attr('ry', 8)
        .style('fill', '#fff')
        .style('stroke', '#e5e7eb')
        .style('stroke-width', 1)
        .style('filter', 'drop-shadow(0 4px 6px rgba(0,0,0,0.1))');

      /* 坐标轴下拉 */
      const axisFO = svg
        .append('foreignObject')
        .attr('x', width - 210)
        .attr('y', 20)
        .attr('width', 200)
        .attr('height', 60);

      axisFO
        .append('xhtml:div')
        .style('display', 'flex')
        .style('flexDirection', 'column')
        .style('gap', '6px')
        .html(`
          <div>
            <div style="display:flex;align-items:center;">
              <label style="margin-right:6px;font-size:0.8rem;color:#374151;">X‑Axis:</label>
              <select id="xSelect" style="padding:4px;border:1px solid #d1d5db;border-radius:4px;">
                <option value="noveltyScore">Novelty</option>
                <option value="feasibilityScore">Feasibility</option>
                <option value="impactScore">Impact</option>
              </select>
            </div>
            <div style="display:flex;align-items:center;margin-top:6px;">
              <label style="margin-right:6px;font-size:0.8rem;color:#374151;">Y‑Axis:</label>
              <select id="ySelect" style="padding:4px;border:1px solid #d1d5db;border-radius:4px;">
                <option value="noveltyScore">Novelty</option>
                <option value="feasibilityScore">Feasibility</option>
                <option value="impactScore">Impact</option>
              </select>
            </div>
          </div>`);

      const xSel = document.getElementById('xSelect');
      const ySel = document.getElementById('ySelect');
      if (xSel) xSel.value = xAxisMetric;
      if (ySel) ySel.value = yAxisMetric;
      if (xSel) xSel.onchange = (e) => {
        setXAxisMetric(e.target.value);
      };
      if (ySel) ySel.onchange = (e) => {
        setYAxisMetric(e.target.value);
      };

      /* 画布与比例尺 */
      const margin = { top: 100, right: 50, bottom: 80, left: 50 };
      const chartW = width - margin.left - margin.right;
      const chartH = height - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      const x = d3.scaleLinear().domain([0, 100]).range([0, chartW]);
      const y = d3.scaleLinear().domain([0, 100]).range([chartH, 0]);

      /* 网格 & 轴 */
      const xGrid = d3.axisBottom(x).tickValues(d3.range(0, 101, 20)).tickSize(-chartH).tickFormat('');
      const yGrid = d3.axisLeft(y).tickValues(d3.range(0, 101, 20)).tickSize(-chartW).tickFormat('');
      g.append('g').attr('class', 'x-grid').attr('transform', `translate(0,${chartH})`).call(xGrid).selectAll('line').style('stroke', '#F1F1F1');
      g.append('g').attr('class', 'y-grid').call(yGrid).selectAll('line').style('stroke', '#F1F1F1');
      g.append('g').attr('transform', `translate(0,${chartH})`).call(d3.axisBottom(x));
      g.append('g').call(d3.axisLeft(y));

      /* 轴标签 */
      g.append('text')
        .attr('x', chartW / 2)
        .attr('y', chartH + 40)
        .style('text-anchor', 'middle')
        .style('fill', '#374151')
        .style('font-size', '1.2rem')
        .text(xAxisMetric.replace('Score', ''));

      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -chartH / 2)
        .attr('y', -30)
        .style('text-anchor', 'middle')
        .style('fill', '#374151')
        .style('font-size', '1.2rem')
        .text(yAxisMetric.replace('Score', ''));
      if (!(operationStatus === 'Evaluating hypotheses...' && isEvaluating)) {


        /* 过滤能绘制的节点 */
        const drawable = nodes.filter(
          (n) => n[xAxisMetric] !== undefined && n[yAxisMetric] !== undefined
        );

        /* ---------- 连线：evaluation + merge ---------- */
        const hasCoords = (n) =>
          n[xAxisMetric] !== undefined && n[yAxisMetric] !== undefined;

        /* 两类连线
        ① original → modified   （浅灰 #ccc, 1px）
        ② mergedResult ↔ originals（深灰 #999, 1.5px） */
        links.forEach((lk) => {
          const s = nodes.find((n) => n.id === lk.source);
          const t = nodes.find((n) => n.id === lk.target);
          if (!s || !t) return;
          if (!hasCoords(s) || !hasCoords(t)) return; // 任一端缺坐标则跳过

          const isMergeEdge = s.isMergedResult || t.isMergedResult;
          g.append('line')
            .attr('x1', x(s[xAxisMetric]))
            .attr('y1', y(s[yAxisMetric]))
            .attr('x2', x(t[xAxisMetric]))
            .attr('y2', y(t[yAxisMetric]))
            .style('stroke', isMergeEdge ? '#999' : '#ccc')
            .style('stroke-width', isMergeEdge ? 1.5 : 1);
        });

        /* 节点绘制 & 拖拽 */
        const nodeG = g
          .selectAll('.node-group')
          .data(drawable, (d) => d.id)
          .enter()
          .append('g')
          .attr('class', 'node-group')
          .attr('transform', (d) => `translate(${x(d[xAxisMetric])},${y(d[yAxisMetric])})`)
          .style('cursor', 'pointer');

        // Add event listeners with unique namespaces to prevent duplicates
        nodeG
          .on('mouseenter.eval-tracking', (e, d) => {
            // Don't track hover if any node is being dragged or events are suppressed after drag
            if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag) {
            }
            // UI functionality
            setHoveredNode(d);
          })
          .on('mouseleave.eval-tracking', (e, d) => {
            // Don't track hover if any node is being dragged or events are suppressed after drag
            if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag) {
            }
            // UI functionality
            setHoveredNode(null);
          })
          .on('click.eval-tracking', (e, d) => {
            // Don't track click if events are suppressed after drag
            if (!d._suppressEventsAfterDrag) {
            }
            // UI functionality
            setSelectedNode(d);
          })
          .call(
            d3
              .drag()
              .on('start', function (event, d) {
                if (isGenerating || isEvaluating) return;
                // Mark drag start but don't track yet - wait for actual movement
                d._dragStarted = false;
                d._dragStartValues = {
                  x: d[xAxisMetric],
                  y: d[yAxisMetric],
                  xAxisMetric: xAxisMetric,
                  yAxisMetric: yAxisMetric
                };
                d3.select(this).raise();
              })
              .on('drag', function (event, d) {
                if (isGenerating || isEvaluating) return;

                // Track drag_start only when actual dragging begins
                if (!d._dragStarted) {
                  d._dragStarted = true;
                }

                const [cx, cy] = d3.pointer(event, g.node());
                d3.select(this).attr('transform', `translate(${cx},${cy})`);
                d._tmpX = cx;
                d._tmpY = cy;
              })
              .on('end', function (event, d) {
                if (isGenerating || isEvaluating) return;
                const endX = d._tmpX ?? d3.pointer(event, g.node())[0];
                const endY = d._tmpY ?? d3.pointer(event, g.node())[1];

                // Only track drag_end if dragging actually occurred
                if (d._dragStarted) {
                  // Convert screen coordinates back to metric values
                  const endXValue = x.invert(endX);
                  const endYValue = y.invert(endY);


                  // Set flag to suppress automatic events after drag
                  d._suppressEventsAfterDrag = true;
                  // Clear the flag after a short delay to allow manual interactions
                  setTimeout(() => {
                    delete d._suppressEventsAfterDrag;
                  }, 100);
                }

                // Clean up temporary variables
                delete d._tmpX;
                delete d._tmpY;
                delete d._dragStarted;
                delete d._dragStartValues;

                /* ---------- ① 重叠检测：触发合并 ---------- */
                const overlap = drawable.find(
                  (n) =>
                    n.id !== d.id &&
                    Math.hypot(x(n[xAxisMetric]) - endX, y(n[yAxisMetric]) - endY) < 10
                );
                if (overlap) {
                  setMergeTargetId(overlap.id);
                  setPendingMerge({
                    nodeA: d,
                    nodeB: overlap,
                    screenX: event.sourceEvent.clientX + 10,
                    screenY: event.sourceEvent.clientY + 10,
                  });

                  // 回到原位
                  d3.select(this).attr(
                    'transform',
                    `translate(${x(d[xAxisMetric])},${y(d[yAxisMetric])})`
                  );
                  return;
                }

                /* ---------- ② 原有拖拽修改逻辑 ---------- */
                const newXVal = x.invert(endX);
                const newYVal = y.invert(endY);
                const deltaX = newXVal - d[xAxisMetric];
                const deltaY = newYVal - d[yAxisMetric];

                let mods = [];
                let behind = null;
                if (Math.abs(deltaX) > 5) {
                  mods.push({ metric: xAxisMetric, direction: deltaX > 0 ? 'increase' : 'decrease' });
                  behind = nodes
                    .filter(
                      (n) => n.id !== d.id && n[xAxisMetric] !== undefined && n[xAxisMetric] < newXVal
                    )
                    .sort((a, b) => b[xAxisMetric] - a[xAxisMetric])[0];
                }
                if (Math.abs(deltaY) > 5) {
                  mods.push({ metric: yAxisMetric, direction: deltaY > 0 ? 'increase' : 'decrease' });
                  behind =
                    behind ||
                    nodes
                      .filter(
                        (n) =>
                          n.id !== d.id && n[yAxisMetric] !== undefined && n[yAxisMetric] < newYVal
                      )
                      .sort((a, b) => b[yAxisMetric] - a[yAxisMetric])[0];
                }

                // 回弹
                d3.select(this).attr(
                  'transform',
                  `translate(${x(d[xAxisMetric])},${y(d[yAxisMetric])})`
                );

                if (mods.length > 0) {
                  const ghostId = generateUniqueId();
                  const ghost = {
                    ...d,
                    id: ghostId,
                    x: d.x,
                    y: d.y,
                    [xAxisMetric]: newXVal,
                    [yAxisMetric]: newYVal,
                    isGhost: true,
                    isNewlyGenerated: true,
                    level: d.level + 1,
                  };
                  setNodes((prev) => [...prev, ghost]);
                  setLinks((prev) => [...prev, { source: d.id, target: ghostId }]);
                  setPendingChange({
                    originalNode: d,
                    ghostNode: ghost,
                    modifications: mods,
                    behindNode: behind,
                    screenX: event.sourceEvent.clientX + 10,
                    screenY: event.sourceEvent.clientY + 10,
                  });
                }
                // Track click and set selected node
                setSelectedNode(d);
              })
          );

        nodeG
          .append('circle')
          .attr('r', (d) =>
            d.isBeingMerged ? 16 : d.id === mergeTargetId ? 12 : 8
          )
          .style('fill', (d) =>
            d.isMergedResult
              ? '#B22222'
              : d.isNewlyGenerated
                ? '#FFD700'
                : colorMap[d.type]
          )
          .style('opacity', (d) => {
            // Check if custom opacity is set
            if (d.evaluationOpacity !== undefined) return d.evaluationOpacity;
            // Otherwise use the default logic
            return d.isGhost ? 0.5 : 1;
          })
          .style('stroke', (d) =>
            selectedNode?.id === d.id ? '#000' : hoveredNode?.id === d.id ? '#555' : '#fff'
          )
          .style('stroke-width', (d) =>
            selectedNode?.id === d.id ? 4 : hoveredNode?.id === d.id ? 3 : 2
          );
      }
      if (operationStatus) {
        svg
          .append('rect')
          .attr('x', 2)
          .attr('y', 2)
          .attr('width', width - 4)
          .attr('height', height - 4)
          .attr('rx', 8)
          .attr('ry', 8)
          .style('fill', 'rgba(255, 255, 255, 0.7)')
          .style('pointer-events', 'none');
        svg
          .append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-size', '1.2rem')
          .style('font-weight', '600')
          .style('fill', '#374151')
          .text(operationStatus);
      }
    }
  }, [
    currentView,
    nodes,
    links,
    xAxisMetric,
    yAxisMetric,
    isGenerating,
    isEvaluating,
    operationStatus,
    // selectedNode and hoveredNode handled by separate useEffect for styling updates only
  ]);

  // Separate useEffect for updating node hover/selection styling without recreating event listeners
  useEffect(() => {
    if (!svgRef.current || currentView === 'home_view') return;

    // Update circle styling for hover and selection states
    d3.select(svgRef.current)
      .selectAll('circle')
      .style('stroke', (d) =>
        selectedNode?.id === d.id ? '#000' : hoveredNode?.id === d.id ? '#555' : '#fff'
      )
      .style('stroke-width', (d) =>
        selectedNode?.id === d.id ? 4 : hoveredNode?.id === d.id ? 3 : 2
      );
  }, [selectedNode, hoveredNode, currentView]);

  const handleAddCustomHypothesis = async (e) => {
    e.preventDefault();
    if (!customHypothesis.title.trim() || !customHypothesis.content.trim()) return;

    setIsGenerating(true);
    setOperationStatus('Adding custom hypothesis...');
    setError(null);

    try {
      const newId = generateUniqueId();
      const newHypothesis = {
        id: newId,
        ...customHypothesis,
      };

      // 添加到假设列表
      const updatedHypothesesList = [...hypothesesList, newHypothesis];
      setHypothesesList(updatedHypothesesList);

      // 创建新节点
      const newNode = {
        id: newId,
        level: selectedNode ? selectedNode.level + 1 : 1,
        title: customHypothesis.title.trim(),
        content: customHypothesis.content.trim(),
        type: 'complex',
        x: selectedNode ? selectedNode.x + Math.random() * 20 - 10 : 0,
        y: selectedNode ? selectedNode.y + 150 + Math.random() * 20 - 10 : 150,
        originalData: newHypothesis, // Preserve original data for coder/writer
      };

      // 添加节点和连接
      setNodes((prev) => [...prev, newNode]);
      if (selectedNode) {
        setLinks((prev) => [...prev, { source: selectedNode.id, target: newId }]);
      }

      // 评估新假设
      await evaluateHypotheses(updatedHypothesesList);

      // 重置表单
      setCustomHypothesis({ title: '', content: '' });
      setIsAddingCustom(false);
    } catch (err) {
      console.error('Error adding custom hypothesis:', err);
      setError(err.message);
    } finally {
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // Load generated files from public directory
  const loadGeneratedFiles = async (experimentDir) => {
    try {
      console.log("Loading generated files from:", experimentDir);

      // Clean up the experiment directory path to ensure it's relative to public
      let cleanDir = experimentDir;
      if (cleanDir.startsWith('../public/')) {
        cleanDir = cleanDir.replace('../public/', '');
      } else if (cleanDir.startsWith('./')) {
        cleanDir = cleanDir.replace('./', '');
      }

      console.log("Cleaned directory path:", cleanDir);

      // Load main experiment file via API
      try {
        const fileUrl = `/api/files/${cleanDir}/experiment.py`;
        console.log("Attempting to fetch:", fileUrl);
        const codeResponse = await fetch(fileUrl);
        console.log("Response status:", codeResponse.status, codeResponse.statusText);

        if (codeResponse.ok) {
          const responseData = await codeResponse.json();
          if (responseData.content) {
            setCodeContent(responseData.content);
            setExperimentFiles(prev => ({
              ...prev,
              'experiment.py': responseData.content
            }));
            // Set experiment.py as the active tab
            setActiveCodeTab('experiment.py');
            console.log("Loaded experiment.py successfully and set as active tab");
          }
        } else {
          console.log("Failed to load experiment.py:", codeResponse.status, codeResponse.statusText);
        }
      } catch (err) {
        console.log("Could not load experiment.py:", err);
      }

      // Load notes file via API
      try {
        const notesResponse = await fetch(`/api/files/${cleanDir}/notes.txt`);
        if (notesResponse.ok) {
          const notesData = await notesResponse.json();
          if (notesData.content) {
            setExperimentFiles(prev => ({
              ...prev,
              'notes.txt': notesData.content
            }));
            console.log("Loaded notes.txt successfully");
          }
        }
      } catch (err) {
        console.log("Could not load notes.txt:", err);
      }

      // Load any results files via API
      try {
        const resultsResponse = await fetch(`/api/files/${cleanDir}/experiment_results.txt`);
        if (resultsResponse.ok) {
          const resultsData = await resultsResponse.json();
          if (resultsData.content) {
            setExperimentFiles(prev => ({
              ...prev,
              'experiment_results.txt': resultsData.content
            }));
            console.log("Loaded experiment_results.txt successfully");
          }
        }
      } catch (err) {
        console.log("Could not load experiment_results.txt:", err);
      }

    } catch (err) {
      console.log("Error loading generated files:", err);
      setCodeContent(`# Generated experiment code\n# Files are being generated in: ${experimentDir}\n\n# Please check the directory for the actual code files.`);
    }
  };

  // Manual file loading function for debugging
  const handleManualLoadFiles = async () => {
    try {
      console.log("Manually loading generated files...");

      // Stop any ongoing code generation polling
      setIsGeneratingCode(false);
      setIsGeneratingPaper(false);
      setOperationStatus('');

      await loadGeneratedFiles("experiments");
      setCurrentView('code_view');
      console.log("Manual file loading completed");

      // Set a fake successful result to satisfy the UI
      setCodeResult({
        status: true,
        success: true,
        experiment_dir: "experiments"
      });

    } catch (err) {
      console.log("Manual file loading failed:", err);
    }
  };

  // Handler for Proceed button - with confirmation dialog
  const handleProceedWithSelectedIdea = () => {
    console.log("=== PROCEED BUTTON CLICKED ===");
    console.log("selectedNode:", selectedNode);
    console.log("selectedNode.originalData:", selectedNode?.originalData);

    if (!selectedNode || !selectedNode.originalData) {
      console.log("ERROR: Missing selectedNode or originalData");
      setProceedError('Please select a hypothesis to proceed');
      return;
    }

    console.log("Showing proceed confirmation dialog");
    setShowProceedConfirm(true);
  };

  // Handler for confirmed proceed action
  const handleConfirmProceed = async () => {
    console.log("=== PROCEED CONFIRMED ===");
    console.log("selectedNode.originalData:", selectedNode?.originalData);

    // Validate Semantic Scholar API key
    if (!s2ApiKey.trim()) {
      setProceedError('Semantic Scholar API key is required');
      return;
    }

    setShowProceedConfirm(false);
    setProceedError(null);

    if (!selectedNode?.originalData) {
      console.log("ERROR: No originalData available");
      setProceedError('No hypothesis selected');
      setShowProceedConfirm(true); // Show dialog again to display error
      return;
    }

    // Set the Semantic Scholar API key as environment variable
    try {
      console.log("Setting Semantic Scholar API key...");
      const apiKeyResponse = await fetch('/api/set-env', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          key: 'S2_API_KEY',
          value: s2ApiKey.trim()
        }),
      });

      if (!apiKeyResponse.ok) {
        throw new Error('Failed to set Semantic Scholar API key');
      }

      console.log("Semantic Scholar API key set successfully");
    } catch (error) {
      console.error("Error setting Semantic Scholar API key:", error);
      setProceedError('Failed to set Semantic Scholar API key: ' + error.message);
      setShowProceedConfirm(true); // Show dialog again to display error
      return;
    }

    // Check if the idea is experimental (AI-related) or not
    const idea = selectedNode.originalData;
    const isExperimental = idea.is_experimental === true;

    console.log(`Idea is experimental: ${isExperimental}`);
    console.log(`Idea details:`, {
      title: idea.Title || idea.Name,
      is_experimental: idea.is_experimental
    });

    try {
      // Check if backend is configured first
      console.log("Checking backend configuration...");
      setOperationStatus('Checking configuration...');

      const configCheck = await fetch('/api/get-prompts', {
        credentials: 'include'
      });

      if (!configCheck.ok) {
        console.log("Backend not configured, attempting to reconfigure...");
        setOperationStatus('Reconfiguring backend...');

        if (!apiKey.trim()) {
          throw new Error('Backend not configured and no API key available. Please configure the API key and model first.');
        }

        // Auto-reconfigure the backend
        const reconfigResponse = await fetch('/api/configure', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            model: selectedModel,
            api_key: apiKey,
          }),
        });

        if (!reconfigResponse.ok) {
          throw new Error('Failed to reconfigure backend');
        }

        console.log("Backend reconfigured successfully");
      }

      if (isExperimental) {
        // For experimental (AI-related) ideas: Generate code first, then paper
        console.log("Processing experimental idea - generating code first...");
        setIsGeneratingCode(true);
        setOperationStatus('Generating experiment code...');

        let codeData = null;
        let timeoutOccurred = false;

      try {
        // Try with a manual timeout using Promise.race
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

        const codeResponse = await fetch('/api/code', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            idea: selectedNode.originalData
          }),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!codeResponse.ok) {
          const errorText = await codeResponse.text();
          throw new Error(`Failed to generate code: ${codeResponse.status} ${errorText}`);
        }

        codeData = await codeResponse.json();
        console.log("Code generation completed:", codeData);

        if (!codeData.success) {
          throw new Error(codeData.error || 'Code generation failed');
        }
      } catch (error) {
        if (error.name === 'AbortError' || error.name === 'TimeoutError' ||
            error.message.includes('ECONNRESET') || error.message.includes('socket hang up') ||
            error.message.includes('Proxy error') || error.message.includes('network')) {
          console.log("Request failed due to connection issue, but backend may still be processing. Checking for generated files...");
          console.log("Error details:", error.message);
          timeoutOccurred = true;
          setOperationStatus('Connection lost, checking if code generation completed...');

          // Immediate check for existing files (backend might have already completed)
          try {
            const immediateCheck = await fetch('/api/files/generated/experiments/experiment.py');
            if (immediateCheck.ok) {
              const fileData = await immediateCheck.json();
              if (fileData.content && fileData.content.length > 1000) {
                console.log("Files already exist! Code generation was completed.");
                codeData = {
                  success: true,
                  experiment_dir: "generated/experiments",
                  status: true,
                  message: "Code generation completed (files found immediately)"
                };
                timeoutOccurred = false; // Skip polling
              }
            }
          } catch (immediateError) {
            console.log("Immediate file check failed, will start polling:", immediateError.message);
          }
        } else {
          throw error;
        }
      }

      // If timeout occurred, poll for files
      if (timeoutOccurred) {
        let attempts = 0;
        const maxAttempts = 30; // Check for 15 minutes (30s intervals)

        while (attempts < maxAttempts) {
          attempts++;
          setOperationStatus(`Code generation in progress, checking for completion (${attempts}/${maxAttempts})...`);

          try {
            // Check if experiment.py file exists and has content
            const fileResponse = await fetch('/api/files/generated/experiments/experiment.py');
            if (fileResponse.ok) {
              const fileData = await fileResponse.json();
              if (fileData.content && fileData.content.length > 1000) { // Check if file has substantial content
                console.log("Generated files found! Code generation completed successfully.");
                codeData = {
                  success: true,
                  experiment_dir: "generated/experiments",
                  status: true,
                  message: "Code generation completed (detected after connection timeout)"
                };
                break;
              }
            }
          } catch (e) {
            console.log(`File check attempt ${attempts} failed:`, e.message);
          }

          // Wait 30 seconds before next attempt
          await new Promise(resolve => setTimeout(resolve, 30000));
        }

        if (!codeData) {
          throw new Error('Code generation timed out and no files were generated');
        }
      }

        // Store results
        setCodeResult(codeData);
        const finalExperimentDir = codeData.experiment_dir;

        // Load generated files
        setOperationStatus('Loading generated files...');
        await loadGeneratedFiles(finalExperimentDir);

        // Mark that code has been generated (this will show Code View tab)
        setHasGeneratedCode(true);

        // Automatically switch to Code View
        setCurrentView('code_view');
        console.log("Switched to Code View to display generated files");
        setOperationStatus('Code generation completed successfully!');

        // Step 2: Generate paper for experimental idea
        setIsGeneratingCode(false);
        setIsGeneratingPaper(true);
        setOperationStatus('Preparing paper generation...');

      // Check configuration again before paper generation
      const paperConfigCheck = await fetch('/api/get-prompts', {
        credentials: 'include'
      });

      if (!paperConfigCheck.ok) {
        console.log("Backend not configured for paper generation, reconfiguring...");
        const reconfigResponse = await fetch('/api/configure', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            model: selectedModel,
            api_key: apiKey,
          }),
        });

        if (!reconfigResponse.ok) {
          throw new Error('Failed to reconfigure backend for paper generation');
        }
      }

      setOperationStatus('Generating paper...');
      const paperResponse = await fetch('/api/write', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          idea: selectedNode.originalData,
          experiment_dir: finalExperimentDir
        }),
      });

      if (!paperResponse.ok) {
        throw new Error('Failed to generate paper');
      }

      const paperData = await paperResponse.json();
      console.log("Paper generation completed:", paperData);
      console.log("PDF path for frontend:", paperData.pdf_path);
      setPaperResult(paperData);

      if (!paperData.success) {
        throw new Error(paperData.error || 'Paper generation failed');
      }

        // Switch to paper view to show results for experimental idea
        setCurrentView('paper_view');
        setOperationStatus('Code and paper generated successfully!');

      } else {
        // For non-experimental (non-AI-related) ideas: Generate paper directly
        console.log("Processing non-experimental idea - generating paper directly...");
        setIsGeneratingPaper(true);
        setOperationStatus('Generating paper...');

        // Check configuration again before paper generation
        const paperConfigCheck = await fetch('/api/get-prompts', {
          credentials: 'include'
        });

        if (!paperConfigCheck.ok) {
          console.log("Backend not configured for paper generation, reconfiguring...");
          const reconfigResponse = await fetch('/api/configure', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({
              model: selectedModel,
              api_key: apiKey,
            }),
          });

          if (!reconfigResponse.ok) {
            throw new Error('Failed to reconfigure backend for paper generation');
          }
        }

        // Generate paper directly without code generation
        const paperResponse = await fetch('/api/write', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            idea: selectedNode.originalData,
            experiment_dir: null // No experiment directory for non-experimental papers
          }),
        });

        if (!paperResponse.ok) {
          throw new Error('Failed to generate paper');
        }

        const paperData = await paperResponse.json();
        console.log("Paper generation completed:", paperData);
        console.log("PDF path for frontend:", paperData.pdf_path);
        setPaperResult(paperData);

        if (!paperData.success) {
          throw new Error(paperData.error || 'Paper generation failed');
        }

        // Switch to paper view to show results for non-experimental idea
        setCurrentView('paper_view');
        setOperationStatus('Paper generated successfully!');
      }

    } catch (err) {
      console.error('=== ERROR IN PROCEED WORKFLOW ===');
      console.error('Error details:', err);
      console.error('Error message:', err.message);
      console.error('Error stack:', err.stack);
      setProceedError(err.message);
    } finally {
      setIsGeneratingCode(false);
      setIsGeneratingPaper(false);
      setOperationStatus('');
    }
  };

  // PDF Comment functions
  const addComment = (pageNumber = 1) => {
    if (newComment.trim()) {
      const comment = {
        id: Date.now(),
        text: newComment,
        pageNumber,
        timestamp: new Date().toLocaleString(),
        author: 'User'
      };
      setPdfComments([...pdfComments, comment]);
      setNewComment('');
    }
  };

  const deleteComment = (commentId) => {
    setPdfComments(pdfComments.filter(c => c.id !== commentId));
  };

  const downloadPDF = (pdfPath) => {
    // Create a link to download the PDF
    const link = document.createElement('a');
    link.href = `http://localhost:8080${pdfPath}?download=true`;
    link.download = 'research_paper.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Enhanced code editing functions
  const switchCodeTab = (tabName) => {
    // Save current content before switching
    if (activeCodeTab && experimentFiles[activeCodeTab] !== undefined) {
      setExperimentFiles(prev => ({
        ...prev,
        [activeCodeTab]: codeContent
      }));
    }

    // Switch to new tab
    setActiveCodeTab(tabName);
    setCodeFileName(tabName);
    setCodeContent(experimentFiles[tabName] || '');
  };



  const downloadExperimentFiles = async () => {
    // Create a zip file with all experiment files
    const filesToDownload = Object.entries(experimentFiles);
    if (filesToDownload.length === 0) {
      alert('No experiment files to download');
      return;
    }

    try {
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();

      // Add all experiment files to the zip
      filesToDownload.forEach(([fileName, content]) => {
        zip.file(fileName, content);
      });

      // Generate the zip file
      const zipBlob = await zip.generateAsync({ type: 'blob' });

      // Create download link
      const link = document.createElement('a');
      link.href = URL.createObjectURL(zipBlob);
      link.download = 'experiment_files.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the object URL
      URL.revokeObjectURL(link.href);
    } catch (error) {
      console.error('Error creating zip file:', error);
      alert('Failed to create zip file. Please try again.');
    }
  };


  // ============== 段落11：整体布局，渲染主界面 JSX ==============
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', position: 'relative' }}>
      {/* 把 showTree 与 setShowTree 传给 TopNav，解决 setShowTree is not a function */}
      <TopNav currentView={currentView} setCurrentView={setCurrentView} showCodeView={hasGeneratedCode} />

      {currentView === 'home_view' ? (
        <OverviewPage />
      ) : currentView === 'code_view' ? (
        /* Enhanced Code View - Full Screen */
        <div style={{ padding: '20px', backgroundColor: '#f9fafb', minHeight: '90vh' }}>
          <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h1 style={{ fontSize: '1.5rem', fontWeight: 600, color: '#1f2937', margin: 0 }}>
                Experiment Code Editor
              </h1>

              {/* Enhanced Toolbar */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>

                {/* Download All Button */}
                <button
                  onClick={downloadExperimentFiles}
                  disabled={Object.keys(experimentFiles).length === 0}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: Object.keys(experimentFiles).length > 0 ? '#4C84FF' : '#9CA3AF',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: Object.keys(experimentFiles).length > 0 ? 'pointer' : 'not-allowed',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  💾 Download All
                </button>

              </div>
            </div>

            {isGeneratingCode && (
              <div style={{
                padding: '20px',
                backgroundColor: '#eff6ff',
                borderRadius: '8px',
                marginBottom: '20px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#1d4ed8', fontSize: '1rem', fontWeight: 500 }}>
                  🔄 Generating experimental code...
                </div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem', marginTop: '8px' }}>
                  This may take several minutes, please wait...
                </div>
              </div>
            )}

            {codeResult && codeResult.success && (
              <div style={{ backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '16px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <span style={{ color: '#16a34a', fontSize: '1.1rem' }}>✅</span>
                  <strong style={{ color: '#166534' }}>Experiment Generated Successfully</strong>
                </div>
                <div style={{ display: 'flex', gap: '20px', fontSize: '0.875rem', color: '#166534' }}>
                  <div>
                    <strong>Directory:</strong>
                    <code style={{ backgroundColor: '#dcfce7', padding: '2px 6px', borderRadius: '4px', marginLeft: '8px' }}>
                      {codeResult.experiment_dir}
                    </code>
                  </div>
                  <div>
                    <strong>Files:</strong> {Object.keys(experimentFiles).length} loaded
                  </div>
                </div>
              </div>
            )}

            {/* Main Layout with Sidebar */}
            <div style={{ display: 'flex', gap: '20px', height: '700px' }}>
              {/* File Explorer Sidebar */}
              <div style={{
                width: '250px',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                display: 'flex',
                flexDirection: 'column'
              }}>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '0.875rem',
                  color: '#64748b',
                  fontWeight: 500
                }}>
                  📁 Experiment Files
                </div>

                <div style={{ flex: 1, padding: '8px' }}>
                  {Object.keys(experimentFiles).length === 0 ? (
                    <div style={{
                      textAlign: 'center',
                      padding: '20px',
                      color: '#6b7280',
                      fontSize: '0.875rem'
                    }}>
                      No files loaded
                    </div>
                  ) : (
                    Object.keys(experimentFiles).map((fileName) => (
                      <div
                        key={fileName}
                        onClick={() => switchCodeTab(fileName)}
                        style={{
                          padding: '8px 12px',
                          cursor: 'pointer',
                          borderRadius: '4px',
                          marginBottom: '4px',
                          backgroundColor: activeCodeTab === fileName ? '#eff6ff' : 'transparent',
                          color: activeCodeTab === fileName ? '#1d4ed8' : '#374151',
                          fontSize: '0.875rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}
                      >
                        {fileName.endsWith('.py') ? '🐍' : fileName.endsWith('.txt') ? '📄' : '📝'}
                        {fileName}
                      </div>
                    ))
                  )}

                  {experimentRuns.length > 0 && (
                    <>
                      <div style={{
                        padding: '8px 12px',
                        fontSize: '0.875rem',
                        fontWeight: 500,
                        color: '#64748b',
                        borderTop: '1px solid #e5e7eb',
                        marginTop: '8px',
                        paddingTop: '12px'
                      }}>
                        📊 Experiment Runs
                      </div>
                      {experimentRuns.map((run, index) => (
                        <div
                          key={index}
                          style={{
                            padding: '8px 12px',
                            borderRadius: '4px',
                            marginBottom: '4px',
                            backgroundColor: '#f9fafb',
                            fontSize: '0.875rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                          }}
                        >
                          {run.success ? '✅' : '❌'} Run {index + 1}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>

              {/* Main Editor Area */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* File Tabs */}
                {Object.keys(experimentFiles).length > 0 && (
                  <div style={{
                    display: 'flex',
                    backgroundColor: 'white',
                    borderRadius: '8px 8px 0 0',
                    border: '1px solid #e5e7eb',
                    borderBottom: 'none'
                  }}>
                    {Object.keys(experimentFiles).map((fileName) => (
                      <div
                        key={fileName}
                        onClick={() => switchCodeTab(fileName)}
                        style={{
                          padding: '8px 16px',
                          cursor: 'pointer',
                          backgroundColor: activeCodeTab === fileName ? '#f8fafc' : 'transparent',
                          borderBottom: activeCodeTab === fileName ? '2px solid #4C84FF' : '2px solid transparent',
                          fontSize: '0.875rem',
                          fontWeight: activeCodeTab === fileName ? 500 : 400,
                          color: activeCodeTab === fileName ? '#1f2937' : '#6b7280'
                        }}
                      >
                        {fileName}
                      </div>
                    ))}
                  </div>
                )}

                {/* Monaco Editor */}
                <div style={{
                  flex: 1,
                  backgroundColor: 'white',
                  borderRadius: Object.keys(experimentFiles).length > 0 ? '0 0 8px 8px' : '8px',
                  border: '1px solid #e5e7eb',
                  overflow: 'hidden',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <Editor
                    height="100%"
                    language={activeCodeTab.endsWith('.py') ? 'python' : activeCodeTab.endsWith('.js') ? 'javascript' : activeCodeTab.endsWith('.ts') ? 'typescript' : activeCodeTab.endsWith('.json') ? 'json' : activeCodeTab.endsWith('.md') ? 'markdown' : 'plaintext'}
                    value={codeContent}
                    onChange={(value) => setCodeContent(value || '')}
                    theme="vs-dark"
                    options={{
                      minimap: { enabled: true },
                      fontSize: 14,
                      lineNumbers: 'on',
                      wordWrap: 'on',
                      automaticLayout: true,
                      scrollBeyondLastLine: false,
                      folding: true,
                      renderLineHighlight: 'all',
                      selectOnLineNumbers: true
                    }}
                  />
                </div>
              </div>

              {/* Console Output Panel */}
              <div style={{
                width: '300px',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                display: 'flex',
                flexDirection: 'column'
              }}>
                <div style={{
                  backgroundColor: '#f8fafc',
                  padding: '12px 16px',
                  borderBottom: '1px solid #e5e7eb',
                  fontSize: '0.875rem',
                  color: '#64748b',
                  fontWeight: 500,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  💻 Console Output
                  <button
                    onClick={() => setConsoleOutput('')}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#6b7280',
                      cursor: 'pointer',
                      fontSize: '0.75rem'
                    }}
                  >
                    Clear
                  </button>
                </div>

                <div style={{
                  flex: 1,
                  padding: '12px',
                  fontSize: '0.75rem',
                  fontFamily: 'monospace',
                  backgroundColor: '#1a1a1a',
                  color: '#e5e7eb',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap'
                }}>
                  {consoleOutput || 'No output yet...'}
                </div>
              </div>
            </div>

            {proceedError && (
              <div style={{
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '8px',
                padding: '16px',
                color: '#dc2626',
                marginTop: '20px'
              }}>
                <strong>Error:</strong> {proceedError}
              </div>
            )}

            {!codeResult && !isGeneratingCode && !proceedError && Object.keys(experimentFiles).length === 0 && (
              <div style={{
                textAlign: 'center',
                padding: '40px',
                color: '#6b7280',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '2px dashed #d1d5db'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '16px' }}>🧪</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 500, marginBottom: '8px' }}>No experiment loaded</div>
                <div style={{ fontSize: '0.875rem' }}>
                  Use the "Proceed" button in Exploration View to generate an experiment, or upload files to get started
                </div>
              </div>
            )}
          </div>
        </div>
      ) : currentView === 'paper_view' ? (
        /* Paper View - Full Screen */
        <div style={{ padding: '20px', backgroundColor: '#f9fafb', minHeight: '90vh' }}>
          <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h1 style={{ fontSize: '1.5rem', fontWeight: 600, color: '#1f2937', margin: 0 }}>
                Research Paper Viewer
              </h1>

              {paperResult && paperResult.pdf_path && (
                <button
                  onClick={() => downloadPDF(paperResult.pdf_path)}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#4C84FF',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  📄 Download PDF
                </button>
              )}
            </div>

            {isGeneratingPaper && (
              <div style={{
                padding: '20px',
                backgroundColor: '#eff6ff',
                borderRadius: '8px',
                marginBottom: '20px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#1d4ed8', fontSize: '1rem', fontWeight: 500 }}>
                  📄 Generating research paper...
                </div>
                <div style={{ color: '#6b7280', fontSize: '0.875rem', marginTop: '8px' }}>
                  This may take several minutes
                </div>
              </div>
            )}

            {paperResult && paperResult.success && (
              <div style={{ backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '16px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <span style={{ color: '#16a34a', fontSize: '1.1rem' }}>✅</span>
                  <strong style={{ color: '#166534' }}>Paper Generated Successfully</strong>
                </div>
                <div style={{ display: 'flex', gap: '20px', fontSize: '0.875rem', color: '#166534' }}>
                  {paperResult.pdf_path && (
                    <div>
                      <strong>PDF:</strong>
                      <code style={{ backgroundColor: '#dcfce7', padding: '2px 6px', borderRadius: '4px', marginLeft: '8px' }}>
                        {paperResult.pdf_path.split('/').pop()}
                      </code>
                    </div>
                  )}
                  {paperResult.latex_path && (
                    <div>
                      <strong>LaTeX:</strong>
                      <code style={{ backgroundColor: '#dcfce7', padding: '2px 6px', borderRadius: '4px', marginLeft: '8px' }}>
                        {paperResult.latex_path.split('/').pop()}
                      </code>
                    </div>
                  )}
                </div>
              </div>
            )}

            {paperResult && paperResult.pdf_path && (
              <div style={{ display: 'flex', gap: '20px', height: '800px' }}>
                {/* PDF Viewer */}
                <div style={{
                  flex: '1',
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  overflow: 'hidden',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <div style={{
                    backgroundColor: '#f8fafc',
                    padding: '12px 16px',
                    borderBottom: '1px solid #e5e7eb',
                    fontSize: '0.875rem',
                    color: '#64748b',
                    fontWeight: 500
                  }}>
                    📄 Research Paper
                  </div>
                  <iframe
                    src={`http://localhost:8080${paperResult.pdf_path}#toolbar=1&navpanes=1&scrollbar=1&page=1&view=FitH&zoom=100`}
                    style={{
                      width: '100%',
                      height: 'calc(100% - 45px)',
                      border: 'none',
                      minHeight: '600px'
                    }}
                    title="Research Paper PDF"
                    onError={(e) => {
                      console.error('PDF iframe load error:', e);
                      console.log('PDF path:', paperResult.pdf_path);
                    }}
                    onLoad={() => {
                      console.log('PDF loaded successfully:', paperResult.pdf_path);
                    }}
                  />
                </div>

                {/* Comments Panel */}
                <div style={{
                  width: '400px',
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  display: 'flex',
                  flexDirection: 'column'
                }}>
                  <div style={{
                    backgroundColor: '#f8fafc',
                    padding: '12px 16px',
                    borderBottom: '1px solid #e5e7eb',
                    fontSize: '0.875rem',
                    color: '#64748b',
                    fontWeight: 500
                  }}>
                    💬 Comments ({pdfComments.length})
                  </div>

                  {/* Add Comment */}
                  <div style={{ padding: '16px', borderBottom: '1px solid #e5e7eb' }}>
                    <textarea
                      value={newComment}
                      onChange={(e) => setNewComment(e.target.value)}
                      placeholder="Add a comment about the paper..."
                      style={{
                        width: '100%',
                        height: '80px',
                        padding: '8px 12px',
                        border: '1px solid #d1d5db',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                        resize: 'vertical',
                        fontFamily: 'inherit'
                      }}
                    />
                    <button
                      onClick={() => addComment()}
                      disabled={!newComment.trim()}
                      style={{
                        marginTop: '8px',
                        padding: '6px 12px',
                        backgroundColor: newComment.trim() ? '#4C84FF' : '#9CA3AF',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: newComment.trim() ? 'pointer' : 'not-allowed',
                        fontSize: '0.8rem',
                        fontWeight: 500
                      }}
                    >
                      Add Comment
                    </button>
                  </div>

                  {/* Comments List */}
                  <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
                    {pdfComments.length === 0 ? (
                      <div style={{
                        textAlign: 'center',
                        padding: '40px 20px',
                        color: '#6b7280',
                        fontSize: '0.875rem'
                      }}>
                        <div style={{ fontSize: '2rem', marginBottom: '8px' }}>💭</div>
                        No comments yet. Add your thoughts about the paper!
                      </div>
                    ) : (
                      pdfComments.map((comment) => (
                        <div key={comment.id} style={{
                          backgroundColor: '#f9fafb',
                          border: '1px solid #e5e7eb',
                          borderRadius: '6px',
                          padding: '12px',
                          marginBottom: '8px'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                            <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                              <strong>{comment.author}</strong> • {comment.timestamp}
                            </div>
                            <button
                              onClick={() => deleteComment(comment.id)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: '#dc2626',
                                cursor: 'pointer',
                                fontSize: '0.75rem',
                                padding: '2px'
                              }}
                            >
                              ✕
                            </button>
                          </div>
                          <div style={{ fontSize: '0.875rem', color: '#374151', lineHeight: '1.4' }}>
                            {comment.text}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}

            {proceedError && (
              <div style={{
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '8px',
                padding: '16px',
                color: '#dc2626',
                marginBottom: '20px'
              }}>
                <strong>Error:</strong> {proceedError}
              </div>
            )}

            {!paperResult && !isGeneratingPaper && !proceedError && (
              <div style={{
                textAlign: 'center',
                padding: '40px',
                color: '#6b7280',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '2px dashed #d1d5db'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '16px' }}>📄</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 500, marginBottom: '8px' }}>No paper generated</div>
                <div style={{ fontSize: '0.875rem' }}>
                  Use the "Proceed" button in Exploration View to generate a research paper
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          {currentView === 'exploration' && !isAnalysisSubmitted && (
            <div
              style={{
                backgroundColor: '#1f2937',
                display: 'flex',
                alignItems: 'center',
                padding: '10px 20px',
              }}
            >
              <form
                ref={analysisFormRef}
                onSubmit={handleAnalysisIntentSubmit}
                style={{ display: 'flex', alignItems: 'center' }}
              >
                <input
                  type="text"
                  value={analysisIntent}
                  onChange={(e) => setAnalysisIntent(e.target.value)}
                  placeholder="Enter analysis intent"
                  style={{
                    padding: '6px 10px',
                    borderRadius: '4px',
                    border: '1px solid #d1d5db',
                    marginRight: '8px',
                  }}
                />
                <button
                  type="submit"
                  disabled={isGenerating || isEvaluating}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: '#4C84FF',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px 0 0 6px',
                    cursor: isGenerating || isEvaluating ? 'not-allowed' : 'pointer',
                    opacity: isGenerating || isEvaluating ? 0.7 : 1,
                    fontSize: '0.875rem',
                    fontWeight: '500',
                  }}
                >
                  {isEvaluating ? 'Evaluating...' : isGenerating ? 'Generating...' : 'Submit'}
                </button>
                <button
                  ref={systemPromptButtonRef}
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setModalAnchorEl(systemPromptButtonRef);
                    setIsEditingSystemPrompt(true);
                  }}
                  style={{
                    padding: '6px 10px',
                    backgroundColor: '#4C84FF',
                    color: 'white',
                    border: 'none',
                    borderLeft: '1px solid #6D97FF',
                    borderRadius: '0 6px 6px 0',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: isGenerating || isEvaluating ? 'not-allowed' : 'pointer',
                    opacity: isGenerating || isEvaluating ? 0.7 : 1,
                  }}
                  title="Edit system prompt"
                >
                  {editIcon}
                </button>
                {error && (
                  <div
                    style={{
                      marginLeft: '16px',
                      color: '#dc2626',
                      fontSize: '0.875rem',
                    }}
                  >
                    {error}
                  </div>
                )}
              </form>
            </div>
          )}

          <div
            style={{
              display: 'flex',
              padding: '20px',
              maxWidth: '1600px',
              margin: '0 auto',
              boxSizing: 'border-box',
            }}
          >
            {/* 左侧图 */}
            <div ref={svgContainerRef} style={{ flexBasis: '60%', marginRight: '20px' }}>
              <svg ref={svgRef} />
            </div>

            {/* 右侧 Dashboard */}
            <div style={{ flexBasis: '40%' }} ref={dashboardContainerRef}>
              <Dashboard
                nodeId={hoveredNode?.id || selectedNode?.id}
                nodes={nodes}
                isEvaluating={isEvaluating}
                showTree={currentView === 'exploration'}
                setModalAnchorEl={setModalAnchorEl}
                setEditingCriteria={setEditingCriteria}
                // Pass props down to ContextAndGenerateCard
                isAddingCustom={isAddingCustom}
                userInput={userInput}
                setUserInput={setUserInput}
                generateChildNodes={generateChildNodes}
                selectedNode={selectedNode}
                isGenerating={isGenerating}
                isAnalysisSubmitted={isAnalysisSubmitted}
                setIsAddingCustom={setIsAddingCustom}
                handleAddCustomHypothesis={handleAddCustomHypothesis}
                customHypothesis={customHypothesis}
                setCustomHypothesis={setCustomHypothesis}
                setIsEditingSystemPrompt={setIsEditingSystemPrompt}
                editIcon={editIcon}
                handleProceedWithSelectedIdea={handleProceedWithSelectedIdea}
              />
            </div>
          </div>
        </>)}

      {/* ========== 段落12：悬浮确认修改的按钮 (pendingChange) ========== */}
      {pendingChange && (
        <div
          style={{
            position: 'absolute',
            left: pendingChange.screenX,
            top: pendingChange.screenY,
            backgroundColor: '#f3f4f6',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            padding: '6px 8px',
            zIndex: 999,
          }}
        >
          <div style={{ display: 'flex', gap: '8px' }}>
            {/* Generate New Hypothesis */}
            <button
              style={{
                padding: '4px 8px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              onClick={() => {
                modifyHypothesisBasedOnModifications(
                  pendingChange.originalNode,
                  pendingChange.ghostNode,
                  pendingChange.modifications,
                  pendingChange.behindNode
                );
                setPendingChange(null);
              }}
            >
              New Hypothesis
            </button>
            {/* Modify Original */}
            <button
              style={{
                padding: '4px 8px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              onClick={() => {
                const modifiedOriginal = {
                  ...pendingChange.originalNode,
                  x: pendingChange.ghostNode.x,
                  y: pendingChange.ghostNode.y,
                  [xAxisMetric]: pendingChange.ghostNode[xAxisMetric],
                  [yAxisMetric]: pendingChange.ghostNode[yAxisMetric],
                };
                // Remove the ghost node and update the original node
                setNodes((prev) =>
                  prev
                    .filter((n) => n.id !== pendingChange.ghostNode.id)
                    .map((n) => (n.id === modifiedOriginal.id ? modifiedOriginal : n))
                );
                setLinks((prev) =>
                  prev.filter((lk) => lk.target !== pendingChange.ghostNode.id)
                );
                setPendingChange(null);
              }}
            >
              Modify Eval
            </button>
            {/* Cancel */}
            <button
              style={{
                padding: '4px 8px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              onClick={() => {
                setNodes((prev) =>
                  prev.filter((nd) => nd.id !== pendingChange.ghostNode.id)
                );
                setLinks((prev) =>
                  prev.filter((lk) => lk.target !== pendingChange.ghostNode.id)
                );
                setPendingChange(null);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {pendingMerge && (
        <div
          style={{
            position: 'absolute',
            left: pendingMerge.screenX,
            top: pendingMerge.screenY,
            backgroundColor: '#f3f4f6',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            padding: '6px 8px',
            zIndex: 1000,
          }}
        >
          <div style={{ marginBottom: '6px', fontSize: '0.8rem', color: '#374151' }}>
            Merge these two hypotheses?
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              style={{
                padding: '4px 12px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              onClick={() => {
                mergeHypotheses(pendingMerge.nodeA, pendingMerge.nodeB);
                setPendingMerge(null);
              }}
            >
              Merge
            </button>
            <button
              style={{
                padding: '4px 12px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              onClick={() => {
                setNodes(prev =>
                  prev.map(n => {
                    if (n.id === pendingMerge.nodeA.id) {
                      return { ...n, evaluationOpacity: 1 };
                    } else if (n.id === pendingMerge.nodeB.id) {
                      return { ...n, isBeingMerged: false };
                    }
                    return n;
                  })
                );
                setMergeTargetId(null);
                setPendingMerge(null)
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {/* System Prompt Edit Modal */}
      <EditModal
        isOpen={isEditingSystemPrompt}
        onClose={() => {
          setIsEditingSystemPrompt(false);
          setModalAnchorEl(null);
        }}
        title="Edit System Prompt"
        defaultText={defaultPrompts.system_prompt}
        initialValue={systemPrompt}
        onSave={async (finalValue) => {
          const promptToSave = finalValue || defaultPrompts.system_prompt;
          const success = await updateSystemPrompt(promptToSave);
          if (success) {
            setSystemPrompt(promptToSave);
            setIsEditingSystemPrompt(false);
            setModalAnchorEl(null);
          }
        }}
        anchorRef={modalAnchorEl}
      />

      {/* Criteria Edit Modal */}
      <EditModal
        isOpen={editingCriteria !== null}
        onClose={() => {
          setEditingCriteria(null);
          setModalAnchorEl(null);
        }}
        title={`Edit ${editingCriteria ? editingCriteria.charAt(0).toUpperCase() + editingCriteria.slice(1) : ''} Criteria`}
        defaultText={
          editingCriteria === 'impact' ? defaultPrompts.impact :
            editingCriteria === 'feasibility' ? defaultPrompts.feasibility :
              defaultPrompts.novelty
        }
        initialValue={
          editingCriteria === 'impact' ? impactCriteria :
            editingCriteria === 'feasibility' ? feasibilityCriteria :
              noveltyCriteria
        }
        onSave={async (finalValue) => {
          const criteriaToSave = finalValue || (editingCriteria === 'impact' ? defaultPrompts.impact : editingCriteria === 'feasibility' ? defaultPrompts.feasibility : defaultPrompts.novelty);

          const currentCriteria = editingCriteria === 'impact' ? impactCriteria :
            editingCriteria === 'feasibility' ? feasibilityCriteria :
              noveltyCriteria;


          const success = await updateCriteria(editingCriteria, criteriaToSave);
          if (success) {
            if (editingCriteria === 'impact') setImpactCriteria(criteriaToSave);
            else if (editingCriteria === 'feasibility') setFeasibilityCriteria(criteriaToSave);
            else if (editingCriteria === 'novelty') setNoveltyCriteria(criteriaToSave);

            setEditingCriteria(null); // Close the modal
          }
        }}
        anchorRef={modalAnchorEl}
        width={dashboardWidth}
      />

      {/* Proceed Confirmation Dialog */}
      {showProceedConfirm && (
        <>
          {/* Modal Backdrop */}
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.5)',
              zIndex: 999,
            }}
            onClick={() => {
              setShowProceedConfirm(false);
              setProceedError(null);
            }}
          />

          {/* Modal Content */}
          <div
            style={{
              position: 'fixed',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              padding: '20px',
              zIndex: 1000,
              boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
              minWidth: '400px',
            }}
          >
            <div style={{ marginBottom: '16px', fontSize: '1rem', color: '#374151', fontWeight: 500 }}>
              Generate Code and Paper
            </div>
            <div style={{ marginBottom: '16px', fontSize: '0.875rem', color: '#6B7280', lineHeight: '1.5' }}>
              This will generate experimental code and write a research paper for the selected hypothesis.
              This process may take several minutes to complete.
            </div>

            {/* Semantic Scholar API Key Input */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{
                display: 'block',
                marginBottom: '6px',
                fontSize: '0.875rem',
                color: '#374151',
                fontWeight: 500
              }}>
                Semantic Scholar API Key *
              </label>
              <input
                type="password"
                value={s2ApiKey}
                onChange={(e) => setS2ApiKey(e.target.value)}
                placeholder="Enter your Semantic Scholar API key"
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  fontSize: '0.875rem',
                  color: '#374151',
                  backgroundColor: '#fff',
                  boxSizing: 'border-box'
                }}
              />
              <div style={{ marginTop: '4px', fontSize: '0.75rem', color: '#6B7280' }}>
                Required for paper generation. Get your API key from{' '}
                <a
                  href="https://www.semanticscholar.org/product/api"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#4C84FF', textDecoration: 'underline' }}
                >
                  Semantic Scholar API
                </a>
              </div>
            </div>

            {proceedError && (
              <div style={{
                marginBottom: '16px',
                padding: '8px 12px',
                backgroundColor: '#FEF2F2',
                border: '1px solid #FECACA',
                borderRadius: '4px',
                fontSize: '0.875rem',
                color: '#DC2626'
              }}>
                {proceedError}
              </div>
            )}
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#fff',
                  color: '#374151',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                }}
                onClick={() => {
                  setShowProceedConfirm(false);
                  setProceedError(null);
                }}
              >
                Cancel
              </button>
              <button
                style={{
                  padding: '8px 16px',
                  backgroundColor: (!s2ApiKey.trim() || isGeneratingCode || isGeneratingPaper) ? '#9CA3AF' : '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: (!s2ApiKey.trim() || isGeneratingCode || isGeneratingPaper) ? 'not-allowed' : 'pointer',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                }}
                onClick={handleConfirmProceed}
                disabled={!s2ApiKey.trim() || isGeneratingCode || isGeneratingPaper}
              >
                {isGeneratingCode || isGeneratingPaper ? 'Processing...' : 'Yes, Proceed'}
              </button>
            </div>
          </div>
        </>
      )}
    </div>

  );
};

export default TreePlotVisualization;
