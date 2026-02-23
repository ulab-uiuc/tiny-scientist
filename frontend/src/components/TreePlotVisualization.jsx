// ============== 段落1：导入依赖 ==============
import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';

import TopNav from './TopNav';
import IdeaCard from './IdeaCard';
import DimensionSelectorPanel from './DimensionSelectorPanel';
import DimensionEditDropdown from './DimensionEditDropdown';
import Evaluation3D from './Evaluation3D';
import { buildNodeContent } from '../utils/contentParser';


const COLOR_MAP = {
  root: '#4C84FF',
  simple: '#45B649',
  complex: '#FF6B6B',
};


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
  handleAddCustomIdea,
  customIdea,
  setCustomIdea,
  setIsEditingSystemPrompt,
  setModalAnchorEl,
  editIcon,
  handleProceedWithSelectedIdea,
  getNodeTrackingInfo
}) => {
  const newEditButtonRef = useRef(null);
  const generateButtonRef = useRef(null);
  const addCustomButtonRef = useRef(null);
  const contextInputRef = useRef(null);
  const customFormRef = useRef(null);

  // Track user interactions (disable hover to avoid conflicts with D3 node tracking)

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
            Add context for new ideas (optional)
          </label>
          <textarea
            ref={contextInputRef}
            id="context-input"
            rows={2}
            value={userInput}
            onChange={(e) => {
              setUserInput(e.target.value);
            }}
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
                Generate New Ideas
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
              Add Custom Idea
            </button>

          </div>
        </div>
      ) : (
        <form
          ref={customFormRef}
          onSubmit={(e) => {
            e.stopPropagation();
            handleAddCustomIdea(e);
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ marginBottom: '12px' }}>
            <label htmlFor="custom-idea-title" style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
              Title (2-3 words)
            </label>
            <input
              id="custom-idea-title"
              type="text"
              value={customIdea.title}
              onChange={(e) => {
                setCustomIdea(prev => ({ ...prev, title: e.target.value }));
              }}
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
            <label htmlFor="custom-idea-content" style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
              Idea Content
            </label>
            <textarea
              id="custom-idea-content"
              value={customIdea.content}
              onChange={(e) => {
                setCustomIdea(prev => ({ ...prev, content: e.target.value }));
              }}
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
              Add Idea
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsAddingCustom(false);
                setCustomIdea({ title: '', content: '' });
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
  fragmentNodes = [], // 新增：Fragment 节点数组
  isEvaluating,
  showTree,
  setModalAnchorEl,
  // Props for ContextAndGenerateCard
  isAddingCustom,
  userInput,
  setUserInput,
  generateChildNodes,
  selectedNode,
  isGenerating,
  isAnalysisSubmitted,
  setIsAddingCustom,
  handleAddCustomIdea,
  customIdea,
  setCustomIdea,
  setIsEditingSystemPrompt,
  editIcon,
  handleProceedWithSelectedIdea,
  getNodeTrackingInfo,
  // Props for score modification
  onModifyScore,
  showModifyButton = false,
  pendingChanges = null,
  currentView = 'evaluation', // Current view context
  selectedDimensionPairs = [], // 添加这个 prop
  activeDimensions = null, // 当前展示的维度平面
  reEvaluateAll, // pass down full re-eval handler
  onShowFragmentMenu, // 新增：Fragment 菜单回调
  activeDimensionIndices, // Active dimension indices
  onToggleDimensionIndex, // Callback to toggle dimension index
  onCreateFragmentFromHighlight, // 点击重点直接生成 fragment
  onSwapDimension = null, // 交换维度方向
  onEditDimension = null, // 编辑维度 (pairIndex, anchorRect) => void
}) => {
  const [showAfter, setShowAfter] = useState(true);

  useEffect(() => {
    setShowAfter(true);
  }, [nodeId]);

  // 先从 nodes 数组查找，如果没找到再从 fragmentNodes 查找
  const node = nodeId ?
    (nodes.find(n => n.id === nodeId) || fragmentNodes.find(f => f.id === nodeId))
    : null;

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
      <IdeaCard
        node={node}
        showAfter={showAfter}
        setShowAfter={setShowAfter}
        onModifyScore={onModifyScore}
        showModifyButton={showModifyButton}
        pendingChanges={pendingChanges}
        currentView={currentView}
        selectedDimensionPairs={selectedDimensionPairs}
        activeDimensions={activeDimensions}
        onReEvaluateAll={reEvaluateAll}
        isEvaluating={isEvaluating}
        onShowFragmentMenu={onShowFragmentMenu}
        activeDimensionIndices={activeDimensionIndices}
        onToggleDimensionIndex={onToggleDimensionIndex}
        onCreateFragmentFromHighlight={onCreateFragmentFromHighlight}
        onSwapDimension={onSwapDimension}
        onEditDimension={onEditDimension}
      />

      {/* Proceed button — only for real idea nodes */}
      {node.type !== 'root' && node.type !== 'fragment' && handleProceedWithSelectedIdea && (
        <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #e5e7eb' }}>
          <button
            onClick={() => handleProceedWithSelectedIdea(node)}
            style={{
              width: '100%',
              padding: '10px 16px',
              backgroundColor: '#0F172A',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              fontSize: '0.9rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1e293b'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#0F172A'}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Proceed with this Idea
          </button>
        </div>
      )}

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
          handleAddCustomIdea={handleAddCustomIdea}
          customIdea={customIdea}
          setCustomIdea={setCustomIdea}
          setIsEditingSystemPrompt={setIsEditingSystemPrompt}
          setModalAnchorEl={setModalAnchorEl}
          editIcon={editIcon}
          handleProceedWithSelectedIdea={handleProceedWithSelectedIdea}
          getNodeTrackingInfo={getNodeTrackingInfo}
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
  // *** 新增：维度选择相关状态 ***
  const [selectedDimensionPairs, setSelectedDimensionPairs] = useState([]);
  const [activeDimensionIndices, setActiveDimensionIndices] = useState([0, 1, 2]); // Default all 3 active
  const [currentFaceIndex, setCurrentFaceIndex] = useState(0); // 0/1/2 对应三张面
  const [showDimensionPanel, setShowDimensionPanel] = useState(false); // 下拉面板控制
  const [currentIntent, setCurrentIntent] = useState(''); // 存储当前 intent
  // *** 新增：维度编辑弹窗状态 ***
  const [editingDimensionIndex, setEditingDimensionIndex] = useState(null);
  const [dimensionDropdownAnchor, setDimensionDropdownAnchor] = useState(null);
  const [isSingleDimensionEvaluating, setIsSingleDimensionEvaluating] = useState(false);
  const svgRef = useRef(null);
  const [ideasList, setIdeasList] = useState([]);
  const [userScoreCorrections, setUserScoreCorrections] = useState([]);
  const [pendingChange, setPendingChange] = useState(null);
  const [pendingMerge, setPendingMerge] = useState(null);
  // Track user drag targets for visualization
  const [userDragTargets, setUserDragTargets] = useState({});
  // *** 新增：用于放大被拖拽覆盖的目标节点 ***
  const [mergeTargetId, setMergeTargetId] = useState(null);
  // *** Fragment 节点相关状态 ***
  const [fragmentNodes, setFragmentNodes] = useState([]); // 存储所有 fragment 节点
  const [fragmentMenuState, setFragmentMenuState] = useState(null); // { x, y, text, parentNodeId }
  const [expandedFragmentId, setExpandedFragmentId] = useState(null); // 当前展开的 fragment ID
  const [dragHoverTarget, setDragHoverTarget] = useState(null); // 拖动时 hover 的目标节点 ID
  const [draggingNodeId, setDraggingNodeId] = useState(null); // 当前正在拖动的节点 ID (用于 z-index 提升)
  // *** 新增：拖动视觉反馈状态 ***
  const [dragVisualState, setDragVisualState] = useState(null); // { type: 'modify'|'merge', sourceNodeId, targetNodeId?, ghostPosition? }
  const cubeDragRef = useRef(null); // 记录立方体自由旋转的拖拽起点
  const snapTimerRef = useRef(null);
  const [cubeRotation, setCubeRotation] = useState({ yaw: 0, pitch: 0 }); // yaw: 绕Y轴, pitch: 绕X轴
  const [isSnapping, setIsSnapping] = useState(false);
  const [isAddingCustom, setIsAddingCustom] = useState(false);
  const [customIdea, setCustomIdea] = useState({ title: '', content: '' });
  const [customIdeaCounter, setCustomIdeaCounter] = useState(1);

  // Initialize custom idea counter based on existing custom ideas
  useEffect(() => {
    const existingCustomIdeas = nodes.filter(node => /^C-\d+$/.test(node.id));
    if (existingCustomIdeas.length > 0) {
      const maxNumber = Math.max(...existingCustomIdeas.map(node => {
        const match = node.id.match(/^C-(\d+)$/);
        return match ? parseInt(match[1]) : 0;
      }));
      setCustomIdeaCounter(maxNumber + 1);
    }
  }, [nodes]);
  // *** 新增：用于主界面模型选择和api-key输入
  const [selectedModel, setSelectedModel] = useState('gpt-5-mini');
  const [apiKey, setApiKey] = useState('');
  const [isConfigured, setIsConfigured] = useState(false);
  const [configError, setConfigError] = useState('');
  const didClearSessionRef = useRef(false);
  // Clear server session on page load to avoid stale ideas after refresh.
  useEffect(() => {
    if (didClearSessionRef.current) return;
    didClearSessionRef.current = true;

    const shouldClearSession = process.env.NODE_ENV === 'production' || process.env.REACT_APP_CLEAR_SESSION_ON_LOAD === 'true';
    if (!shouldClearSession) return;

    const clearSession = async () => {
      try {
        await fetch('/api/clear-session', {
          method: 'POST',
          credentials: 'include',
        });
        setIsConfigured(false);
      } catch (err) {
        console.error('[WARN] Failed to clear session on load:', err);
      }
    };
    clearSession();
  }, []);
  // *** 新增：用于用户自定义prompts
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isEditingSystemPrompt, setIsEditingSystemPrompt] = useState(false);
  const [defaultPrompts, setDefaultPrompts] = useState({
    system_prompt: '',
  });
  // Add refs for positioning modals
  const systemPromptButtonRef = useRef(null);
  const [modalAnchorEl, setModalAnchorEl] = useState(null);
  const dashboardContainerRef = useRef(null);
  const analysisFormRef = useRef(null);
  const evaluationContainerRef = useRef(null); // Ref for plot view container
  const explorationContainerRef = useRef(null); // Ref for tree view container
  const fragmentDragOriginRef = useRef(null);

  // Node merging state
  const [mergeMode, setMergeMode] = useState({
    active: false,
    firstNode: null,
    secondNode: null,
    cursorPosition: { x: 0, y: 0 },
    showDialog: false
  });
  const [mergeAnimationState, setMergeAnimationState] = useState(null); // Persist merge visuals while backend merges

  // Toggle for hiding plot view
  const [hideEvaluationView] = useState(false);

  // ============== Workflow state (Code → Write → Review) ==============
  const [workflowIdea, setWorkflowIdea] = useState(null); // node being processed
  const [workflowStep, setWorkflowStep] = useState(null); // 'coding'|'code_done'|'code_error'|'writing'|'paper_done'|'paper_error'|'reviewing'|'review_done'|'review_error'
  const [workflowError, setWorkflowError] = useState(null);
  const [codeResult, setCodeResult] = useState(null);
  const [codeFiles, setCodeFiles] = useState([]); // [{name, path}] list of files in experiment_dir
  const [selectedCodeFile, setSelectedCodeFile] = useState(null); // {name, path, content}
  const [showWriterPrompt, setShowWriterPrompt] = useState(false);
  const [paperResult, setPaperResult] = useState(null);
  const [reviewResult, setReviewResult] = useState(null);
  const [s2ApiKey, setS2ApiKey] = useState('');

  // Helper function to create modifications array from score change
  const createModificationFromScoreChange = (nodeId, metric, previousScore, newScore) => {
    return [{
      metric,
      previousScore,
      newScore,
      change: newScore - previousScore
    }];
  };

  const normalizeIdeaOriginalData = useCallback((source, fallback = {}) => {
    const raw = source && typeof source === 'object' ? source : {};
    const title = raw.Title || raw.title || raw.Name || raw.name || fallback.title;
    const name = raw.Name || raw.name || raw.Title || raw.title || fallback.title;
    const problem = raw.Problem || raw.problem || raw.content || fallback.content;
    const normalized = { ...raw };
    const id = raw.id || raw.ID || fallback.id;

    if (id) normalized.id = id;
    if (title) normalized.Title = title;
    if (name) normalized.Name = name;
    if (problem) normalized.Problem = problem;
    if (raw.problem_highlights || fallback.problemHighlights) {
      normalized.problem_highlights = raw.problem_highlights || fallback.problemHighlights;
    }

    delete normalized.title;
    delete normalized.content;
    delete normalized.originalData;
    delete normalized.problemHighlights;

    return normalized;
  }, []);

  const colorMap = COLOR_MAP;

  // Unified node color helper to keep Exploration/Cube Views consistent
  const getNodeColor = useCallback((node) => {
    if (!node) return '#FF6B6B';
    if (node.isMergedResult) return '#B22222';
    if (node.isNewlyGenerated || node.isModified) return '#FFD700';
    return colorMap[node.type] || '#FF6B6B';
  }, [colorMap]);

  // Helper function to create ghost node with modified score
  const createGhostNodeWithModifiedScore = (originalNode, metric, newScore) => {
    const ghostId = `${originalNode.id}-Xghost-${Date.now()}`;
    const nextScores = {
      ...(originalNode.scores || {}),
      [metric]: newScore
    };
    const ghostNode = {
      ...originalNode,
      id: ghostId,
      level: originalNode.level + 1, // Put on next level
      scores: nextScores,
      originalData: originalNode.originalData
        ? {
          ...originalNode.originalData,
          scores: {
            ...(originalNode.originalData.scores || {}),
            [metric]: newScore
          }
        }
        : originalNode.originalData,
      isGhost: true,
      isModified: true, // Mark as modified for different color
      isNewlyGenerated: true, // Use same yellow color as plot view
      x: originalNode.x + Math.random() * 40 - 20, // Random offset like child nodes
      y: originalNode.y + 150 + Math.random() * 20 - 10 // Position below like child nodes
    };
    return ghostNode;
  };

  // Score modification handler for tree view
  const handleScoreModification = (nodeId, metric, previousScore, newScore, screenX, screenY) => {
    const originalNode = nodes.find(n => n.id === nodeId);
    if (!originalNode) return;

    // Create modifications array
    const modifications = createModificationFromScoreChange(nodeId, metric, previousScore, newScore);

    // Create ghost node with modified score
    const ghostNode = createGhostNodeWithModifiedScore(originalNode, metric, newScore);

    // Add ghost node to nodes
    setNodes(prev => [...prev, ghostNode]);
    setLinks(prev => [...prev, { source: originalNode.id, target: ghostNode.id }]);

    // Set pending change to trigger the 3-option modal positioned beneath the score
    setPendingChange({
      originalNode,
      ghostNode,
      modifications,
      behindNode: null, // Not needed for score modifications
      screenX: screenX || window.innerWidth / 2,
      screenY: screenY || window.innerHeight / 2
    });
  };

  const applyModifyEvaluation = useCallback((change) => {
    if (!change || !change.originalNode || !change.modifications?.length) {
      return;
    }

    const { originalNode, ghostNode, modifications } = change;

    const corrections = modifications
      .filter(mod => Math.abs(mod.change) > 5)
      .map(mod => ({
        ideaTitle: originalNode.title,
        ideaId: originalNode.id,
        metric: mod.metric,
        previousScore: mod.previousScore,
        newScore: mod.newScore,
        change: mod.change,
        timestamp: new Date().toISOString()
      }));

    if (corrections.length > 0) {
      setUserScoreCorrections(prev => [...prev, ...corrections]);
    }

    setNodes(prevNodes => {
      const filteredNodes = ghostNode
        ? prevNodes.filter(node => node.id !== ghostNode.id)
        : [...prevNodes];

      const updatedNodes = filteredNodes.map(node => {
        if (node.id !== originalNode.id) {
          return node;
        }

        const updatedNode = {
          ...node,
          // 更新 scores 对象（新的动态维度系统）
          scores: { ...(node.scores || {}) },
          originalData: node.originalData ? { ...node.originalData } : node.originalData
        };

        modifications.forEach(mod => {
          // 更新动态维度分数（scores 对象）
          if (updatedNode.scores) {
            updatedNode.scores[mod.metric] = mod.newScore;
          }

          if (updatedNode.originalData) {
            // 更新 originalData 中的 scores 对象
            if (!updatedNode.originalData.scores) {
              updatedNode.originalData.scores = {};
            }
            updatedNode.originalData.scores[mod.metric] = mod.newScore;
          }
        });

        return updatedNode;
      });

      return updatedNodes;
    });

    setIdeasList(prevIdeas => prevIdeas.map(idea => {
      if (idea.id !== originalNode.id) {
        return idea;
      }

      const updatedIdea = {
        ...idea,
        // 更新 scores 对象
        scores: { ...(idea.scores || {}) },
        originalData: idea.originalData ? { ...idea.originalData } : idea.originalData
      };

      modifications.forEach(mod => {
        // 更新动态维度分数
        if (updatedIdea.scores) {
          updatedIdea.scores[mod.metric] = mod.newScore;
        }

        if (updatedIdea.originalData) {
          // 更新 originalData 中的 scores 对象
          if (!updatedIdea.originalData.scores) {
            updatedIdea.originalData.scores = {};
          }
          updatedIdea.originalData.scores[mod.metric] = mod.newScore;
        }
      });

      return updatedIdea;
    }));

    if (ghostNode) {
      setLinks(prev => prev.filter(link => link.target !== ghostNode.id));
    }

    // 清理拖动视觉状态
    setDragVisualState(null);
    setPendingChange(null);
  }, [setIdeasList, setLinks, setPendingChange, setNodes, setUserScoreCorrections]);

  // Clear drag visual state when newly generated node becomes visible (isGhost: false)
  // Use ref to track if we've already cleared for a specific node to avoid repeated clears
  const clearedNodesRef = useRef(new Set());

  useEffect(() => {
    if (!dragVisualState) return;

    // Only clear if we have a pending operation and that specific node becomes visible
    if (dragVisualState.type === 'modify' && dragVisualState.sourceNodeId) {
      // Check if there's a newly generated node related to this drag operation
      const relatedNode = nodes.find(n =>
        n.isNewlyGenerated &&
        !n.isGhost &&
        n.isModified &&
        // Match by checking if this node was modified from the source
        (n.previousState?.id === dragVisualState.sourceNodeId || n.id.includes(dragVisualState.sourceNodeId))
      );
      if (relatedNode && !clearedNodesRef.current.has(relatedNode.id)) {
        console.log('[DEBUG] Related modified node visible, clearing drag visual state:', relatedNode.id);
        clearedNodesRef.current.add(relatedNode.id);
        setDragVisualState(null);
      }
    } else if (dragVisualState.type === 'merge' && (dragVisualState.sourceNodeId || dragVisualState.targetNodeId)) {
      // Check if merged result related to this specific merge is visible
      const relatedMerge = nodes.find(n =>
        n.isMergedResult &&
        !n.isGhost &&
        n.isNewlyGenerated &&
        n.parentIds &&
        (n.parentIds.includes(dragVisualState.sourceNodeId) || n.parentIds.includes(dragVisualState.targetNodeId))
      );
      if (relatedMerge && !clearedNodesRef.current.has(relatedMerge.id)) {
        console.log('[DEBUG] Related merged node visible, clearing drag visual state:', relatedMerge.id);
        clearedNodesRef.current.add(relatedMerge.id);
        setDragVisualState(null);
      }
    }
  }, [nodes, dragVisualState]);

  // Initialize tracking hooks

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

  // 清理旋转吸附计时器
  useEffect(() => {
    return () => {
      if (snapTimerRef.current) {
        clearTimeout(snapTimerRef.current);
      }
    };
  }, []);

  // Evaluation metric scaling via external control lines (one per axis) with movable block (Option A1 refined)
  // Control block at center (0.5) => scale=1.0.
  const [xScalingCenter, setXScalingCenter] = useState(50); // Center point for X-axis scaling (0-100)
  const [yScalingCenter, setYScalingCenter] = useState(50); // Center point for Y-axis scaling (0-100)

  // Unified scale control for both axes (replaces individual xScaleControl/yScaleControl)
  const [unifiedScale, setUnifiedScale] = useState(1.0); // 1.0 = normal, >1 = zoomed in, <1 = zoomed out
  const [userHasInteractedWithScale, setUserHasInteractedWithScale] = useState(false); // Track if user has manually adjusted scale

  // ============== Fragment 节点相关函数 ==============
  const clearSelection = useCallback(() => {
    const selection = window.getSelection && window.getSelection();
    if (selection && selection.removeAllRanges) {
      selection.removeAllRanges();
    }
  }, []);

  /**
   * 隐藏 Fragment 菜单
   */
  const hideFragmentMenu = useCallback(() => {
    setFragmentMenuState(null);
    clearSelection();
  }, [clearSelection]);

  /**
   * 显示 Fragment 菜单
   */
  const showFragmentMenu = useCallback((x, y, selectedText, parentNodeId) => {
    setFragmentMenuState({ x, y, text: selectedText, parentNodeId });
  }, []);

  /**
   * 创建 Fragment 节点
   * @param {string} selectedText - 用户选中的文本
   * @param {string} parentNodeId - 父节点 ID
   */
  const createFragmentNode = useCallback((selectedText, parentNodeId) => {
    const cleanText = (selectedText || '').trim();
    if (!cleanText || !parentNodeId) return;

    // 计算当前父节点已有的 Fragment 数量
    const existingFragments = fragmentNodes.filter(fn =>
      fn.id.startsWith(parentNodeId + '-S')
    );
    const nextIndex = existingFragments.length + 1;
    const fragmentId = `${parentNodeId}-S${nextIndex}`;

    // 创建 Fragment 节点（与正常节点结构一样，但无评分）
    const fragmentNode = {
      id: fragmentId,
      title: cleanText.length > 50 ? cleanText.substring(0, 50) + '...' : cleanText,
      content: cleanText,
      type: 'fragment',
      parentId: parentNodeId,
      timestamp: Date.now(),
      // 无评分字段
    };

    setFragmentNodes(prev => [...prev, fragmentNode]);

  }, [fragmentNodes]);

  /**
   * 确认创建 Fragment 节点
   */
  const handleFragmentConfirm = useCallback(() => {
    if (!fragmentMenuState) return;

    createFragmentNode(fragmentMenuState.text, fragmentMenuState.parentNodeId);
    hideFragmentMenu();
    clearSelection();
  }, [fragmentMenuState, createFragmentNode, hideFragmentMenu, clearSelection]);

  /**
   * 删除 Fragment 节点
   */
  const deleteFragmentNode = useCallback((fragmentId) => {
    setFragmentNodes(prev => prev.filter(fn => fn.id !== fragmentId));

  }, []);

  // 点击外部关闭 Fragment 菜单
  useEffect(() => {
    if (!fragmentMenuState) return;

    const handleClickOutside = (e) => {
      // 检查点击是否在 Fragment 菜单外部
      const target = e.target;
      const isMenuClick = target.closest('[data-fragment-menu]');

      if (!isMenuClick) {
        hideFragmentMenu();
      }
    };

    // 延迟添加监听器，避免立即触发
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [fragmentMenuState, hideFragmentMenu]);

  // Fallback merge ID (matches backend _merge_id logic)
  const createMergeId = useCallback((nodeAId, nodeBId) => `${nodeAId}-Y-${nodeBId}-Y`, []);

  // Helper function to extract comprehensive node information for tracking
  const getNodeTrackingInfo = (node) => {
    return {
      nodeId: node.id,
      nodeTitle: node.title,
      nodeContent: buildNodeContent(node)
    };
  };

  // ============== 新增：辅助函数获取节点在动态维度系统下的分数 ==============
  /**
   * 获取节点在指定维度对上的分数
   * @param {Object} node - 节点对象
   * @param {Object} dimensionPair - 维度对对象 { dimensionA, dimensionB }
   * @returns {number|undefined} - 分数值 (0-100)，如果不存在则返回 undefined
   */
  const getNodeDimensionScore = useCallback((node, dimensionPair) => {
    if (!node || !dimensionPair) return undefined;

    const extractScoreValue = (scoreEntry) => {
      if (scoreEntry === null || scoreEntry === undefined) return undefined;
      if (typeof scoreEntry === 'number') return scoreEntry;
      if (typeof scoreEntry === 'object' && scoreEntry.value !== undefined) return scoreEntry.value;
      return undefined;
    };

    const orientScoreValue = (value, flipped) => {
      if (!flipped || typeof value !== 'number') return value;
      if (value >= -50 && value <= 50) return -value;
      if (value >= 0 && value <= 100) return 100 - value;
      return -value;
    };

    const getScoreEntry = (key) => {
      if (!key) return undefined;
      if (node.scores && node.scores[key] !== undefined) return node.scores[key];
      if (node.originalData && node.originalData.scores && node.originalData.scores[key] !== undefined) {
        return node.originalData.scores[key];
      }
      return undefined;
    };

    // 新系统：从 node.scores 对象中获取
    const key = `${dimensionPair.dimensionA}-${dimensionPair.dimensionB}`;
    const primaryValue = extractScoreValue(getScoreEntry(key));
    if (primaryValue !== undefined) {
      return primaryValue;
    }
    // Fallback：如果用户交换了维度方向，尝试反向 key，并翻转符号
    const reverseKey = `${dimensionPair.dimensionB}-${dimensionPair.dimensionA}`;
    const reverseValue = extractScoreValue(getScoreEntry(reverseKey));
    if (reverseValue !== undefined) {
      return orientScoreValue(reverseValue, true);
    }

    return undefined;
  }, []);

  /**
   * 获取当前选中的维度对（X轴和Y轴）
   * @returns {{ xDimension: Object|null, yDimension: Object|null }}
   */
  const getCurrentDimensions = useCallback((faceIndexOverride = currentFaceIndex) => {
    if (!selectedDimensionPairs || selectedDimensionPairs.length === 0) {
      return { xDimension: null, yDimension: null, faceIndex: 0, faceName: '', xFlip: false, yFlip: false };
    }

    // Cube faces: 0 (Front), 1 (Right), 2 (Top), 3 (Back), 4 (Left), 5 (Bottom)
    if (selectedDimensionPairs.length >= 3) {
      const makeFace = (xPair, yPair, idx, isBack = false) => ({
        xDimension: xPair,
        yDimension: yPair,
        faceIndex: idx,
        faceName: `Face ${idx}: ${xPair.dimensionA}-${xPair.dimensionB} vs ${yPair.dimensionA}-${yPair.dimensionB}${isBack ? ' (Back)' : ''}`,
        xFlip: isBack, // Use xFlip only for coordinate inversion, NOT for text mirroring
        yFlip: false   // Never flip Y for now, unless we want upside down
      });

      // Face 0 (Front): Pair 0 vs Pair 1
      // Face 1 (Right): Pair 1 vs Pair 2
      // Face 2 (Top): Pair 2 vs Pair 0
      // Face 3 (Back): Pair 0 vs Pair 1 (inverted X)
      // Face 4 (Left): Pair 1 vs Pair 2 (inverted X)
      // Face 5 (Bottom): Pair 2 vs Pair 0 (inverted X)

      const faces = [
        makeFace(selectedDimensionPairs[0], selectedDimensionPairs[1], 0, false),
        makeFace(selectedDimensionPairs[1], selectedDimensionPairs[2], 1, false),
        makeFace(selectedDimensionPairs[2], selectedDimensionPairs[0], 2, false),
        makeFace(selectedDimensionPairs[0], selectedDimensionPairs[1], 3, true),
        makeFace(selectedDimensionPairs[1], selectedDimensionPairs[2], 4, true),
        makeFace(selectedDimensionPairs[2], selectedDimensionPairs[0], 5, true)];

      const idx = ((faceIndexOverride % faces.length) + faces.length) % faces.length;
      return faces[idx];
    }

    // Fallback: only two pairs selected
    if (selectedDimensionPairs.length === 2) {
      return {
        xDimension: selectedDimensionPairs[0],
        yDimension: selectedDimensionPairs[1],
        faceIndex: 0,
        faceName: `Face 0: ${selectedDimensionPairs[0].dimensionA}-${selectedDimensionPairs[0].dimensionB} vs ${selectedDimensionPairs[1].dimensionA}-${selectedDimensionPairs[1].dimensionB}`,
        xFlip: false,
        yFlip: false
      };
    }

    // Single pair fallback
    return {
      xDimension: selectedDimensionPairs[0],
      yDimension: selectedDimensionPairs[0],
      faceIndex: 0,
      faceName: `Face 0: ${selectedDimensionPairs[0].dimensionA}-${selectedDimensionPairs[0].dimensionB}`,
      xFlip: false,
      yFlip: false
    };
  }, [currentFaceIndex, selectedDimensionPairs]);

  const { xDimension: activeXDimension, yDimension: activeYDimension } = getCurrentDimensions();
  const xAxisMetric = activeXDimension ? `${activeXDimension.dimensionA}-${activeXDimension.dimensionB}` : '';
  const yAxisMetric = activeYDimension ? `${activeYDimension.dimensionA}-${activeYDimension.dimensionB}` : '';
  const xAxisLabel = activeXDimension ? `${activeXDimension.dimensionA} vs ${activeXDimension.dimensionB}` : '';
  const yAxisLabel = activeYDimension ? `${activeYDimension.dimensionA} vs ${activeYDimension.dimensionB}` : '';

  useEffect(() => {
    setUserDragTargets(prev => {
      const filtered = {};
      Object.entries(prev).forEach(([nodeId, target]) => {
        if (target.xAxisMetric === xAxisMetric && target.yAxisMetric === yAxisMetric) {
          filtered[nodeId] = target;
        }
      });
      return filtered;
    });
  }, [xAxisMetric, yAxisMetric]);

  /**
   * 获取节点的 X 坐标值（基于当前选中的维度对）
   * @param {Object} node - 节点对象
   * @returns {number|undefined} - X 坐标值
   */
  const getNodeX = useCallback((node) => {
    const { xDimension } = getCurrentDimensions();
    if (xDimension) {
      return getNodeDimensionScore(node, xDimension);
    }
    return undefined;
  }, [getCurrentDimensions, getNodeDimensionScore]);

  /**
   * 获取节点的 Y 坐标值（基于当前选中的维度对）
   * @param {Object} node - 节点对象
   * @returns {number|undefined} - Y 坐标值
   */
  const getNodeY = useCallback((node) => {
    const { yDimension } = getCurrentDimensions();
    if (yDimension) {
      return getNodeDimensionScore(node, yDimension);
    }
    return undefined;
  }, [getCurrentDimensions, getNodeDimensionScore]);

  // Reset face to the first plane when dimension selection changes

  const calculateNodeMidpoints = useCallback(() => {
    const visibleNodes = nodes.filter(n => {
      const xVal = getNodeX(n);
      const yVal = getNodeY(n);
      return xVal !== undefined && yVal !== undefined && !n.isGhost;
    });

    if (visibleNodes.length === 0) {
      return { xCenter: 70, yCenter: 70 };
    }

    const xValues = visibleNodes.map(n => getNodeX(n));
    const yValues = visibleNodes.map(n => getNodeY(n));

    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const xCenter = (xMin + xMax) / 2;
    const yCenter = (yMin + yMax) / 2;

    return { xCenter, yCenter };
  }, [nodes, getNodeX, getNodeY]);


  /**
   * 从象限内的像素坐标反向计算分数
   * @param {number} pixelX - 象限内的 X 像素坐标
   * @param {number} pixelY - 象限内的 Y 像素坐标
   * @param {number} quadrantWidth - 象限宽度
   * @param {number} quadrantHeight - 象限高度
   * @param {number} xMin - X 轴最小分数
   * @param {number} xMax - X 轴最大分数
   * @param {number} yMin - Y 轴最小分数
   * @param {number} yMax - Y 轴最大分数
   * @returns {{ xScore: number, yScore: number }} - 计算出的分数 (0-100)
   */
  const pixelToScore = (pixelX, pixelY, quadrantWidth, quadrantHeight, xMin, xMax, yMin, yMax) => {
    // 反向计算：从90%空间映射回去（对应正向映射的90%容器空间）
    const rawX = (pixelX - quadrantWidth * 0.05) / (quadrantWidth * 0.9);
    const rawY = (pixelY - quadrantHeight * 0.05) / (quadrantHeight * 0.9);

    // 将像素坐标归一化到 0-1 范围
    const normalizedX = Math.max(0, Math.min(1, rawX));
    const normalizedY = Math.max(0, Math.min(1, 1 - rawY)); // Y 轴翻转

    // 映射到分数区间 (支持 signed ranges e.g., -50..50)
    const xScoreRaw = xMin + normalizedX * (xMax - xMin);
    const yScoreRaw = yMin + normalizedY * (yMax - yMin);

    // 四舍五入并在提供的区间内截断
    const xScore = Math.round(xScoreRaw);
    const yScore = Math.round(yScoreRaw);

    const clamp = (v, a, b) => Math.max(Math.min(v, Math.max(a, b)), Math.min(a, b));

    return {
      xScore: clamp(xScore, xMin, xMax),
      yScore: clamp(yScore, yMin, yMax)
    };
  };

  /**
   * 检测拖动节点是否与其他节点碰撞(用于 Merge 检测)
   * @param {number} dragX - 拖动节点中心的屏幕 X 坐标
   * @param {number} dragY - 拖动节点中心的屏幕 Y 坐标
   * @param {string} dragNodeId - 拖动节点的 ID
   * @param {Array} allNodes - 所有节点列表
   * @param {number} threshold - 碰撞检测阈值(像素)
   * @returns {Object|null} - 碰撞的目标节点,如果没有碰撞返回 null
   */
  const detectNodeCollision = (dragX, dragY, dragNodeId, allNodes, threshold = 30) => {
    for (const node of allNodes) {
      if (node.id === dragNodeId) continue; // 跳过自己

      const nodeElement = document.querySelector(`[data-node-id="${node.id}"]`);
      if (!nodeElement) continue;

      const rect = nodeElement.getBoundingClientRect();
      const nodeCenterX = rect.left + rect.width / 2;
      const nodeCenterY = rect.top + rect.height / 2;

      const distance = Math.sqrt(
        Math.pow(dragX - nodeCenterX, 2) +
        Math.pow(dragY - nodeCenterY, 2)
      );

      if (distance < threshold) {
        return node;
      }
    }
    return null;
  };

  // 获取节点中心相对于容器的坐标（用于在 merge 过程中维持视觉状态）
  const getNodeCenterRelativeToContainer = useCallback((nodeId) => {
    const container = (currentView === 'evaluation' ? evaluationContainerRef : explorationContainerRef).current;
    const nodeElement = document.querySelector(`[data-node-id="${nodeId}"]`);
    if (!container || !nodeElement) return null;
    const containerRect = container.getBoundingClientRect();
    const rect = nodeElement.getBoundingClientRect();
    return {
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top + rect.height / 2
    };
  }, [currentView]);
  // ============== 配置模型和API Key ==============
  const modelOptions = [
    { value: 'gpt-5.2', label: 'GPT-5.2' },
    { value: 'gpt-5.2-pro', label: 'GPT-5.2 Pro' },
    { value: 'gpt-5.2-codex', label: 'GPT-5.2 Codex' },
    { value: 'gpt-5', label: 'GPT-5' },
    { value: 'gpt-5-pro', label: 'GPT-5 Pro' },
    { value: 'gpt-5-mini', label: 'GPT-5 Mini' },
    { value: 'gpt-5-nano', label: 'GPT-5 Nano' },
    { value: 'gpt-4.1', label: 'GPT-4.1' },
    { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini' },
    { value: 'gpt-4.1-nano', label: 'GPT-4.1 Nano' },
    { value: 'o3', label: 'O3' },
    { value: 'o4-mini-deep-research', label: 'O4 Mini Deep Research' },
    { value: 'claude-opus-4-6', label: 'Claude Opus 4.6' },
    { value: 'claude-opus-4-5', label: 'Claude Opus 4.5' },
    { value: 'claude-sonnet-4-5', label: 'Claude Sonnet 4.5' },
    { value: 'claude-haiku-4-5', label: 'Claude Haiku 4.5' },
    { value: 'claude-opus-4', label: 'Claude Opus 4' },
    { value: 'claude-sonnet-4', label: 'Claude Sonnet 4' },
    { value: 'deepseek-chat', label: 'DeepSeek Chat' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner' },
    { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite' },
    { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' }];
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
      setCurrentView('exploration'); // Auto-switch to tree view

    } catch (err) {
      console.error('Configuration error:', err);
      setConfigError(err.message);
      setOperationStatus('');
    }
  };

  // Overview Component
  const OverviewPage = () => (
    <div
      data-uatrack-suppress-hover="true"
      data-uatrack-suppress-click="true"
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
          data-panel-root="edit-modal"
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
  // ============== 段落3：评估假设（evaluateIdeas） ==============
  // Helper: determine which ideas need evaluation (missing scores)
  const getIdeasNeedingEvaluation = useCallback((ideasArr) => {
    return ideasArr.filter(i => {
      if (!selectedDimensionPairs || selectedDimensionPairs.length < 3) {
        return true;
      }

      const hasAll = selectedDimensionPairs.slice(0, 3).every(pair => {
        const val = getNodeDimensionScore(i, pair);
        return val !== undefined && val !== null;
      });

      return !hasAll;
    }).map(i => i.id);
  }, [getNodeDimensionScore, selectedDimensionPairs]);

  const evaluateIdeas = async (ideas, { mode = 'incremental', allowAutoCenter = false, dimensionPairs = null } = {}) => {

    if (!ideas || !Array.isArray(ideas) || ideas.length === 0) {
      return;
    }

    // Don't clear drag targets at start - they should persist during evaluation

    setIsEvaluating(true);
    setOperationStatus('Evaluating ideas...');
    setError(null);

    try {
      // Prepare the request body - ensuring merged ideas use their originalData properly
      console.log('[DEBUG] Before filtering ideas:', {
        totalIdeas: ideas.length,
        ideaStructure: ideas.map(h => ({
          id: h.id,
          title: h.title,
          hasOriginalData: !!h.originalData,
          originalDataFields: h.originalData ? Object.keys(h.originalData) : []
        }))
      });

      // Log each idea separately for easier debugging
      ideas.forEach((idea, index) => {
        console.log(`[DEBUG] Idea ${index}:`, {
          id: idea.id,
          title: idea.title,
          hasOriginalData: !!idea.originalData,
          originalData: idea.originalData
        });
      });

      const requestIdeas = ideas
        .filter(h => {
          // Exclude root node (level 0) from evaluation
          if (h.level === 0) {
            console.log('[DEBUG] Excluding root node from evaluation:', h.id);
            return false;
          }
          // For ideas from ideasList, they are already the original data
          // For ideas from nodes, they need originalData
          return h.originalData || (h.id && h.title && !h.level); // level indicates it's a node, not an idea
        })
        .map(h => {
          if (h.originalData) {
            // This is a node with originalData
            const od = h.originalData;
            return {
              ...od,
              id: od.id || h.id,
              Title: od.Title || od.Name || h.title || h.name || '',
              Name: od.Name || od.Title || h.title || h.name || ''
            };
          } else {
            // This is already an idea from ideasList
            return {
              ...h,
              Title: h.Title || h.Name || h.title || h.name || '',
              Name: h.Name || h.Title || h.title || h.name || ''
            };
          }
        });

      console.log('[DEBUG] After filtering and mapping:', {
        requestIdeasCount: requestIdeas.length,
        requestIdeaIds: requestIdeas.map(i => i.id),
        requestIdeaTitles: requestIdeas.map(i => i.Title || i.Name)
      });

      // Use selectedDimensionPairs if provided via parameter, otherwise use component state
      const effectiveDimensionPairs = dimensionPairs || selectedDimensionPairs;

      if (!effectiveDimensionPairs || effectiveDimensionPairs.length < 3) {
        throw new Error('Please select 3 dimension pairs before evaluating ideas.');
      }

      const requestBody = {
        ideas: requestIdeas,
        intent: analysisIntent,
        userScoreCorrections: userScoreCorrections,
        mode,
        // Only send explicit targetIds when incremental; backend will infer otherwise.
        targetIds: mode === 'incremental' ? getIdeasNeedingEvaluation(requestIdeas) : undefined,
        // Add dimension_pairs to request if available
        dimension_pairs: effectiveDimensionPairs
      };


      // 1. Make the API call.
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 2. Wait for the full response body to arrive as text.
      const responseText = await response.text();
      console.log('[DEBUG] Evaluation response text length:', responseText.length);
      console.log('[DEBUG] Evaluation response text preview:', responseText.substring(0, 1000));
      console.log('[DEBUG] Evaluation response text end:', responseText.substring(responseText.length - 500));

      // 3. Parse the text string into JSON.
      const parsed = JSON.parse(responseText);
      let evaluatedIdeas = [];
      let metaMode = mode;
      if (Array.isArray(parsed)) {
        evaluatedIdeas = parsed; // legacy array shape
      } else if (parsed.ideas) {
        evaluatedIdeas = parsed.ideas; // new shape {ideas, meta}
        if (parsed.meta && parsed.meta.mode) metaMode = parsed.meta.mode;
      } else if (parsed.scores) {
        evaluatedIdeas = parsed.scores; // older shape
      }

      // Debug: Log evaluation response structure
      console.log('[DEBUG] Parsed evaluation response:', {
        evaluatedIdeasCount: evaluatedIdeas.length,
        evaluatedIdeas: evaluatedIdeas.map(ev => ({
          id: ev.id,
          title: ev.title,
          hasScores: !!(ev.scores && Object.keys(ev.scores).length > 0),
          hasReasoning: !!(ev.Dimension1Reason || ev.Dimension2Reason || ev.Dimension3Reason),
          // Debug: Show all keys to see what's actually available
          allKeys: Object.keys(ev),
          // Debug: Check for any reasoning-related fields
          reasoningFields: Object.keys(ev).filter(key => key.toLowerCase().includes('reason'))
        }))
      });

      // Debug: Log the raw evaluatedIdeas for custom ideas
      const customIdeas = evaluatedIdeas.filter(ev => ev.id && (ev.id.includes('-CUSTOM-') || /^C-\d+$/.test(ev.id)));
      if (customIdeas.length > 0) {
        console.log('[DEBUG] Custom ideas in evaluation response:', customIdeas.map(ev => ({
          id: ev.id,
          title: ev.title,
          allKeys: Object.keys(ev),
          reasoningFields: Object.keys(ev).filter(key => key.toLowerCase().includes('reason')),
          rawData: ev
        })));
      }

      // 4. Check if the backend returned a specific error message.
      if (evaluatedIdeas.error) {
        throw new Error(evaluatedIdeas.error);
      }


      // 5. Success! The data is ready. Log it and update the state.
      console.log("Evaluation successful:", evaluatedIdeas);

      // This logic now correctly runs inside the same block where 'evaluatedIdeas' is defined.
      setNodes((prevNodes) => {
        // Build map keyed by backend idea id (each evaluated idea.id is backend id)
        const evalMap = new Map(evaluatedIdeas.map(h => [h.id, h]));
        const existingBackendIds = new Set(prevNodes.map(n => n.originalData && n.originalData.id).filter(Boolean));

        console.log('[DEBUG] Frontend evaluation mapping:', {
          evaluatedIdeas: evaluatedIdeas.map(e => ({
            id: e.id,
            title: e.title,
            hasScores: !!(e.scores && Object.keys(e.scores).length > 0)
          })),
          nodes: prevNodes.map(n => ({
            nodeId: n.id,
            backendId: n.originalData?.id,
            title: n.title,
            hasExistingScores: !!(n.scores && Object.keys(n.scores).length > 0)
          }))
        });

        const updatedNodes = prevNodes.map(node => {
          const backendId = node.originalData && (node.originalData.id || node.originalData.ID);
          let ev = backendId ? evalMap.get(backendId) : evalMap.get(node.id);

          // Additional fallback: try finding by title if ID doesn't match
          if (!ev && node.title) {
            ev = evaluatedIdeas.find(e =>
              e.title === node.title ||
              e.Title === node.title ||
              (e.title && e.title.toLowerCase() === node.title.toLowerCase())
            );
          }

          if (!ev) {
            console.log(`[DEBUG] No evaluation match for node: ${node.id} (backend: ${backendId}, title: "${node.title}")`);
            console.log(`[DEBUG] Available evaluations:`, evaluatedIdeas.map(e => ({ id: e.id, title: e.title })));
            return node;
          }

          console.log(`[DEBUG] Matched node ${node.id} with eval ${ev.id}, scores:`, JSON.stringify(ev.scores || {}));

          // Debug: Check if this is a custom idea and log reasoning fields
          const isCustomIdea = node.id && (node.id.includes('-CUSTOM-') || /^C-\d+$/.test(node.id));
          if (isCustomIdea) {
            console.log(`[DEBUG] Custom idea ${node.id} evaluation data:`, {
              Dimension1Reason: ev.Dimension1Reason,
              Dimension2Reason: ev.Dimension2Reason,
              Dimension3Reason: ev.Dimension3Reason,
              Dimension1Score: ev.Dimension1Score,
              Dimension2Score: ev.Dimension2Score,
              Dimension3Score: ev.Dimension3Score,
              DimensionReasons: ev.Dimension1Reason || ev.Dimension2Reason || ev.Dimension3Reason,
              hasDimension1Reason: !!ev.Dimension1Reason,
              hasDimension2Reason: !!ev.Dimension2Reason,
              hasDimension3Reason: !!ev.Dimension3Reason,
              // Debug: Show all available fields
              allEvKeys: Object.keys(ev),
              reasoningFields: Object.keys(ev).filter(key => key.toLowerCase().includes('reason'))
            });
          }

          // Calculate correct position based on actual scores (new or old system)
          const updatedNode = {
            ...node,
            // 新系统分数
            scores: ev.scores || node.scores,
            dimension1Score: ev.dimension1Score ?? node.dimension1Score,
            dimension2Score: ev.dimension2Score ?? node.dimension2Score,
            dimension3Score: ev.dimension3Score ?? node.dimension3Score,
            // Preserve original content and title from the node (don't overwrite with evaluation data)
            title: node.title || ev.title || ev.Title || ev.Name,
            content: node.content || ev.content || ev.Problem || '',
            problemHighlights: node.problemHighlights || node.originalData?.problem_highlights || ev.problem_highlights || ev.problemHighlights || [],
            // Update originalData to include reasoning fields for all ideas
            originalData: {
              ...(node.originalData || {}),
              // 新系统字段 - 维度评分原因
              Dimension1Reason: ev.Dimension1Reason || '',
              Dimension2Reason: ev.Dimension2Reason || '',
              Dimension3Reason: ev.Dimension3Reason || '',
              Dimension3Score: ev.Dimension3Score || ev.dimension3Score,
              scores: ev.scores || node.scores, // 重要: 也存到 originalData
            },
            // Explicitly preserve userDragTarget for visualization
            userDragTarget: node.userDragTarget,
            // Reveal ghost nodes when they receive scores (turn off ghost mode)
            isGhost: false,
            // Clear pending evaluation status
            isPendingEvaluation: false
          };

          // Debug: Log userDragTarget preservation
          if (node.userDragTarget) {
            console.log('[DEBUG] Preserved userDragTarget for node:', node.id, node.userDragTarget);
          }

          // If this was a ghost node, clear explicit coordinates to use score-based positioning
          if (node.isGhost) {
            console.log(`[DEBUG] Converting ghost node ${node.id} to scored node - clearing explicit coordinates`);
            // Clear explicit coordinates so the node positions based on its scores
            delete updatedNode.x;
            delete updatedNode.y;
            delete updatedNode._tmpViewX;
            delete updatedNode._tmpViewY;
          }

          // DON'T delete userDragTarget here - keep it for visualization until next evaluation
          // delete updatedNode.userDragTarget;

          return updatedNode;
        });

        // If full mode, ensure every evaluated idea has a node; if missing, append (edge safety)
        if (metaMode === 'full') {
          evaluatedIdeas.forEach(ev => {
            if (!existingBackendIds.has(ev.id)) {
              updatedNodes.push({
                id: ev.id, // fallback to backend id (no lineage info available)
                level: 1,
                title: ev.title || ev.Title || 'Untitled',
                content: ev.content || '',
                problemHighlights: ev.problem_highlights || ev.problemHighlights || [],
                type: 'complex',
                x: 0,
                y: 0,
                originalData: ev,
                // 新系统分数
                scores: ev.scores || {},
                dimension1Score: ev.dimension1Score,
                dimension2Score: ev.dimension2Score,
                dimension3Score: ev.dimension3Score,
              });
            }
          });
        }

        // Track evaluation updates

        // Auto-center after evaluation completion (only if user hasn't manually adjusted scale and auto-centering is allowed)
        if (!userHasInteractedWithScale && allowAutoCenter) {
          setTimeout(() => {
            // Keep xy scale at 0-100 for initial generation, don't auto-adjust to node midpoints
            setXScalingCenter(50);
            setYScalingCenter(50);
            console.log(`[DEBUG] Kept scale centered at 50,50 for stable 0-100 view`);
          }, 100);
        }

        return updatedNodes;
      });
    } catch (err) {
      console.error('Error evaluating ideas:', err);
      setError(err.message);
    } finally {
      setIsEvaluating(false);
      setOperationStatus('');

      // Clear temporary view coordinates so nodes use score-based positioning
      setNodes(prev => {
        const updatedNodes = prev.map(node => {
          const updatedNode = { ...node };
          delete updatedNode._tmpViewX;
          delete updatedNode._tmpViewY;
          return updatedNode;
        });

        return updatedNodes;
      });

      // Clear ALL drag targets at evaluation end, then regenerate only the most recent one
      console.log('[DEBUG] Clearing ALL drag targets at evaluation end');
      setUserDragTargets({});

      // Find the most recently modified node (highest timestamp)
      let mostRecentDragTarget = null;
      let mostRecentNodeId = null;
      let mostRecentTimestamp = 0;

      setNodes(currentNodes => {
        // First pass: find the most recent drag target
        currentNodes.forEach(node => {
          if (node.userDragTarget && node.userDragTarget.timestamp > mostRecentTimestamp) {
            mostRecentTimestamp = node.userDragTarget.timestamp;
            mostRecentDragTarget = node.userDragTarget;
            mostRecentNodeId = node.id;
          }
        });

        console.log('[DEBUG] Most recent drag target:', mostRecentNodeId, 'timestamp:', mostRecentTimestamp);

        // Second pass: clear ALL userDragTarget from ALL nodes
        const clearedNodes = currentNodes.map(node => {
          if (node.userDragTarget) {
            const cleanedNode = { ...node };
            delete cleanedNode.userDragTarget;
            console.log('[DEBUG] Cleared userDragTarget from node:', node.id);
            return cleanedNode;
          }
          return node;
        });

        return clearedNodes;
      });

      // Regenerate drag target only for the most recently modified node
      setTimeout(() => {
        if (mostRecentDragTarget && mostRecentNodeId) {
          console.log('[DEBUG] Restoring drag target only for most recent node:', mostRecentNodeId);

          // Restore userDragTarget only to the most recent node
          setNodes(currentNodes => currentNodes.map(node => {
            if (node.id === mostRecentNodeId) {
              console.log('[DEBUG] Restoring userDragTarget for node:', node.id);
              return {
                ...node,
                userDragTarget: mostRecentDragTarget
              };
            }
            return node;
          }));

          // Set visualization for only this node
          setUserDragTargets({
            [mostRecentNodeId]: mostRecentDragTarget
          });
        } else {
          console.log('[DEBUG] No recent drag targets to restore');
        }
      }, 50);
    }
  };

  const evaluateIdeasRef = useRef(null);
  evaluateIdeasRef.current = evaluateIdeas;

  // New: re-evaluate all ideas handler
  const reEvaluateAll = () => {
    if (isEvaluating || nodes.length === 0) return;
    evaluateIdeasRef.current(nodes, { mode: 'full', allowAutoCenter: true });
  };

  // ============== 段落4：分析意图提交处理 (handleAnalysisIntentSubmit) ==============
  const handleAnalysisIntentSubmit = async (e) => {
    e.preventDefault();
    if (!analysisIntent.trim()) return;


    if (!isConfigured) {
      setError('Please configure the model first');
      return;
    }

    // 保存 intent 并打开下拉面板让用户选择维度
    setCurrentIntent(analysisIntent);
    setShowDimensionPanel(true);
  };

  // ============== 新增：维度确认后生成 Ideas ==============
  const handleDimensionConfirm = async (dimensionPairs) => {
    setSelectedDimensionPairs(dimensionPairs);
    setCurrentFaceIndex(0); // Reset to front face
    setShowDimensionPanel(false); // 关闭面板

    setIsGenerating(true);
    setOperationStatus('Generating initial ideas...');
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
          num_ideas: 1,
          dimension_pairs: dimensionPairs // 传递维度对
        }),
      });
      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        throw new Error(`Failed to generate ideas: ${errorText}`);
      }

      // Wait for the full response as text (handles heartbeat stream)
      const responseText = await response.text();

      // Parse the text into JSON, ignoring heartbeats
      const data = JSON.parse(responseText);

      if (data.error) {
        throw new Error(data.error);
      }
      console.log("Received data from API:", data);

      const ideas = data.ideas;
      const ideasWithId = ideas; // backend already provides hierarchical ids
      const updatedIdeasList = [...ideasList, ...ideasWithId];
      setIdeasList(updatedIdeasList);

      // Determine if tree already exists (root already set up)
      const treeAlreadyExists = nodes.some(n => n.id === 'root');

      if (!treeAlreadyExists) {
        // First idea: create root node + first child
        const rootNode = {
          id: 'root',
          level: 0,
          title: analysisIntent,
          content: analysisIntent,
          type: 'root',
          x: 0,
          y: 0,
        };

        const childNodes = ideasWithId.map((hyp, i) => ({
          id: hyp.id,
          level: 1,
          title: hyp.title.trim(),
          content: hyp.content.trim(),
          type: 'complex',
          x: i * 200,
          y: 150,
          originalData: hyp.originalData,
          problemHighlights: hyp.problemHighlights || hyp.originalData?.problem_highlights || []
        }));

        setNodes([rootNode, ...childNodes]);
        setLinks(childNodes.map((nd) => ({ source: 'root', target: nd.id })));
      } else {
        // Subsequent ideas: add new child nodes connected to root
        const existingRootX = nodes.find(n => n.id === 'root')?.x ?? 0;
        const existingLevel1 = nodes.filter(n => n.level === 1);
        const nextX = existingLevel1.length > 0
          ? Math.max(...existingLevel1.map(n => n.x || 0)) + 200
          : existingRootX;

        const newChildNodes = ideasWithId.map((hyp, i) => ({
          id: hyp.id,
          level: 1,
          title: hyp.title.trim(),
          content: hyp.content.trim(),
          type: 'complex',
          x: nextX + i * 200,
          y: 150,
          originalData: hyp.originalData,
          problemHighlights: hyp.problemHighlights || hyp.originalData?.problem_highlights || []
        }));

        setNodes(prev => [...prev, ...newChildNodes]);
        setLinks(prev => [
          ...prev,
          ...newChildNodes.map(nd => ({ source: 'root', target: nd.id }))
        ]);
      }

      setIsAnalysisSubmitted(true);

      // Incremental evaluation: only score the new idea
      await evaluateIdeas(updatedIdeasList, { mode: 'incremental', allowAutoCenter: true, dimensionPairs });
      setIsGenerating(false);
      setOperationStatus('');
    } catch (err) {
      console.error('Error generating initial ideas:', err);
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
          parent_id: selectedNode.id,
          context: userInput
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate child ideas');
      }

      // Wait for the full response as text (handles heartbeat stream)
      const responseText = await response.text();

      // Parse the text into JSON, ignoring heartbeats
      const data = JSON.parse(responseText);

      if (data.error) {
        throw new Error(data.error);
      }
      const ideas = data.ideas;
      const newIdeasWithId = ideas; // backend already supplied hierarchical ids
      const updatedIdeasList = [...ideasList, ...newIdeasWithId];
      setIdeasList(updatedIdeasList);

      // 布局
      const childSpacing = 200;
      const totalWidth = (ideas.length - 1) * childSpacing;
      const startX = selectedNode.x - totalWidth / 2;

      const newNodes = newIdeasWithId.map((hyp, i) => ({
        id: hyp.id,
        level: selectedNode.level + 1,
        title: hyp.title.trim(),
        content: hyp.content.trim(),
        type: 'complex',
        x: startX + i * childSpacing + Math.random() * 20 - 10,
        y: selectedNode.y + 150 + Math.random() * 20 - 10,
        originalData: hyp.originalData,
        problemHighlights: hyp.problemHighlights || hyp.originalData?.problem_highlights || []
      }));

      const newLinks = newNodes.map((nd) => ({ source: selectedNode.id, target: nd.id }));
      setNodes((prev) => {
        const existingIds = new Set(prev.map(n => n.id));
        const uniqueNewNodes = newNodes.filter(n => !existingIds.has(n.id));
        return [...prev, ...uniqueNewNodes];
      });
      setLinks((prev) => [...prev, ...newLinks]);

      // Track child nodes generation will happen after evaluation

      // 评估 - 传递维度对
      await evaluateIdeas(updatedIdeasList, { allowAutoCenter: true });

      setIsGenerating(false);
      setOperationStatus('');
      setUserInput('');
    } catch (err) {
      console.error('Error generating ideas:', err);
      setError(err.message);
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // ============== 段落6：根据拖拽修改假设 (modifyIdeaBasedOnModifications) ==============
  const modifyIdeaBasedOnModifications = async (originalNode, ghostNode, modifications, behindNode) => {
    const deepClone = (value) => {
      if (value === null || value === undefined) return value;
      try {
        return JSON.parse(JSON.stringify(value));
      } catch (err) {
        console.warn('[WARN] Failed to deep clone value, returning original reference.', err);
        return value;
      }
    };
    console.log('[DEBUG] modifyIdeaBasedOnModifications called with:', {
      originalNodeId: originalNode?.id,
      ghostNodeId: ghostNode?.id,
      modifications,
      behindNodeId: behindNode?.id
    });

    setError(null);
    setIsGenerating(true);
    setOperationStatus('Modifying idea...');
    try {
      if (!selectedDimensionPairs || selectedDimensionPairs.length < 3) {
        throw new Error('Please select 3 dimension pairs before modifying ideas.');
      }

      console.log('[DEBUG] About to call /api/modify...');
      const response = await fetch('/api/modify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          original_idea: originalNode.originalData || {
            id: originalNode.id,
            title: originalNode.title,
            content: originalNode.content
          },
          original_id: originalNode.id,
          modifications,
          behind_idea: behindNode ? (behindNode.originalData || {
            id: behindNode.id,
            title: behindNode.title,
            content: behindNode.content
          }) : null,
          dimension_pairs: selectedDimensionPairs
        })
      });
      console.log('[DEBUG] /api/modify response status:', response.status);
      if (!response.ok) throw new Error('Failed to modify idea');
      const responseText = await response.text();
      console.log('[DEBUG] /api/modify response received, parsing...');
      const data = JSON.parse(responseText);
      if (data.error) throw new Error(data.error);
      console.log('[DEBUG] /api/modify successful, data:', { id: data.id, title: data.title, scores: data.scores });

      // Update ghost node to show as modified
      const oldGhostId = ghostNode.id;
      const backendId = data.id || (data.originalData && data.originalData.id) || oldGhostId;
      const resolvedLevel = Number.isFinite(originalNode.level)
        ? originalNode.level + 1
        : (Number.isFinite(ghostNode.level) ? ghostNode.level : 1);

      // Create a new updated ghost node object (don't mutate existing object)
      const originalSnapshot = originalNode ? {
        id: originalNode.id,
        title: originalNode.title,
        content: originalNode.content,
        type: originalNode.type,
        level: originalNode.level,
        scores: deepClone(originalNode.scores),
        dimension1Score: originalNode.dimension1Score,
        dimension2Score: originalNode.dimension2Score,
        originalData: deepClone(originalNode.originalData)
      } : null;

      const updatedGhostNode = {
        ...ghostNode,
        id: backendId,
        title: data.title,
        content: data.content,
        problemHighlights: data.problemHighlights || data.originalData?.problem_highlights || ghostNode.problemHighlights || originalNode?.problemHighlights || [],
        isModified: true,
        previousState: originalSnapshot,
        isGhost: false, // Show the modified node
        isPendingEvaluation: false, // Has been evaluated by backend
        level: resolvedLevel, // Keep it directly under the original node
        isNewlyGenerated: true, // Always highlight modified result
        // Apply scores from backend (already evaluated by modify_idea)
        scores: data.scores || ghostNode.scores || {},
        dimension1Score: data.dimension1Score ?? ghostNode.dimension1Score,
        dimension2Score: data.dimension2Score ?? ghostNode.dimension2Score,
        originalData: {
          id: backendId,
          ...(data.originalData || {}),
          Title: data.title,
          Name: data.title,
          content: data.content
        }
      };

      const modifiedSnapshot = {
        id: backendId,
        title: updatedGhostNode.title,
        content: updatedGhostNode.content,
        type: updatedGhostNode.type,
        level: updatedGhostNode.level,
        scores: deepClone(updatedGhostNode.scores),
        dimension1Score: updatedGhostNode.dimension1Score,
        dimension2Score: updatedGhostNode.dimension2Score,
        originalData: deepClone(updatedGhostNode.originalData)
      };

      // Preserve user drag target for visualization
      const originalNodeId = originalNode.id;
      console.log('[DEBUG] Checking userDragTargets for originalNodeId:', originalNodeId, 'oldGhostId:', oldGhostId, userDragTargets);

      if (userDragTargets[originalNodeId]) {
        console.log('[DEBUG] Found userDragTarget for', originalNodeId, ':', userDragTargets[originalNodeId]);
        updatedGhostNode.userDragTarget = userDragTargets[originalNodeId];

        // Update userDragTargets key to use new backend ID
        setUserDragTargets(prev => {
          const newTargets = { ...prev };
          newTargets[backendId] = newTargets[originalNodeId];
          delete newTargets[originalNodeId];
          console.log('[DEBUG] Updated userDragTargets keys:', Object.keys(newTargets));
          return newTargets;
        });
        console.log('[DEBUG] Assigned userDragTarget to updatedGhostNode:', updatedGhostNode.userDragTarget);
      } else {
        console.log('[DEBUG] No userDragTarget found for originalNodeId:', originalNodeId);
        console.log('[DEBUG] Available userDragTargets keys:', Object.keys(userDragTargets));
      }

      // Create newIdea with scores already applied (from backend evaluation)
      const newIdea = {
        id: updatedGhostNode.id,
        title: updatedGhostNode.title,
        content: updatedGhostNode.content,
        problemHighlights: updatedGhostNode.problemHighlights || [],
        originalData: updatedGhostNode.originalData,
        // Include scores so it won't be re-evaluated
        scores: updatedGhostNode.scores,
        dimension1Score: updatedGhostNode.dimension1Score,
        dimension2Score: updatedGhostNode.dimension2Score
      };

      // Replace ghost node entry and relink edges
      console.log('[DEBUG] Replacing ghost node:', { oldGhostId, newId: updatedGhostNode.id, isGhost: updatedGhostNode.isGhost, isModified: updatedGhostNode.isModified, scores: updatedGhostNode.scores });
      setNodes(prev => {
        let replaced = false;
        const updated = prev.map(n => {
          if (n.id === oldGhostId) {
            replaced = true;
            return updatedGhostNode;
          }
          if (n.id === originalNode.id) {
            return {
              ...n,
              modifiedState: modifiedSnapshot
            };
          }
          return n;
        });

        if (!replaced) {
          console.warn('[WARN] Ghost node missing during modify replacement, appending modified node directly:', oldGhostId, '→', updatedGhostNode.id);
          return [...updated, updatedGhostNode];
        }

        console.log('[DEBUG] Nodes after replacement:', updated.map(n => ({ id: n.id, isGhost: n.isGhost, isModified: n.isModified, scores: n.scores })));
        return updated;
      });
      setLinks(prev => {
        let rewiredParentLink = false;
        const updatedLinks = prev.map(l => {
          if (l.source === oldGhostId) {
            return { ...l, source: backendId };
          }
          if (l.target === oldGhostId) {
            rewiredParentLink = true;
            return { ...l, target: backendId };
          }
          return l;
        });

        if (!rewiredParentLink && originalNode?.id) {
          console.warn('[WARN] Missing parent link for modified node, creating fallback link:', originalNode.id, '→', backendId);
          updatedLinks.push({ source: originalNode.id, target: backendId });
        }

        return updatedLinks;
      });

      // Add to idea list for evaluation - use current ideasList + newIdea directly
      console.log('[DEBUG] Starting modify evaluation process...');
      console.log('[DEBUG] New idea for evaluation:', { id: newIdea.id, title: newIdea.title, hasOriginalData: !!newIdea.originalData });
      console.log('[DEBUG] Current ideas list count:', ideasList.length);

      const updatedIdeasList = [...ideasList, newIdea];
      console.log('[DEBUG] Updated ideas count:', updatedIdeasList.length);
      console.log('[DEBUG] Updated ideas IDs:', updatedIdeasList.map(i => i.id));

      setIdeasList(updatedIdeasList);

      // Check if there are other ideas that need evaluation
      const ideasNeedingEval = getIdeasNeedingEvaluation(updatedIdeasList);
      console.log('[DEBUG] Ideas needing evaluation:', ideasNeedingEval);

      if (ideasNeedingEval.length > 0) {
        setOperationStatus('Evaluating remaining ideas...');
        console.log('[DEBUG] About to call evaluateIdeas for remaining ideas...');

        try {
          // Use the updated list for evaluation
          await evaluateIdeas(updatedIdeasList, { mode: 'incremental' });
          console.log('[DEBUG] Evaluation completed successfully');
        } catch (evalError) {
          console.error('[DEBUG] Evaluation failed:', evalError);
          throw evalError;
        }
      } else {
        console.log('[DEBUG] All ideas already have scores, skipping evaluation');
      }

      // Node will be revealed automatically by evaluation when scores are applied
      setOperationStatus('');
    } catch (err) {
      console.error('Error modifying idea:', err);
      setError(err.message);
      // Clear drag visual state on error
      setDragVisualState(null);
    } finally {
      setIsGenerating(false);
    }
  };

  // ============== 段落7：节点点击事件处理 ==============
  // Note: Hover events are now handled by unified tracker to prevent duplicates

  const zoomTransformRef = useRef(null);
  // Separate zoom state for evaluation (scatter) view so tree zoom persists independently
  const evaluationZoomTransformRef = useRef(d3.zoomIdentity);
  const layoutNodesRef = useRef([]);

  // Merge functionality handlers
  const handleMergeConfirm = useCallback(async (overrideFirst = null, overrideSecond = null) => {
    const firstNode = overrideFirst || mergeMode.firstNode;
    const secondNode = overrideSecond || mergeMode.secondNode;

    if (firstNode && secondNode) {
      // Store references before clearing merge mode

      // Check if either node is a fragment
      const isFirstFragment = firstNode.type === 'fragment';
      const isSecondFragment = secondNode.type === 'fragment';

      if (!mergeAnimationState) {
        const ghostPos =
          getNodeCenterRelativeToContainer(firstNode.id) ||
          getNodeCenterRelativeToContainer(secondNode.id);
        setMergeAnimationState({
          sourceId: firstNode.id,
          targetId: secondNode.id,
          ghostPosition: ghostPos
        });
      }

      // Immediately hide the merge dialog and dashed circles
      setMergeMode({
        active: false,
        firstNode: null,
        secondNode: null,
        cursorPosition: { x: 0, y: 0 },
        showDialog: false
      });

      setIsGenerating(true);
      setOperationStatus('Merging ideas...');
      setError(null);

      try {
        // Call backend merge API
        const response = await fetch('/api/merge', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            idea_a: firstNode.originalData || {
              id: firstNode.id,
              title: firstNode.title,
              content: firstNode.content
            },
            idea_b: secondNode.originalData || {
              id: secondNode.id,
              title: secondNode.title,
              content: secondNode.content
            },
            idea_a_id: firstNode.id,
            idea_b_id: secondNode.id
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to merge ideas');
        }

        // Wait for the full response as text (handles heartbeat stream)
        const responseText = await response.text();

        // Parse the text into JSON, ignoring heartbeats
        const data = JSON.parse(responseText);

        if (data.error) {
          throw new Error(data.error);
        }

        // Create merged node with AI-generated content
        const mergedNodeId = data.id || createMergeId(firstNode.id, secondNode.id);
        const mergedOriginalData = normalizeIdeaOriginalData(
          data.originalData,
          {
            id: mergedNodeId,
            title: data.title,
            content: data.content,
            problemHighlights: data.problemHighlights
          }
        );
        const mergedNode = {
          id: mergedNodeId,
          title: data.title.trim(),
          content: data.content.trim(),
          level: Math.max(firstNode.level || 0, secondNode.level || 0) + 1,
          type: 'complex',
          isMergedResult: true,
          parentIds: [firstNode.id, secondNode.id],
          mergeTimestamp: Date.now(),
          problemHighlights: data.problemHighlights || data.originalData?.problem_highlights || [],
          originalData: mergedOriginalData,
          x: (firstNode.x || 0 + secondNode.x || 0) / 2,
          y: Math.max(firstNode.y || 0, secondNode.y || 0) + 150
        };

        // Add the merged node to the nodes list and reset parent node visuals
        setNodes(prevNodes => {
          const cleaned = prevNodes.map(n => {
            if (n.id === firstNode.id || n.id === secondNode.id) {
              const updated = { ...n };
              delete updated.evaluationOpacity;
              delete updated.isBeingMerged;
              return updated;
            }
            return n;
          });
          return [...cleaned, mergedNode];
        });

        // Delete fragment nodes if they were merged
        if (isFirstFragment) {
          setFragmentNodes(prev => prev.filter(f => f.id !== firstNode.id));
        }
        if (isSecondFragment) {
          setFragmentNodes(prev => prev.filter(f => f.id !== secondNode.id));
        }

        // Add to ideas list for evaluation (same structure as original ideas)
        const newIdea = {
          id: mergedNodeId,
          title: data.title,
          content: data.content,
          problemHighlights: data.problemHighlights || data.originalData?.problem_highlights || [],
          originalData: mergedOriginalData
        };
        setIdeasList(prevIdeas => [...prevIdeas, newIdea]);


        // Create links from both parent nodes to the merged node
        const newLinks = [
          {
            source: firstNode.id,
            target: mergedNode.id,
            type: 'merge'
          },
          {
            source: secondNode.id,
            target: mergedNode.id,
            type: 'merge'
          }
        ];
        setLinks(prevLinks => [...prevLinks, ...newLinks]);

        // Track the merge action

        // Select the new merged node
        setSelectedNode(mergedNode);

        // Evaluate the new merged idea with full context
        const updatedIdeasList = [...ideasList, newIdea];
        setIdeasList(updatedIdeasList);
        await evaluateIdeasRef.current(updatedIdeasList, { mode: 'incremental' });

      } catch (err) {
        console.error('Error merging ideas:', err);
        setError(err.message);
      } finally {
        setIsGenerating(false);
        setOperationStatus('');
        setMergeAnimationState(null);

        // Reset merge animation states for nodes
        setNodes((prev) =>
          prev.map((n) => {
            if (n.evaluationOpacity === 0 || n.isBeingMerged) {
              const updated = { ...n };
              delete updated.evaluationOpacity;
              delete updated.isBeingMerged;
              return updated;
            }
            return n;
          })
        );

        // Reset merge animation states for fragments (in case of error)
        setFragmentNodes((prev) =>
          prev.map((f) => {
            if (f.evaluationOpacity === 0 || f.isPendingMerge) {
              const updated = { ...f };
              delete updated.evaluationOpacity;
              delete updated.isPendingMerge;
              return updated;
            }
            return f;
          })
        );

        setMergeTargetId(null);
      }

      // Merge mode already reset at the beginning
    }
  }, [createMergeId, getNodeCenterRelativeToContainer, ideasList, mergeAnimationState, mergeMode, normalizeIdeaOriginalData]);

  const handleMergeCancel = useCallback(() => {
    // Track the cancel action
    if (mergeMode.firstNode && mergeMode.secondNode) {
    }

    // Reset merge mode
    setMergeMode({
      active: false,
      firstNode: null,
      secondNode: null,
      cursorPosition: { x: 0, y: 0 },
      showDialog: false
    });
  }, [mergeMode]);

  // Cleanup is handled by D3's selectAll('*').remove() in the main useEffect

  // ============== 段落8：D3 渲染逻辑 useEffect ==============
  useEffect(() => {
    if (!svgRef.current || currentView === 'home_view') return;

    const width = 800;
    const height = 600;

    // 清空 SVG and remove all event listeners to prevent duplicates
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.on('.tree-tracking', null); // Remove all tree-tracking events
    svg.on('.scatter-tracking', null); // Remove all scatter-tracking events
    svg.on('.tree-merge', null); // Remove tree merge events

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
          // Also transform merge UI layer if it exists
          const mergeUILayer = svg.select('.merge-ui-layer');
          if (!mergeUILayer.empty()) {
            mergeUILayer.attr('transform', event.transform);
          }
          zoomTransformRef.current = event.transform;
        });

      // Apply zoom behavior to SVG
      svg.call(zoom);

      // Set initial transform
      const transformToUse = zoomTransformRef.current || initialTransform;
      svg.call(zoom.transform, transformToUse);

      // Add mouse move listener for merge mode cursor tracking
      svg.on('mousemove.merge', (event) => {
        if (mergeMode.active && mergeMode.firstNode && !mergeMode.secondNode) {
          const [x, y] = d3.pointer(event, zoomGroup.node());
          setMergeMode(prev => ({
            ...prev,
            cursorPosition: { x, y }
          }));
        }
      });

      // Add click-outside handler to cancel merge mode
      svg.on('click.merge-cancel', (event) => {
        // Check if click was on background (not on a node)
        if (mergeMode.active && event.target === svg.node()) {
          handleMergeCancel();
        }
      });

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
            // Check if this is a custom idea (no parent links)
            const hasParentLink = links.some(link => link.target === node.id);
            if (!hasParentLink && node.level === 1) {
              // Custom ideas without parents should be sorted after regular nodes
              return 1000; // Large value to put them at the end
            }

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

            const aId = a.id || '';
            const bId = b.id || '';
            return aId.localeCompare(bId);
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

      // Store layoutNodes in ref for merge UI useEffect
      layoutNodesRef.current = layoutNodes;

      /* 连线 - filter out links to temporary ghost nodes in tree view */
      const visibleLinks = links.filter(lk => {
        const targetNode = layoutNodes.find(n => n.id === lk.target);
        return !targetNode || !(targetNode.isGhost && !targetNode.isModified);
      });

      visibleLinks.forEach((lk) => {
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

      /* 节点 - filter out only temporary ghost nodes in tree view */
      const visibleNodes = layoutNodes.filter(node => !(node.isGhost && !node.isModified));
      const nodeG = g
        .selectAll('.node')
        .data(visibleNodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', (d) => `translate(${d.x},${d.y})`);

      // Add event listeners with unified tracking (no duplicates)
      nodeG
        .on('mouseenter.tree-tracking', function (event, d) {
          // Don't track hover if any node is being dragged, events are suppressed after drag, just clicked, or in merge mode
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag && !d._justClicked && !mergeMode.active) {
            // Prevent duplicate tracking by checking if this exact same hover was just tracked
            if (!d._lastHoverTime || Date.now() - d._lastHoverTime > 100) {
              d._lastHoverTime = Date.now();
            }
          }
          // UI functionality
          setHoveredNode(d);
          // Add hover styling directly to circle only
          // Note: z-ordering is handled by selectedNode useEffect
          d3.select(this).select('circle').style('stroke', '#000').style('stroke-width', 4);
        })
        .on('mouseleave.tree-tracking', function (event, d) {
          // Don't track hover if any node is being dragged, events are suppressed after drag, or in merge mode
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag && !mergeMode.active) {
          }
          // UI functionality
          setHoveredNode(null);
          // Restore original styling to circle only (check if selected)
          const isSelected = selectedNode?.id === d.id;
          d3.select(this).select('circle').style('stroke', isSelected ? '#000' : '#fff').style('stroke-width', isSelected ? 4 : 2);
        })
        .on('click.tree-tracking', (_, d) => {
          // Don't track click if events are suppressed after drag
          if (!d._suppressEventsAfterDrag) {
            // Set flag to prevent hover tracking immediately after click
            d._justClicked = true;
            setTimeout(() => { d._justClicked = false; }, 200);

            // Clear any existing single click timeout
            if (d._singleClickTimeout) {
              clearTimeout(d._singleClickTimeout);
              d._singleClickTimeout = null;
            }

            // Delay single click tracking to distinguish from double click
            d._singleClickTimeout = setTimeout(() => {
              d._singleClickTimeout = null;
            }, 300); // 300ms delay to detect double clicks
          }
          // UI functionality
          setSelectedNode(d);
        })
        .on('dblclick.tree-merge', (event, d) => {
          event.stopPropagation();

          // Cancel any pending single click tracking since this is a double click
          if (d._singleClickTimeout) {
            clearTimeout(d._singleClickTimeout);
            d._singleClickTimeout = null;
          }

          if (!mergeMode.active) {
            // First node selected - enter merge mode
            setMergeMode({
              active: true,
              firstNode: d,
              secondNode: null,
              cursorPosition: { x: d.x || 0, y: d.y || 0 },
              showDialog: false
            });

          } else if (mergeMode.firstNode && mergeMode.firstNode.id !== d.id) {
            // Second node selected - complete merge selection
            setMergeMode(prev => ({
              ...prev,
              secondNode: d,
              showDialog: true
            }));

          } else if (mergeMode.firstNode && mergeMode.firstNode.id === d.id) {
            // Same node double-clicked - cancel merge mode
            setMergeMode({
              active: false,
              firstNode: null,
              secondNode: null,
              cursorPosition: { x: 0, y: 0 },
              showDialog: false
            });

          }
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
          selectedNode?.id === d.id ? '#000' : '#fff'
        )
        .style('stroke-width', (d) =>
          selectedNode?.id === d.id ? 4 : 2
        )
        .style('cursor', 'pointer');

      // Add text label with a styled rectangular background
      const label = nodeG.append('g')
        .attr('class', 'label-group')
        .style('pointer-events', 'none'); // The label should not capture mouse events

      // Add text first to measure it
      label.append('text')
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

      // Create or get persistent merge UI layer
      let mergeUILayer = svg.select('.merge-ui-layer');
      if (mergeUILayer.empty()) {
        mergeUILayer = svg.append('g').attr('class', 'merge-ui-layer');
      }
      // Note: We no longer clear merge UI elements here to prevent flickering

      // Apply current zoom transform to merge UI layer
      if (zoomTransformRef.current) {
        mergeUILayer.attr('transform', zoomTransformRef.current);
      }

      // Merge UI is now handled by a separate useEffect to prevent flickering

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
          svg.transition().call(zoom.scaleBy, 1.1);
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
          svg.transition().call(zoom.scaleBy, 0.9);
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

      if (operationStatus && (operationStatus.toLowerCase().includes('generating') || (hideEvaluationView && (operationStatus.toLowerCase().includes('merging') || operationStatus.toLowerCase().includes('evaluating') || operationStatus.toLowerCase().includes('modifying'))))) {
        // Add status text at top of plot instead of overlay
        svg
          .append('text')
          .attr('x', width / 2)
          .attr('y', 30)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-size', '1.2rem')
          .style('font-weight', '600')
          .style('fill', '#374151')
          .style('background', 'rgba(255, 255, 255, 0.9)')
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

      /* 获取当前维度对 */
      const { xDimension, yDimension } = getCurrentDimensions();

      /* 为每个节点计算并缓存坐标值（兼容新旧系统）*/
      nodes.forEach(node => {
        node._evalX = getNodeX(node);
        node._evalY = getNodeY(node);
      });

      // 定义获取节点坐标的辅助函数（用于后续代码）
      const getX = (n) => n._evalX;
      const getY = (n) => n._evalY;

      /* 显示当前选中的维度对（替代原来的下拉选择器）*/
      if (xDimension && yDimension) {
        const dimInfoFO = svg
          .append('foreignObject')
          .attr('x', width - 250)
          .attr('y', 20)
          .attr('width', 240)
          .attr('height', 80);

        dimInfoFO
          .append('xhtml:div')
          .style('display', 'flex')
          .style('flexDirection', 'column')
          .style('gap', '8px')
          .style('padding', '12px')
          .style('backgroundColor', '#F9FAFB')
          .style('border', '1px solid #E5E7EB')
          .style('borderRadius', '6px')
          .html(`
            <div style="font-size:0.75rem;color:#6B7280;font-weight:600;margin-bottom:4px;">Current Dimensions</div>
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="font-size:0.7rem;color:#9CA3AF;">X:</span>
              <span style="font-size:0.75rem;color:#374151;font-weight:500;">${xDimension.dimensionA}</span>
              <span style="font-size:0.7rem;color:#9CA3AF;">←→</span>
              <span style="font-size:0.75rem;color:#374151;font-weight:500;">${xDimension.dimensionB}</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="font-size:0.7rem;color:#9CA3AF;">Y:</span>
              <span style="font-size:0.75rem;color:#374151;font-weight:500;">${yDimension.dimensionA}</span>
              <span style="font-size:0.7rem;color:#9CA3AF;">←→</span>
              <span style="font-size:0.75rem;color:#374151;font-weight:500;">${yDimension.dimensionB}</span>
            </div>
          `);
      }

      /* 画布与比例尺 */
      const margin = { top: 100, right: 50, bottom: 80, left: 50 };
      const chartW = width - margin.left - margin.right;
      const chartH = height - margin.top - margin.bottom;
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      // Linear zoom-window scaling around centers with uniform grid spacing
      const scaleX = unifiedScale;
      const scaleY = unifiedScale;

      // Compute visible domains based on center + scale (clamped to [0,100])
      let xMin = Math.max(0, xScalingCenter - 50 / scaleX);
      let xMax = Math.min(100, xScalingCenter + 50 / scaleX);
      if (xMax - xMin < 1e-6) { // Avoid degenerate domain
        const mid = (xMin + xMax) / 2;
        xMin = Math.max(0, mid - 0.5);
        xMax = Math.min(100, mid + 0.5);
      }

      let yMin = Math.max(0, yScalingCenter - 50 / scaleY);
      let yMax = Math.min(100, yScalingCenter + 50 / scaleY);
      if (yMax - yMin < 1e-6) {
        const mid = (yMin + yMax) / 2;
        yMin = Math.max(0, mid - 0.5);
        yMax = Math.min(100, mid + 0.5);
      }

      // View scales map the current visible domain to screen space
      const xView = d3.scaleLinear().domain([xMin, xMax]).range([0, chartW]);
      const yView = d3.scaleLinear().domain([yMin, yMax]).range([chartH, 0]);

      // Coordinate functions for nodes and ticks (uniform spacing)
      const coordX = (raw) => xView(raw);
      const coordY = (raw) => yView(raw);
      const tickCoordX = (t) => xView(t);
      const tickCoordY = (t) => yView(t);

      // Static content layer (removed pan/move feature)
      const contentLayer = g.append('g').attr('class', 'evaluation-zoom-layer');

      /* 动态网格 & 轴 (reflect scaling + distribution) */
      const allMajorTicks = d3.range(0, 101, 20);
      const allMinorTicks = d3.range(0, 101, 10).filter(t => !allMajorTicks.includes(t));
      // Only show ticks within the visible domain (hide out-of-bounds)
      const tickMajorX = allMajorTicks.filter(t => t >= xMin && t <= xMax);
      const tickMinorX = allMinorTicks.filter(t => t >= xMin && t <= xMax);
      const tickMajorY = allMajorTicks.filter(t => t >= yMin && t <= yMax);
      const tickMinorY = allMinorTicks.filter(t => t >= yMin && t <= yMax);
      // X grid lines
      const xGridGroup = g.append('g').attr('class', 'x-grid');
      xGridGroup.selectAll('line.major')
        .data(tickMajorX)
        .enter().append('line')
        .attr('class', 'major')
        .attr('x1', d => tickCoordX(d))
        .attr('x2', d => tickCoordX(d))
        .attr('y1', 0)
        .attr('y2', chartH)
        .style('stroke', '#E5E7EB').style('stroke-width', 1);
      xGridGroup.selectAll('line.minor')
        .data(tickMinorX)
        .enter().append('line')
        .attr('class', 'minor')
        .attr('x1', d => tickCoordX(d))
        .attr('x2', d => tickCoordX(d))
        .attr('y1', 0)
        .attr('y2', chartH)
        .style('stroke', '#F3F4F6').style('stroke-width', 1);
      // Y grid lines
      const yGridGroup = g.append('g').attr('class', 'y-grid');
      yGridGroup.selectAll('line.major')
        .data(tickMajorY)
        .enter().append('line')
        .attr('class', 'major')
        .attr('x1', 0)
        .attr('x2', chartW)
        .attr('y1', d => tickCoordY(d))
        .attr('y2', d => tickCoordY(d))
        .style('stroke', '#E5E7EB').style('stroke-width', 1);
      yGridGroup.selectAll('line.minor')
        .data(tickMinorY)
        .enter().append('line')
        .attr('class', 'minor')
        .attr('x1', 0)
        .attr('x2', chartW)
        .attr('y1', d => tickCoordY(d))
        .attr('y2', d => tickCoordY(d))
        .style('stroke', '#F3F4F6').style('stroke-width', 1);
      // Custom X axis
      const xAxisGroup = g.append('g').attr('class', 'dynamic-x-axis').attr('transform', `translate(0,${chartH})`);
      xAxisGroup.selectAll('line.tick')
        .data(tickMajorX)
        .enter().append('line')
        .attr('class', 'tick')
        .attr('x1', d => tickCoordX(d))
        .attr('x2', d => tickCoordX(d))
        .attr('y1', 0)
        .attr('y2', 6)
        .style('stroke', '#374151');
      xAxisGroup.selectAll('text.tick-label')
        .data(tickMajorX)
        .enter().append('text')
        .attr('class', 'tick-label')
        .attr('x', d => tickCoordX(d))
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', '#374151')
        .text(d => d);
      // Edge indicators for X (show true boundary values when 0/100 are out of view)
      const EDGE_EPS = 1e-6;
      const fmtEdge = (v) => Math.round(v);
      // Left border (xMin)
      if (xMin > 0 + EDGE_EPS) {
        // Border-aligned grid line (left)
        xGridGroup.append('line')
          .attr('class', 'edge-grid')
          .attr('x1', 0)
          .attr('x2', 0)
          .attr('y1', 0)
          .attr('y2', chartH)
          .style('stroke', '#E5E7EB').style('stroke-width', 1);
        // Axis tick and label at border
        xAxisGroup.append('line')
          .attr('class', 'edge-tick')
          .attr('x1', 0)
          .attr('x2', 0)
          .attr('y1', 0)
          .attr('y2', 6)
          .style('stroke', '#374151');
        xAxisGroup.append('text')
          .attr('class', 'edge-tick-label')
          .attr('x', 0)
          .attr('y', 20)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(fmtEdge(xMin));
      }
      // Right border (xMax)
      if (xMax < 100 - EDGE_EPS) {
        // Border-aligned grid line (right)
        xGridGroup.append('line')
          .attr('class', 'edge-grid')
          .attr('x1', chartW)
          .attr('x2', chartW)
          .attr('y1', 0)
          .attr('y2', chartH)
          .style('stroke', '#E5E7EB').style('stroke-width', 1);
        // Axis tick and label at border
        xAxisGroup.append('line')
          .attr('class', 'edge-tick')
          .attr('x1', chartW)
          .attr('x2', chartW)
          .attr('y1', 0)
          .attr('y2', 6)
          .style('stroke', '#374151');
        xAxisGroup.append('text')
          .attr('class', 'edge-tick-label')
          .attr('x', chartW)
          .attr('y', 20)
          .attr('text-anchor', 'middle')
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(fmtEdge(xMax));
      }
      // Custom Y axis
      const yAxisGroup = g.append('g').attr('class', 'dynamic-y-axis');
      yAxisGroup.selectAll('line.tick')
        .data(tickMajorY)
        .enter().append('line')
        .attr('class', 'tick')
        .attr('x1', -6)
        .attr('x2', 0)
        .attr('y1', d => tickCoordY(d))
        .attr('y2', d => tickCoordY(d))
        .style('stroke', '#374151');
      yAxisGroup.selectAll('text.tick-label')
        .data(tickMajorY)
        .enter().append('text')
        .attr('class', 'tick-label')
        .attr('x', -10)
        .attr('y', d => tickCoordY(d) + 3)
        .attr('text-anchor', 'end')
        .style('font-size', '10px')
        .style('fill', '#374151')
        .text(d => d);
      // Edge indicators for Y (show true boundary values when 0/100 are out of view)
      // Bottom border (yMin)
      if (yMin > 0 + EDGE_EPS) {
        // Border-aligned grid line (bottom)
        yGridGroup.append('line')
          .attr('class', 'edge-grid')
          .attr('x1', 0)
          .attr('x2', chartW)
          .attr('y1', chartH)
          .attr('y2', chartH)
          .style('stroke', '#E5E7EB').style('stroke-width', 1);
        // Axis tick and label at border
        yAxisGroup.append('line')
          .attr('class', 'edge-tick')
          .attr('x1', -6)
          .attr('x2', 0)
          .attr('y1', chartH)
          .attr('y2', chartH)
          .style('stroke', '#374151');
        yAxisGroup.append('text')
          .attr('class', 'edge-tick-label')
          .attr('x', -10)
          .attr('y', chartH + 3)
          .attr('text-anchor', 'end')
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(fmtEdge(yMin));
      }
      // Top border (yMax)
      if (yMax < 100 - EDGE_EPS) {
        // Border-aligned grid line (top)
        yGridGroup.append('line')
          .attr('class', 'edge-grid')
          .attr('x1', 0)
          .attr('x2', chartW)
          .attr('y1', 0)
          .attr('y2', 0)
          .style('stroke', '#E5E7EB').style('stroke-width', 1);
        // Axis tick and label at border
        yAxisGroup.append('line')
          .attr('class', 'edge-tick')
          .attr('x1', -6)
          .attr('x2', 0)
          .attr('y1', 0)
          .attr('y2', 0)
          .style('stroke', '#374151');
        yAxisGroup.append('text')
          .attr('class', 'edge-tick-label')
          .attr('x', -10)
          .attr('y', 3)
          .attr('text-anchor', 'end')
          .style('font-size', '10px')
          .style('fill', '#374151')
          .text(fmtEdge(yMax));
      }

      // Ensure node/content layer is above grids & axes (z-order)
      contentLayer.raise();

      // Removed pan reset button (no movement allowed now)

      /* 轴标签 - 显示维度对名称 */
      if (xDimension && yDimension) {
        // X 轴标签：左侧显示 dimensionA，右侧显示 dimensionB
        g.append('text')
          .attr('x', 0)
          .attr('y', chartH + 60)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(xDimension.dimensionA);

        g.append('text')
          .attr('x', chartW / 2)
          .attr('y', chartH + 60)
          .style('text-anchor', 'middle')
          .style('fill', '#9CA3AF')
          .style('font-size', '0.75rem')
          .text('←→');

        g.append('text')
          .attr('x', chartW)
          .attr('y', chartH + 60)
          .style('text-anchor', 'end')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(xDimension.dimensionB);

        // Y 轴标签：底部显示 dimensionA，顶部显示 dimensionB
        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH)
          .attr('y', -40)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(yDimension.dimensionA);

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH / 2)
          .attr('y', -40)
          .style('text-anchor', 'middle')
          .style('fill', '#9CA3AF')
          .style('font-size', '0.75rem')
          .text('←→');

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', 0)
          .attr('y', -40)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(yDimension.dimensionB);
      } else {
        // Fallback: 显示旧的轴名称
        g.append('text')
          .attr('x', chartW / 2)
          .attr('y', chartH + 40)
          .style('text-anchor', 'middle')
          .style('fill', '#374151')
          .style('font-size', '1.2rem')
      .text(xAxisLabel || xAxisMetric);

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH / 2)
          .attr('y', -30)
          .style('text-anchor', 'middle')
          .style('fill', '#374151')
          .style('font-size', '1.2rem')
      .text(yAxisLabel || yAxisMetric);
      }

      // (Zoom controls removed previously)
      // Remove any previous scaling controls before re-render (avoid duplicates)
      svg.selectAll('.scaling-controls').remove();

      // Unified scaling controls (similar to tree view)
      const controlLayer = svg.append('g').attr('class', 'scaling-controls');

      // Check if there are visible nodes to enable/disable buttons
      const hasVisibleNodes = nodes.some(n => getX(n) !== undefined && getY(n) !== undefined && !n.isGhost);
      // Unified scaling controls positioned in top-right like tree view
      const controls = controlLayer.append('g')
        .attr('class', 'unified-scale-controls')
        .attr('transform', `translate(${width - 40}, 20)`);

      // Lock scaling if any node touches the visible border
      const pixelTol = 10; // include node radius and label padding
      const lockScaling = hasVisibleNodes && nodes.some(n => {
        if (getX(n) === undefined || getY(n) === undefined || n.isGhost) return false;
        const px = coordX(getX(n));
        const py = coordY(getY(n));
        return px <= 0 + pixelTol || px >= chartW - pixelTol || py <= 0 + pixelTol || py >= chartH - pixelTol;
      });

      // Separate disabled states for buttons (restore remains enabled when nodes exist)
      const upDisabled = !hasVisibleNodes || lockScaling;
      const downDisabled = !hasVisibleNodes || lockScaling;
      const restoreDisabled = !hasVisibleNodes;
      const upOpacity = upDisabled ? 0.3 : 1;
      const downOpacity = downDisabled ? 0.3 : 1;
      const restoreOpacity = restoreDisabled ? 0.3 : 1;
      const upCursor = upDisabled ? 'not-allowed' : 'pointer';
      const downCursor = downDisabled ? 'not-allowed' : 'pointer';
      const restoreCursor = restoreDisabled ? 'not-allowed' : 'pointer';

      // Scale up button (+)

      controls.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('opacity', upOpacity)
        .style('cursor', upCursor)
        .on('click', () => {
          if (!upDisabled) {
            // Mark that user has interacted with scale
            setUserHasInteractedWithScale(true);
            // Smart zoom: re-center to nodes while scaling up
            const { xCenter, yCenter } = calculateNodeMidpoints();
            setXScalingCenter(xCenter);
            setYScalingCenter(yCenter);
            setUnifiedScale(prev => Math.min(4.0, prev + 0.1));
          }
        });
      controls.append('text')
        .attr('x', 14)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .style('opacity', upOpacity)
        .text('+');

      // Scale down button (-)
      controls.append('rect')
        .attr('x', 0)
        .attr('y', 32)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('opacity', downOpacity)
        .style('cursor', downCursor)
        .on('click', () => {
          if (!downDisabled) {
            // Mark that user has interacted with scale
            setUserHasInteractedWithScale(true);
            // Smart zoom: re-center to nodes while scaling down
            const { xCenter, yCenter } = calculateNodeMidpoints();
            setXScalingCenter(xCenter);
            setYScalingCenter(yCenter);
            setUnifiedScale(prev => Math.max(0.25, prev - 0.1));
          }
        });
      controls.append('text')
        .attr('x', 14)
        .attr('y', 48)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .style('opacity', downOpacity)
        .text('−');

      // Restore button (reset to 1.0 with dynamic centering)
      controls.append('rect')
        .attr('x', 0)
        .attr('y', 64)
        .attr('width', 28)
        .attr('height', 28)
        .attr('rx', 4)
        .style('fill', '#f3f4f6')
        .style('stroke', '#d1d5db')
        .style('opacity', restoreOpacity)
        .style('cursor', restoreCursor)
        .on('click', () => {
          if (!restoreDisabled) {
            // True restore: reset to initial state (full 0-100 view)
            setUnifiedScale(1.0);
            setXScalingCenter(50);
            setYScalingCenter(50);
            // Reset user interaction flag so auto-centering can work again
            setUserHasInteractedWithScale(false);
          }
        });
      controls.append('text')
        .attr('x', 13)
        .attr('y', 78)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '18px')
        .style('pointer-events', 'none')
        .style('opacity', restoreOpacity)
        .text('⟲');

      // Reapply stored transform if previously zoomed
      // Zoom persistence removed

      /* 轴标签 - 显示维度对名称 */
      if (xDimension && yDimension) {
        // X 轴标签：左侧显示 dimensionA，右侧显示 dimensionB
        g.append('text')
          .attr('x', 0)
          .attr('y', chartH + 60)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(xDimension.dimensionA);

        g.append('text')
          .attr('x', chartW / 2)
          .attr('y', chartH + 60)
          .style('text-anchor', 'middle')
          .style('fill', '#9CA3AF')
          .style('font-size', '0.75rem')
          .text('←→');

        g.append('text')
          .attr('x', chartW)
          .attr('y', chartH + 60)
          .style('text-anchor', 'end')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(xDimension.dimensionB);

        // Y 轴标签：底部显示 dimensionA，顶部显示 dimensionB
        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH)
          .attr('y', -40)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(yDimension.dimensionA);

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH / 2)
          .attr('y', -40)
          .style('text-anchor', 'middle')
          .style('fill', '#9CA3AF')
          .style('font-size', '0.75rem')
          .text('←→');

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', 0)
          .attr('y', -40)
          .style('text-anchor', 'start')
          .style('fill', '#374151')
          .style('font-size', '0.9rem')
          .style('font-weight', '600')
          .text(yDimension.dimensionB);
      } else {
        // Fallback: 显示旧的轴名称
        g.append('text')
          .attr('x', chartW / 2)
          .attr('y', chartH + 40)
          .style('text-anchor', 'middle')
          .style('fill', '#374151')
          .style('font-size', '1.2rem')
      .text(xAxisLabel || xAxisMetric);

        g.append('text')
          .attr('transform', 'rotate(-90)')
          .attr('x', -chartH / 2)
          .attr('y', -30)
          .style('text-anchor', 'middle')
          .style('fill', '#374151')
          .style('font-size', '1.2rem')
      .text(yAxisLabel || yAxisMetric);
      }

      // Ensure node/content layer is above grids & axes (z-order)
      contentLayer.raise();

      /* 过滤能绘制的节点 - keep all nodes visible during evaluation */
      const drawable = nodes.filter((n) => {
        // Must have coordinates to be drawable
        const hasCoords = getX(n) !== undefined && getY(n) !== undefined;
        // Always show all nodes with coordinates, even during evaluation
        return hasCoords;
      });

      /* ---------- 连线：evaluation + merge ---------- */
      const hasCoords = (n) => getX(n) !== undefined && getY(n) !== undefined;
      const isNodeActive = (nodeId) => {
        if (!nodeId) return false;
        if (selectedNode && selectedNode.id === nodeId) return true;
        if (hoveredNode && hoveredNode.id === nodeId) return true;
        return false;
      };

      /* 两类连线
      ① original → modified   （浅灰 #ccc, 1px）  - Always visible
      ② mergedResult ↔ originals（深灰 #999, 1.5px） - Always visible
      ③ parent → children (generated ideas) - Only visible on hover */
      links.forEach((lk) => {
        const s = nodes.find((n) => n.id === lk.source);
        const t = nodes.find((n) => n.id === lk.target);
        if (!s || !t) return;
        if (!hasCoords(s) || !hasCoords(t)) return; // 任一端缺坐标则跳过

        const isMergeEdge = s.isMergedResult || t.isMergedResult;
        const isModificationEdge = t.previousState && t.previousState.id === s.id;

        // Check if this is a parent-child generation link (hide by default, show on hover)
        const isParentChildGeneration = !isMergeEdge && !isModificationEdge;

        // Modification edges: only show when either endpoint is hovered or selected
        if (isModificationEdge && !isNodeActive(s.id) && !isNodeActive(t.id)) {
          return;
        }

        // Show parent-child generation edges only when one endpoint selected
        if (isParentChildGeneration &&
          (!selectedNode || (selectedNode.id !== lk.source && selectedNode.id !== lk.target))) {
          return;
        }

        contentLayer.append('line')
          .attr('x1', coordX(getX(s)))
          .attr('y1', coordY(getY(s)))
          .attr('x2', coordX(getX(t)))
          .attr('y2', coordY(getY(t)))
          .style('stroke', (() => {
            if (isModificationEdge) return '#fbbf24';
            return isMergeEdge ? '#999' : '#ccc';
          })())
          .style('stroke-width', isModificationEdge ? 2 : (isMergeEdge ? 1.5 : 1))
          .style('stroke-dasharray', isModificationEdge ? '6,4' : 'none')
          .style('opacity', isParentChildGeneration ? 0.8 : 1);
      });

      /* 节点绘制 & 拖拽 */
      const nodeG = contentLayer
        .selectAll('.node-group')
        .data(drawable, (d) => d.id)
        .enter()
        .append('g')
        .attr('class', 'node-group')
        .attr('transform', (d) => {
          if (d._tmpViewX !== undefined && d._tmpViewY !== undefined) {
            return `translate(${d._tmpViewX},${d._tmpViewY})`;
          }
          return `translate(${coordX(getX(d))},${coordY(getY(d))})`;
        })
        .style('cursor', 'pointer');

      // Add event listeners with unified tracking (no duplicates)
      nodeG
        .on('mouseenter.scatter-tracking', function (e, d) {
          // Don't track hover if any node is being dragged, events are suppressed after drag, or just clicked
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag && !d._justClicked) {
            // Prevent duplicate tracking by checking if this exact same hover was just tracked
            if (!d._lastHoverTime || Date.now() - d._lastHoverTime > 100) {
              d._lastHoverTime = Date.now();
            }
          }
          // UI functionality
          setHoveredNode(d);
          // Add hover styling directly to circle only
          // Note: z-ordering is handled by selectedNode useEffect
          d3.select(this).select('circle').style('stroke', '#000').style('stroke-width', 4);
        })
        .on('mouseleave.scatter-tracking', function (e, d) {
          // Don't track hover if any node is being dragged or events are suppressed after drag
          if (!d._dragStarted && !nodes.some(n => n._dragStarted) && !d._suppressEventsAfterDrag) {
          }
          // UI functionality
          setHoveredNode(null);
          // Restore original styling to circle only (check if selected)
          const isSelected = selectedNode?.id === d.id;
          d3.select(this).select('circle').style('stroke', isSelected ? '#000' : '#fff').style('stroke-width', isSelected ? 4 : 2);
          // Selected node z-ordering is handled by useEffect
        })
        .on('click.scatter-tracking', function (e, d) {
          // Don't track click if events are suppressed after drag
          if (!d._suppressEventsAfterDrag) {
            // Set flag to prevent hover tracking immediately after click
            d._justClicked = true;
            setTimeout(() => { d._justClicked = false; }, 200);

            // Clear any existing single click timeout
            if (d._singleClickTimeout) {
              clearTimeout(d._singleClickTimeout);
              d._singleClickTimeout = null;
            }

            // Delay single click tracking to distinguish from double click
            d._singleClickTimeout = setTimeout(() => {
              d._singleClickTimeout = null;
            }, 300); // 300ms delay to detect double clicks
          }
          // UI functionality
          setSelectedNode(d);
          // Note: z-ordering is handled by selectedNode useEffect
        })
        .call(
          d3
            .drag()
            .on('start', function (event, d) {
              if (isGenerating || isEvaluating) return;
              // Disable dragging when zoomed (non-identity) to avoid coordinate confusion
              if (evaluationZoomTransformRef.current && evaluationZoomTransformRef.current.k !== 1) {
                return;
              }
              // Mark drag start but don't track yet - wait for actual movement
              d._dragStarted = false;
              d._dragStartValues = {
                x: getNodeX(d),
                y: getNodeY(d),
                xAxisMetric: xAxisMetric,
                yAxisMetric: yAxisMetric
              };
              d3.select(this).raise();
            })
            .on('drag', function (event, d) {
              if (isGenerating || isEvaluating) return;
              if (evaluationZoomTransformRef.current && evaluationZoomTransformRef.current.k !== 1) {
                return;
              }

              if (!d._dragStarted) {
                d._dragStarted = true;
              }
              const [cx, cy] = d3.pointer(event, g.node());
              d3.select(this).attr('transform', `translate(${cx},${cy})`);
              d._tmpViewX = cx;
              d._tmpViewY = cy;
            })
            .on('end', function (event, d) {
              if (isGenerating || isEvaluating) return;
              if (evaluationZoomTransformRef.current && evaluationZoomTransformRef.current.k !== 1) {
                return;
              }
              const endX = d._tmpViewX ?? d3.pointer(event, g.node())[0];
              const endY = d._tmpViewY ?? d3.pointer(event, g.node())[1];

              // Only track drag if dragging actually occurred
              if (d._dragStarted) {
                // Convert screen coordinates back to metric values (account for metric scaling)
                const endXValue = xView.invert(endX);
                const endYValue = yView.invert(endY);


                // Record user drag target for visualization (with safety checks)
                if (d.id && !isNaN(endXValue) && !isNaN(endYValue) && xAxisMetric && yAxisMetric) {
                  setUserDragTargets(prev => ({
                    ...prev,
                    [d.id]: {
                      x: endXValue,
                      y: endYValue,
                      xAxisMetric: xAxisMetric,
                      yAxisMetric: yAxisMetric,
                      timestamp: Date.now()
                    }
                  }));
                }

                // Set flag to suppress automatic events after drag
                d._suppressEventsAfterDrag = true;
                // Clear the flag after a short delay to allow manual interactions
                setTimeout(() => {
                  delete d._suppressEventsAfterDrag;
                }, 100);
              }

              // Clean up temporary variables
              delete d._tmpViewX;
              delete d._tmpViewY;
              delete d._dragStarted;
              delete d._dragStartValues;

              /* ---------- ① 重叠检测：触发合并 ---------- */
              const overlap = drawable.find((n) =>
                n.id !== d.id &&
                Math.hypot(coordX(getX(n)) - endX, coordY(getY(n)) - endY) < 10
              );
              if (overlap) {
                setMergeTargetId(overlap.id);

                // Apply merge animation immediately when dragging A to B
                setNodes((prev) =>
                  prev.map((n) => {
                    if (n.id === d.id) {
                      // Make nodeA (dragged node) disappear
                      return { ...n, evaluationOpacity: 0 };
                    } else if (n.id === overlap.id) {
                      // Make nodeB (target node) bigger
                      return { ...n, isBeingMerged: true };
                    }
                    return n;
                  })
                );

                setPendingMerge({
                  nodeA: d,
                  nodeB: overlap,
                  screenX: event.sourceEvent.clientX + 10,
                  screenY: event.sourceEvent.clientY + 10,
                });

                // 回到原位
                d3.select(this).attr('transform', `translate(${coordX(getX(d))},${coordY(getY(d))})`);
                return;
              }

              // Keep the node at the dropped position visually
              d3.select(this).attr('transform', `translate(${endX},${endY})`);

              /* ---------- ② 原有拖拽修改逻辑 ---------- */
              const newXVal = xView.invert(endX);
              const newYVal = yView.invert(endY);
              const deltaX = newXVal - getX(d);
              const deltaY = newYVal - getY(d);

              let mods = [];
              let behind = null;
              const TOLERANCE = 5; // Only record changes larger than 5 points

              if (Math.abs(deltaX) > TOLERANCE) {
                mods.push({
                  metric: xAxisMetric,
                  previousScore: Math.round(getX(d)),
                  newScore: Math.round(newXVal),
                  change: Math.round(deltaX)
                });
                behind = nodes
                  .filter(
                    (n) => n.id !== d.id && getX(n) !== undefined && getX(n) < newXVal
                  )
                  .sort((a, b) => getX(b) - getX(a))[0];
              }
              if (Math.abs(deltaY) > TOLERANCE) {
                mods.push({
                  metric: yAxisMetric,
                  previousScore: Math.round(getY(d)),
                  newScore: Math.round(newYVal),
                  change: Math.round(deltaY)
                });
                behind =
                  behind ||
                  nodes
                    .filter(
                      (n) =>
                        n.id !== d.id && getY(n) !== undefined && getY(n) < newYVal
                    )
                    .sort((a, b) => getY(b) - getY(a))[0];
              }

              // After a short delay, snap back if no modification is triggered.
              setTimeout(() => {
                const currentDOMNode = d3.select(this);
                if (mods.length === 0) {
                  currentDOMNode.transition().duration(300)
                    .attr('transform', `translate(${coordX(getX(d))},${coordY(getY(d))})`);
                }
              }, 100);


              if (mods.length > 0) {
                // Snap the original node back; the ghost represents the drag target.
                d3.select(this).attr('transform', `translate(${coordX(getX(d))},${coordY(getY(d))})`);

                // Update _evalX and _evalY for the ghost node
                const ghost = {
                  ...d,
                  id: `${d.id}-Xghost-${Date.now()}`,
                  x: d.x,
                  y: d.y,
                  _evalX: newXVal,
                  _evalY: newYVal,
                  isGhost: true,
                  isNewlyGenerated: true,
                  level: d.level + 1,
                  _tmpViewX: endX,
                  _tmpViewY: endY,
                };
                setNodes((prev) => [...prev, ghost]);

                // Track ghost node creation
                setLinks((prev) => [...prev, { source: d.id, target: ghost.id }]);
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
              // Set flag to prevent hover tracking immediately after click
              d._justClicked = true;
              setTimeout(() => { d._justClicked = false; }, 200);

              setSelectedNode(d);
            })
        );

      // Add titles first (so they appear behind nodes by default)
      const label = nodeG.append('g')
        .attr('class', 'label-group')
        .style('pointer-events', 'none'); // The label should not capture mouse events

      // Add text first to measure it
      label.append('text')

        .attr('y', -20) // Position above the circle node (radius is 8, so -20 gives good spacing)
        .attr('text-anchor', 'middle')
        .style('font-family', 'Arial, sans-serif')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .style('fill', '#1f2937') // A dark gray for good contrast
        .style('opacity', d => {
          // Default transparency in plot view, solid when hovered or selected
          const isHovered = hoveredNode && hoveredNode.id === d.id;
          const isSelected = selectedNode && selectedNode.id === d.id;
          return isHovered || isSelected ? 1.0 : 0.2;
        })
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
            .attr('width', bbox.width + 2 * padding.x)
            .attr('height', bbox.height + 2 * padding.y)
            .attr('rx', 4)
            .attr('ry', 4)
            .style('fill', 'rgba(255, 255, 255, 0.9)')
            .style('stroke', '#e5e7eb')
            .style('stroke-width', 1)
            .style('opacity', () => {
              // Match the text opacity for backing rectangle
              const isHovered = hoveredNode && hoveredNode.id === d.id;
              const isSelected = selectedNode && selectedNode.id === d.id;
              return isHovered || isSelected ? 0.7 : 0.2;
            });
        }
      });

      // Add circles after labels (so they appear on top by default)
      nodeG
        .append('circle')
        .attr('r', (d) =>
          d.isBeingMerged ? 16 : d.id === mergeTargetId ? 12 : 8
        )
        .style('fill', (d) => getNodeColor(d))
        .style('opacity', (d) => {
          // Check if custom opacity is set
          if (d.evaluationOpacity !== undefined) return d.evaluationOpacity;
          // Otherwise use the default logic
          return d.isGhost ? 0.5 : 1;
        })
        .style('stroke', (d) =>
          selectedNode?.id === d.id ? '#000' : '#fff'
        )
        .style('stroke-width', (d) =>
          selectedNode?.id === d.id ? 4 : 2
        );

      // Add user drag target visualization
      const dragTargetData = nodes.filter(d => {
        if (!d || !d.userDragTarget) return false;

        // Check for data integrity
        if (typeof d.userDragTarget.x !== 'number' ||
          typeof d.userDragTarget.y !== 'number' ||
          !d.userDragTarget.xAxisMetric ||
          !d.userDragTarget.yAxisMetric) {
          console.log('[DEBUG] Invalid userDragTarget data for node:', d.id, d.userDragTarget);
          return false;
        }

        // Check if the axis metrics match what was recorded during drag
        if (d.userDragTarget.xAxisMetric !== xAxisMetric ||
          d.userDragTarget.yAxisMetric !== yAxisMetric) {
          console.log('[DEBUG] Axis metrics mismatch for node:', d.id,
            'recorded:', d.userDragTarget.xAxisMetric, d.userDragTarget.yAxisMetric,
            'current:', xAxisMetric, yAxisMetric);
          return false; // Don't show if axis metrics have changed
        }

        // Get current position using dynamic axis metrics
        const currentXValue = getX(d) || 0;
        const currentYValue = getY(d) || 0;

        const xDiff = Math.abs(currentXValue - d.userDragTarget.x);
        const yDiff = Math.abs(currentYValue - d.userDragTarget.y);

        console.log('[DEBUG] Drag target check for node:', d.id, {
          currentPos: [currentXValue, currentYValue],
          dragTarget: [d.userDragTarget.x, d.userDragTarget.y],
          differences: [xDiff, yDiff],
          willShow: xDiff > 1 || yDiff > 1
        });

        // Only show if current position differs from drag target
        return xDiff > 1 || yDiff > 1;
      });

      console.log('[DEBUG] dragTargetData count:', dragTargetData.length,
        'nodeIds:', dragTargetData.map(d => d.id));

      // Debug: Check all nodes for userDragTarget
      const nodesWithDragTargets = nodes.filter(d => d.userDragTarget);
      console.log('[DEBUG] Total nodes with userDragTarget:', nodesWithDragTargets.length);
      nodesWithDragTargets.forEach(d => {
        console.log('[DEBUG] Node with userDragTarget:', d.id, d.userDragTarget);
      });

      // Draw user drag target circles (where user originally wanted the node)
      if (dragTargetData.length > 0 && xView && yView) {
        g.selectAll('.drag-target-circle')
          .data(dragTargetData, d => d.id)
          .join('circle')
          .attr('class', 'drag-target-circle')
          .attr('cx', d => xView(d.userDragTarget.x))  // User's original target position
          .attr('cy', d => yView(d.userDragTarget.y))
          .attr('r', 6)
          .style('fill', 'none')
          .style('stroke', '#666')
          .style('stroke-width', 2)
          .style('stroke-dasharray', '4,4')
          .style('opacity', 0.85);

        // Draw connection lines from evaluation result (current node position) to user's original target
        g.selectAll('.drag-target-line')
          .data(dragTargetData, d => d.id)
          .join('line')
          .attr('class', 'drag-target-line')
          .attr('x1', d => xView(getX(d) || 0))  // Current evaluation position
          .attr('y1', d => yView(getY(d) || 0))
          .attr('x2', d => xView(d.userDragTarget.x))   // User's original target position
          .attr('y2', d => yView(d.userDragTarget.y))
          .style('stroke', '#666')
          .style('stroke-width', 1.5)
          .style('stroke-dasharray', '4,2')
          .style('opacity', 0.7);
      }
    }
    if (operationStatus) {
      // Add status text at top of plot instead of overlay
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '1.2rem')
        .style('font-weight', '600')
        .style('fill', '#374151')
        .style('background', 'rgba(255, 255, 255, 0.9)')
        .text(operationStatus);
    }
  }, [
    currentView,
    nodes,
    links,
    xAxisMetric,
    yAxisMetric,
    xAxisLabel,
    yAxisLabel,
    isGenerating,
    isEvaluating,
    operationStatus,
    selectedNode, // Added to trigger re-render when selecting to show/hide links
    mergeMode, // Added to trigger re-render when merge mode state changes
    unifiedScale, // re-render when unified scale changes
    xScalingCenter, // re-render when X scaling center changes
    yScalingCenter, // re-render when Y scaling center changes
    calculateNodeMidpoints,
    colorMap,
    getCurrentDimensions,
    getNodeColor,
    getNodeX,
    getNodeY,
    mergeTargetId,
    handleMergeConfirm,
    handleMergeCancel,
    hideEvaluationView, // trigger re-render when plot view visibility changes
    userDragTargets, // trigger re-render when drag targets change
    hoveredNode
  ]);

  // Separate useEffect for updating node hover/selection styling without recreating event listeners
  useEffect(() => {
    if (!svgRef.current || currentView === 'home_view') return;

    // Update circle styling for selection states only
    d3.select(svgRef.current)
      .selectAll('circle')
      .style('stroke', (d) =>
        d && d.id ? (selectedNode?.id === d.id ? '#000' : '#fff') : '#fff'
      )
      .style('stroke-width', (d) =>
        d && d.id ? (selectedNode?.id === d.id ? 4 : 2) : 2
      );
  }, [selectedNode, currentView]);

  // Separate useEffect for merge UI updates to prevent flickering
  useEffect(() => {
    if (!svgRef.current || currentView === 'home_view') return;

    const svg = d3.select(svgRef.current);
    let mergeUILayer = svg.select('.merge-ui-layer');

    // Clear existing merge UI elements
    if (!mergeUILayer.empty()) {
      mergeUILayer.selectAll('*').remove();
    }

    // Only render merge UI if in merge mode and in tree view
    if (mergeMode.active && mergeMode.firstNode && nodes.length > 0 && currentView === 'exploration') {
      // Ensure merge UI layer exists
      if (mergeUILayer.empty()) {
        mergeUILayer = svg.append('g').attr('class', 'merge-ui-layer');
      }

      // Apply current zoom transform to merge UI layer
      if (zoomTransformRef.current) {
        mergeUILayer.attr('transform', zoomTransformRef.current);
      }

      // Get the positioned layout nodes from ref
      const layoutNodes = layoutNodesRef.current;
      const firstNodeElement = layoutNodes ? layoutNodes.find(n => n.id === mergeMode.firstNode.id) : null;


      if (firstNodeElement) {
        // Always add special stroke around the first selected node
        mergeUILayer.append('circle')
          .attr('class', 'merge-first-node')
          .attr('cx', firstNodeElement.x)
          .attr('cy', firstNodeElement.y)
          .attr('r', 32)
          .style('fill', 'none')
          .style('stroke', 'rgb(192,192,192)')
          .style('stroke-width', 4)
          .style('pointer-events', 'none')
          .style('opacity', 1)
          .style('z-index', 1000);

        if (!mergeMode.secondNode) {
          // Draw line from first node to cursor
          mergeUILayer.append('path')
            .attr('class', 'merge-cursor-line')
            .attr('d', `M${firstNodeElement.x},${firstNodeElement.y} L${mergeMode.cursorPosition.x},${mergeMode.cursorPosition.y}`)
            .style('stroke', 'rgb(192,192,192)')
            .style('stroke-width', 2)
            .style('pointer-events', 'none')
            .style('opacity', 0.7);
        } else {
          // Draw line between first and second node
          const secondNodeElement = layoutNodes.find(n => n.id === mergeMode.secondNode.id);
          if (secondNodeElement) {
            // Add special stroke around second selected node
            mergeUILayer.append('circle')
              .attr('class', 'merge-second-node')
              .attr('cx', secondNodeElement.x)
              .attr('cy', secondNodeElement.y)
              .attr('r', 32)
              .style('fill', 'none')
              .style('stroke', 'rgb(192,192,192)')
              .style('stroke-width', 4)
              .style('pointer-events', 'none')
              .style('opacity', 1)
              .style('z-index', 1000);

            // Dashed line between nodes
            mergeUILayer.append('path')
              .attr('class', 'merge-connection-line')
              .attr('d', `M${firstNodeElement.x},${firstNodeElement.y} L${secondNodeElement.x},${secondNodeElement.y}`)
              .style('stroke', 'rgb(192,192,192)')
              .style('stroke-width', 3)
              .style('pointer-events', 'none');

            // Merge dialog in the middle
            if (mergeMode.showDialog) {
              const midX = (firstNodeElement.x + secondNodeElement.x) / 2;
              const midY = (firstNodeElement.y + secondNodeElement.y) / 2;

              // Dialog background with dashed border
              const dialogGroup = mergeUILayer.append('g')
                .attr('class', 'merge-dialog')
                .attr('data-panel-root', 'merge-dialog')
                .attr('transform', `translate(${midX}, ${midY})`);

              dialogGroup.append('rect')
                .attr('x', -50)
                .attr('y', -35)
                .attr('width', 100)
                .attr('height', 70)
                .attr('rx', 8)
                .style('fill', 'white')
                .style('stroke', 'rgb(192,192,192)')
                .style('stroke-width', 2)
                .style('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))');

              // Merge button (top)
              const mergeButton = dialogGroup.append('g')
                .attr('class', 'merge-button')
                .style('cursor', 'pointer');

              mergeButton.append('rect')
                .attr('x', -35)
                .attr('y', -25)
                .attr('width', 70)
                .attr('height', 20)
                .attr('rx', 4)
                .style('fill', '#4CAF50')
                .style('stroke', 'none');

              mergeButton.append('text')
                .attr('x', 0)
                .attr('y', -15)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .style('fill', 'white')
                .style('font-size', '12px')
                .style('font-weight', 'bold')
                .style('pointer-events', 'none')
                .text('Merge');

              // Cancel button (bottom)
              const cancelButton = dialogGroup.append('g')
                .attr('class', 'cancel-button')
                .style('cursor', 'pointer');

              cancelButton.append('rect')
                .attr('x', -35)
                .attr('y', 5)
                .attr('width', 70)
                .attr('height', 20)
                .attr('rx', 4)
                .style('fill', '#f44336')
                .style('stroke', 'none');

              cancelButton.append('text')
                .attr('x', 0)
                .attr('y', 15)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .style('fill', 'white')
                .style('font-size', '12px')
                .style('font-weight', 'bold')
                .style('pointer-events', 'none')
                .text('Cancel');

              // Add click handlers for buttons
              mergeButton.on('click', () => {
                handleMergeConfirm(mergeMode.firstNode, mergeMode.secondNode);
              });

              cancelButton.on('click', () => {
                handleMergeCancel();
              });
            }
          }
        }
      }
    }
  }, [
    mergeMode.active,
    mergeMode.firstNode,
    mergeMode.secondNode,
    mergeMode.showDialog,
    mergeMode.cursorPosition,
    nodes,
    currentView,
    handleMergeConfirm,
    handleMergeCancel
  ]);

  // Keyboard event listener for merge mode cancellation
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && mergeMode.active) {
        event.preventDefault();
        handleMergeCancel();
      }
    };

    if (mergeMode.active) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [mergeMode.active, handleMergeCancel]);

  // Unified useEffect for node and title layering
  useEffect(() => {
    // Exit early if the SVG isn't ready or we are in the home view
    if (!svgRef.current || currentView === 'home_view') return;

    const svg = d3.select(svgRef.current);

    // --- Tree View Logic ---
    // This logic remains unchanged.
    if (currentView === 'exploration') {
      if (selectedNode) {
        svg.selectAll('.node')
          .filter(d => d && d.id === selectedNode.id)
          .raise();
      }
      return; // End execution for tree view
    }

    // --- Plot View Logic ---
    // This simplified logic correctly prioritizes the layering rules.
    if (currentView === 'evaluation') {
      const allNodeGroups = svg.selectAll('.node-group');
      if (allNodeGroups.empty()) return;

      // --- Hover State (The Exception) ---
      // If a node is hovered, its entire group (title and circle) is
      // brought to the absolute front. This is the highest-priority action.
      if (hoveredNode) {
        allNodeGroups
          .filter(d => d && d.id === hoveredNode.id)
          .raise();
      } else {
        // --- Default & Selected State (The Rule) ---
        // In all non-hover states, the primary rule is that all circles
        // must be on top of all titles. This single line enforces that rule.
        // The selected node is distinguished by its stroke color, not by
        // layering its title over other circles.
        allNodeGroups.selectAll('circle').raise();
      }
    }
  }, [hoveredNode, selectedNode, currentView]);

  const handleAddCustomIdea = async (e) => {
    e.preventDefault();
    if (!customIdea.title.trim() || !customIdea.content.trim()) return;

    setIsGenerating(true);
    setOperationStatus('Adding custom idea...');
    setError(null);

    try {
      // Generate custom idea ID in C-1, C-2, C-3 format
      const newId = `C-${customIdeaCounter}`;
      setCustomIdeaCounter(prev => prev + 1);

      const newIdea = {
        id: newId,
        ...customIdea,
        // Add required fields for evaluation API
        Name: customIdea.title.trim(),
        Title: customIdea.title.trim(),
        Problem: customIdea.content.trim(),
        Importance: '',
        Difficulty: '',
        NoveltyComparison: '',
        Approach: ''
      };

      // 添加到假设列表
      const updatedIdeasList = [...ideasList, newIdea];
      setIdeasList(updatedIdeasList);

      // 创建新节点 - custom idea始终在第一层，不连接任何节点
      const newNode = {
        id: newId,
        level: 1, // 始终在第一层
        title: customIdea.title.trim(),
        content: customIdea.content.trim(),
        type: 'complex',
        x: 0, // 初始位置，后续会通过布局算法调整
        y: 150, // 初始位置，后续会通过布局算法调整
        originalData: newIdea,
      };

      console.log('[DEBUG] Created custom idea node:', {
        id: newNode.id,
        title: newNode.title,
        originalData: newNode.originalData
      });

      // 添加节点（不添加任何连接）
      setNodes((prev) => [...prev, newNode]);

      // Custom idea不连接任何节点，保持独立

      // 评估新假设
      await evaluateIdeas(updatedIdeasList, { allowAutoCenter: true });

      // 重置表单
      setCustomIdea({ title: '', content: '' });
      setIsAddingCustom(false);
    } catch (err) {
      console.error('Error adding custom idea:', err);
      setError(err.message);
    } finally {
      setIsGenerating(false);
      setOperationStatus('');
    }
  };

  // ============== Workflow: Code → Write → Review ==============
  const handleProceedWithSelectedIdea = async (node) => {
    if (!node || node.type === 'root' || node.type === 'fragment') return;
    const idea = node.originalData || { id: node.id, title: node.title, content: node.content };
    setWorkflowIdea(node);
    setWorkflowStep('coding');
    setWorkflowError(null);
    setCodeResult(null);
    setCodeFiles([]);
    setSelectedCodeFile(null);
    setShowWriterPrompt(false);
    setPaperResult(null);
    setReviewResult(null);
    setCurrentView('code_view');

    try {
      const response = await fetch('/api/code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ idea: { originalData: idea } }),
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Code generation failed');
      setCodeResult(data);
      setWorkflowStep('code_done');

      // Fetch file list from the experiment directory
      if (data.experiment_dir) {
        await fetchExperimentFiles(data.experiment_dir);
      }
    } catch (err) {
      setWorkflowError(err.message);
      setWorkflowStep('code_error');
    }
  };

  const fetchExperimentFiles = async (expDir) => {
    // Known files the coder always produces
    const candidates = [
      'experiment.py',
      'experiment_results.txt',
      'notes.txt',
    ];
    // Also check run_N.py variants (up to 5 runs)
    for (let i = 1; i <= 5; i++) {
      candidates.push(`run_${i}.py`);
      candidates.push(`run_${i}/final_info.json`);
    }

    const found = [];
    await Promise.all(
      candidates.map(async (name) => {
        const path = `${expDir}/${name}`;
        try {
          const r = await fetch(`/api/files/${path}`, { credentials: 'include' });
          if (r.ok) found.push({ name, path });
        } catch (_) { /* file absent, skip */ }
      })
    );

    // Sort: experiment.py first, then alphabetical
    found.sort((a, b) => {
      if (a.name === 'experiment.py') return -1;
      if (b.name === 'experiment.py') return 1;
      return a.name.localeCompare(b.name);
    });

    setCodeFiles(found);

    // Auto-open experiment.py if present
    const main = found.find(f => f.name === 'experiment.py') || found[0];
    if (main) await loadCodeFile(main);
  };

  const loadCodeFile = async (file) => {
    try {
      const r = await fetch(`/api/files/${file.path}`, { credentials: 'include' });
      const data = await r.json();
      setSelectedCodeFile({ ...file, content: data.content || '' });
    } catch (_) {
      setSelectedCodeFile({ ...file, content: '(failed to load file)' });
    }
  };

  const handleRerunCoder = async () => {
    if (!workflowIdea) return;
    const idea = workflowIdea.originalData || { id: workflowIdea.id, title: workflowIdea.title, content: workflowIdea.content };
    setWorkflowStep('coding');
    setWorkflowError(null);
    setCodeResult(null);
    setCodeFiles([]);
    setSelectedCodeFile(null);
    setShowWriterPrompt(false);
    setPaperResult(null);
    setReviewResult(null);

    try {
      const response = await fetch('/api/code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ idea: { originalData: idea } }),
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Code generation failed');
      setCodeResult(data);
      setWorkflowStep('code_done');
      if (data.experiment_dir) {
        await fetchExperimentFiles(data.experiment_dir);
      }
    } catch (err) {
      setWorkflowError(err.message);
      setWorkflowStep('code_error');
    }
  };

  const handleWritePaper = async () => {
    if (!workflowIdea) return;
    setShowWriterPrompt(false);
    setWorkflowStep('writing');
    setWorkflowError(null);
    setCurrentView('paper_view');
    const idea = workflowIdea.originalData || { id: workflowIdea.id, title: workflowIdea.title };
    try {
      const response = await fetch('/api/write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          idea: { originalData: idea },
          experiment_dir: codeResult?.experiment_dir || null,
          s2_api_key: s2ApiKey,
        }),
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Paper generation failed');
      setPaperResult(data);
      setWorkflowStep('paper_done');
    } catch (err) {
      setWorkflowError(err.message);
      setWorkflowStep('paper_error');
    }
  };

  const handleReviewPaper = async () => {
    if (!paperResult?.pdf_path) return;
    setWorkflowStep('reviewing');
    setWorkflowError(null);
    try {
      const response = await fetch('/api/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ pdf_path: paperResult.pdf_path, s2_api_key: s2ApiKey }),
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Review failed');
      setReviewResult(data.review);
      setWorkflowStep('review_done');
    } catch (err) {
      setWorkflowError(err.message);
      setWorkflowStep('review_error');
    }
  };

  // === 拖放处理函数 ===
  const handleNodeDropOnEvaluation = (e, targetNode) => {
    e.preventDefault();
    const draggedNodeId = e.dataTransfer.getData('nodeId');
    const isFragment = e.dataTransfer.getData('isFragment') === 'true';

    if (!draggedNodeId) return;

    // 从 nodes 或 fragmentNodes 查找被拖动的节点
    const draggedNode = nodes.find(n => n.id === draggedNodeId) ||
      fragmentNodes.find(f => f.id === draggedNodeId);

    if (!draggedNode) return;

    // Fragment 只能拖到节点上进行 merge，不能拖到空白区域
    if (isFragment && !targetNode) {
      console.log('[Fragment] Cannot drop fragment on empty space - fragments can only merge with nodes');
      return;
    }


    // 如果是 fragment 拖到另一个节点上，触发 merge
    if (isFragment && targetNode) {
      // 计算目标节点在屏幕上的位置(象限正下方)
      const targetElement = document.querySelector(`[data-node-id="${targetNode.id}"]`);
      let screenX = e.clientX; // Default to mouse position for 3D
      let screenY = e.clientY; // Default to mouse position for 3D

      if (targetElement) {
        const rect = targetElement.getBoundingClientRect();
        screenX = rect.left + rect.width / 2; // Center of node
        screenY = rect.bottom + 10; // 节点下方10px
      }

      // Adjust for modal width (centered)
      screenX = screenX - 150;

      const container = evaluationContainerRef.current;
      let ghostPosition = fragmentDragOriginRef.current;
      if (!ghostPosition && container) {
        const containerRect = container.getBoundingClientRect();
        ghostPosition = {
          x: e.clientX - containerRect.left,
          y: e.clientY - containerRect.top
        };
      }

      setDragVisualState({
        type: 'merge',
        sourceNodeId: draggedNode.id,
        targetNodeId: targetNode.id,
        ghostPosition
      });

      setPendingMerge({
        action: 'merge',
        sourceNode: draggedNode,
        targetNode: targetNode,
        isFragmentMerge: true,
        screenX,
        screenY
      });
      fragmentDragOriginRef.current = null;
    } else {
      // 显示操作选择：New Idea 或 Modify Eval
      setMergeTargetId(null);
      setPendingMerge({
        sourceNode: draggedNode,
        targetNode: targetNode, // 可能是 null（空白区域）
        action: 'evaluation_drag',
        isFragment
      });
    }
  };

  // ============== 3D Cube Navigation Logic ==============

  const clampPitch = useCallback((value) => Math.max(-90, Math.min(90, value)), []);

  const normalizeYaw = useCallback((value) => {
    let v = value % 360;
    if (v > 180) v -= 360;
    if (v <= -180) v += 360;
    return v;
  }, []);

  const snapRotation = useCallback((rotation) => {
    const snappedYaw = normalizeYaw(Math.round(rotation.yaw / 90) * 90);
    const snappedPitch = clampPitch(Math.round(rotation.pitch / 90) * 90);
    return { yaw: snappedYaw, pitch: snappedPitch };
  }, [clampPitch, normalizeYaw]);

  const getFaceFromRotation = useCallback((rotation) => {
    const yaw = normalizeYaw(rotation.yaw);
    const pitch = clampPitch(rotation.pitch);

    if (pitch > 45) return 2; // Top
    if (pitch < -45) return 5; // Bottom
    if (yaw >= 45 && yaw < 135) return 1; // Right
    if (yaw <= -45 && yaw > -135) return 4; // Left
    if (yaw >= 135 || yaw < -135) return 3; // Back
    return 0; // Front
  }, [clampPitch, normalizeYaw]);

  const beginSnapToFace = useCallback((rotation) => {
    const snapped = snapRotation(rotation);
    setIsSnapping(true);
    setCubeRotation(snapped);
    setCurrentFaceIndex(getFaceFromRotation(snapped));
    if (snapTimerRef.current) {
      clearTimeout(snapTimerRef.current);
    }
    snapTimerRef.current = setTimeout(() => setIsSnapping(false), 350);
  }, [getFaceFromRotation, snapRotation]);

  const trackCubeRotate = useCallback((source, fromRotation, toRotation) => {
  }, []);

  const handleStepRotation = useCallback((direction, source = 'unknown') => {
    let deltaYaw = 0;
    let deltaPitch = 0;
    if (direction === 'right') deltaYaw = 90;
    if (direction === 'left') deltaYaw = -90;
    if (direction === 'up') deltaPitch = 90;
    if (direction === 'down') deltaPitch = -90;

    const next = {
      yaw: cubeRotation.yaw + deltaYaw,
      pitch: clampPitch(cubeRotation.pitch + deltaPitch)
    };

    const snapped = snapRotation(next);
    trackCubeRotate(source, cubeRotation, snapped);
    beginSnapToFace(next);
  }, [beginSnapToFace, clampPitch, cubeRotation, snapRotation, trackCubeRotate]);

  // Add keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (currentView !== 'exploration') return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      switch (e.key) {
        case 'ArrowRight': handleStepRotation('right', 'keyboard'); break;
        case 'ArrowLeft': handleStepRotation('left', 'keyboard'); break;
        case 'ArrowUp': handleStepRotation('up', 'keyboard'); break;
        case 'ArrowDown': handleStepRotation('down', 'keyboard'); break;
        default: break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentView, handleStepRotation]);

  // Pointer Drag Handlers (use pointer capture so we always get the "up" event)
  const handleFacePointerDown = (e) => {
    if (e.button !== 0) return;
    if (e.target.closest('[data-node-id]')) return;
    e.preventDefault();

    cubeDragRef.current = {
      pointerId: e.pointerId,
      x: e.clientX,
      y: e.clientY,
      startYaw: cubeRotation.yaw,
      startPitch: cubeRotation.pitch,
      currentYaw: cubeRotation.yaw,
      currentPitch: cubeRotation.pitch,
      moved: false
    };
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setIsSnapping(false);
  };

  const handleFacePointerMove = (e) => {
    if (!cubeDragRef.current) return;
    if (cubeDragRef.current.pointerId !== e.pointerId) return;
    e.preventDefault();

    const dx = e.clientX - cubeDragRef.current.x;
    const dy = e.clientY - cubeDragRef.current.y;
    const sensitivity = 0.4;
    const yaw = normalizeYaw(cubeDragRef.current.startYaw + dx * sensitivity);
    const pitch = clampPitch(cubeDragRef.current.startPitch - dy * sensitivity);
    cubeDragRef.current.moved = true;
    cubeDragRef.current.currentYaw = yaw;
    cubeDragRef.current.currentPitch = pitch;
    setCubeRotation({ yaw, pitch });
  };

  const handleFacePointerUp = (e) => {
    if (!cubeDragRef.current) return;
    if (cubeDragRef.current.pointerId !== e.pointerId) return;
    e.preventDefault();
    e.currentTarget.releasePointerCapture?.(e.pointerId);

    const { moved, startYaw, startPitch, currentYaw, currentPitch } = cubeDragRef.current;
    cubeDragRef.current = null;
    if (!moved) return;

    const endRotation = {
      yaw: typeof currentYaw === 'number' ? currentYaw : cubeRotation.yaw,
      pitch: typeof currentPitch === 'number' ? currentPitch : cubeRotation.pitch
    };
    const snapped = snapRotation(endRotation);
    trackCubeRotate('drag', { yaw: startYaw, pitch: startPitch }, snapped);
    beginSnapToFace(endRotation);
  };

  const handleFacePointerCancel = (e) => {
    if (!cubeDragRef.current) return;
    if (cubeDragRef.current.pointerId !== e.pointerId) return;
    cubeDragRef.current = null;
  };

  // Render navigation buttons
  const renderNavigationButtons = () => (
    <>
      <div onClick={() => handleStepRotation('right', 'button')} style={navBtnStyle('right')}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6" /></svg>
      </div>
      <div onClick={() => handleStepRotation('left', 'button')} style={navBtnStyle('left')}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6" /></svg>
      </div>
      <div onClick={() => handleStepRotation('up', 'button')} style={navBtnStyle('top')}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M18 15l-6-6-6 6" /></svg>
      </div>
      <div onClick={() => handleStepRotation('down', 'button')} style={navBtnStyle('bottom')}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6" /></svg>
      </div>
    </>
  );

  const navBtnStyle = (pos) => {
    const base = {
      position: 'absolute',
      backgroundColor: '#ffffff',
      borderRadius: '4px', // Flatter
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      cursor: 'pointer',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      zIndex: 200,
      fontSize: '16px',
      userSelect: 'none',
      color: '#6b7280',
      transition: 'all 0.2s ease',
      border: '1px solid #e5e7eb'
    };

    // Flat, space-saving bars outside the cube
    if (pos === 'right') return {
      ...base,
      width: '24px', height: '60px',
      top: '50%', right: '-34px',
      transform: 'translateY(-50%)'
    };
    if (pos === 'left') return {
      ...base,
      width: '24px', height: '60px',
      top: '50%', left: '-34px',
      transform: 'translateY(-50%)'
    };
    if (pos === 'top') return {
      ...base,
      width: '60px', height: '24px',
      top: '-34px', left: '50%',
      transform: 'translateX(-50%)'
    };
    if (pos === 'bottom') return {
      ...base,
      width: '60px', height: '24px',
      bottom: '-34px', left: '50%',
      transform: 'translateX(-50%)'
    };
    return base;
  };

  // === CSS Grid 四象限布局渲染函数 (Planning.md 要求) ===
  const renderQuadrantLayout = () => {
    const { xDimension, yDimension } = getCurrentDimensions();

    if (!xDimension || !yDimension) {
      return (
        <div style={{
          width: '100%',
          height: '600px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#f9fafb',
          borderRadius: '8px',
          border: '2px dashed #d1d5db'
        }}>
          <div style={{ textAlign: 'center', color: '#6b7280' }}>
            <div style={{ fontSize: '1.2rem', marginBottom: '8px' }}>📊</div>
            <div>Please select 3 dimension pairs to visualize</div>
          </div>
        </div>
      );
    }

    const containerWidth = 800;
    const containerHeight = 600;

    // Helper to render a single face content
    const renderFaceContent = (faceIdx, isAnimating = false) => {
      // Re-calculate dimensions for this specific face index
      const targetFaceInfo = getCurrentDimensions(faceIdx);
      const { xDimension: xDim, yDimension: yDim } = targetFaceInfo;

      // Filter nodes for this face
      const faceNodes = nodes.filter(n => {
        const xScore = getNodeDimensionScore(n, xDim);
        const yScore = getNodeDimensionScore(n, yDim);
        return xScore !== undefined && yScore !== undefined && !n.isGhost;
      });

      const getArrowPoints = (from, to, size = 10, spread = 5) => {
        const dx = to.pixelX - from.pixelX;
        const dy = to.pixelY - from.pixelY;
        const len = Math.hypot(dx, dy);
        if (!len) return '';
        const ux = dx / len;
        const uy = dy / len;
        const baseX = to.pixelX - ux * size;
        const baseY = to.pixelY - uy * size;
        const perpX = -uy * spread;
        const perpY = ux * spread;
        return `${to.pixelX},${to.pixelY} ${baseX + perpX},${baseY + perpY} ${baseX - perpX},${baseY - perpY}`;
      };

      const billboardTransform = `rotateX(${-cubeRotation.pitch}deg) rotateY(${-cubeRotation.yaw}deg)`;
      const axisLabelBase = {
        position: 'absolute',
        backgroundColor: 'rgba(255,255,255,0.95)',
        padding: '4px 10px',
        borderRadius: '10px',
        fontSize: '12px',
        color: '#374151',
        zIndex: 2,
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
        backfaceVisibility: 'hidden',
        transformStyle: 'preserve-3d',
        pointerEvents: 'none',
        whiteSpace: 'nowrap'
      };

      // Precompute positions so merge visuals can reference targets
      const positionedNodes = faceNodes.map(node => {
        const baseXScore = getNodeDimensionScore(node, xDim);
        const baseYScore = getNodeDimensionScore(node, yDim);

        const finalXScore = baseXScore;
        const finalYScore = baseYScore;

        const normalizedX = (finalXScore + 50) / 100;
        const normalizedY = (finalYScore + 50) / 100;

        const pixelX = normalizedX * containerWidth * 0.9 + containerWidth * 0.05;
        const is1D = activeDimensionIndices.length === 1;
        const pixelY = is1D
          ? containerHeight / 2
          : (1 - normalizedY) * containerHeight * 0.9 + containerHeight * 0.05;

        return { node, pixelX, pixelY };
      });

      return (
        <div style={{
          width: '100%', height: '100%', position: 'absolute', top: 0, left: 0,
          backgroundColor: '#ffffff',
          backgroundImage: 'linear-gradient(135deg, rgba(240,244,255,0.9), #fff)',
          boxSizing: 'border-box',
          padding: '20px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gridTemplateRows: '1fr 1fr',
          gap: '2px',
          backfaceVisibility: 'hidden', // Important for 3D
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          boxShadow: '0 18px 40px rgba(0,0,0,0.12), inset 0 0 0 1px rgba(255,255,255,0.5)',
          pointerEvents: isAnimating ? 'none' : 'auto' // Disable interaction during animation
        }}>
          {/* Quadrant Lines */}
          <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: '1px', backgroundColor: '#e5e7eb', zIndex: 0 }} />
          {activeDimensionIndices.length > 1 && (
            <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: '1px', backgroundColor: '#e5e7eb', zIndex: 0 }} />
          )}

          {/* Labels */}
          {activeDimensionIndices.length > 1 && (
            <div style={{
              ...axisLabelBase,
              top: '-8px',
              left: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              writingMode: 'vertical-rl',
              transform: `translate(-50%, 0) ${billboardTransform}`
            }}>
              {yDim.dimensionB}
            </div>
          )}
          {activeDimensionIndices.length > 1 && (
            <div style={{
              ...axisLabelBase,
              bottom: '-8px',
              left: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              writingMode: 'vertical-rl',
              transform: `translate(-50%, 0) ${billboardTransform}`
            }}>
              {yDim.dimensionA}
            </div>
          )}
          <div style={{
            ...axisLabelBase,
            left: '-8px',
            top: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transform: `translate(0, -50%) ${billboardTransform}`
          }}>
            {xDim.dimensionA}
          </div>
          <div style={{
            ...axisLabelBase,
            right: '-8px',
            top: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transform: `translate(0, -50%) ${billboardTransform}`
          }}>
            {xDim.dimensionB}
          </div>

          {/* Nodes */}
          {positionedNodes.map(({ node, pixelX, pixelY }) => {
            const isSelected = selectedNode && selectedNode.id === node.id;
            const isHovered = hoveredNode && hoveredNode.id === node.id;

            // Drag Visual State Checks
            const activeMergeSourceId = dragVisualState?.type === 'merge'
              ? dragVisualState.sourceNodeId
              : mergeAnimationState?.sourceId;
            const activeMergeTargetId = dragVisualState?.type === 'merge'
              ? dragVisualState.targetNodeId
              : mergeAnimationState?.targetId;
            const activeGhostPosition = dragVisualState?.ghostPosition || mergeAnimationState?.ghostPosition;
            const isFragmentMergeSource = activeMergeSourceId
              ? fragmentNodes.some(f => f.id === activeMergeSourceId)
              : false;
            const targetPosition = activeMergeTargetId
              ? positionedNodes.find(p => p.node.id === activeMergeTargetId)
              : null;
            const isMergeSource = activeMergeSourceId === node.id;
            const isMergeTarget = activeMergeTargetId === node.id;
            const isModifySource = dragVisualState && dragVisualState.type === 'modify' && dragVisualState.sourceNodeId === node.id;
            const isMergeGrow = isMergeTarget;

            const overridePosition = (isModifySource && dragVisualState?.newPosition)
              ? { left: dragVisualState.newPosition.x - 25, top: dragVisualState.newPosition.y - 25 }
              : null;

            const shouldShowGhost = activeGhostPosition && (isMergeSource || (isMergeTarget && isFragmentMergeSource));
            const showModifyLine = isModifySource && dragVisualState?.ghostPosition && dragVisualState?.newPosition;
            const showMergeLine = activeGhostPosition && targetPosition && (isMergeSource || (isMergeTarget && isFragmentMergeSource));
            const circleBaseSize = 50;
            const circleHoverSize = 58;
            const circleMergeSize = 64;
            const circleSize = isMergeGrow ? circleMergeSize : (dragHoverTarget === node.id ? circleHoverSize : circleBaseSize);
            const circleOffset = isMergeGrow
              ? -(circleSize - circleBaseSize) / 2
              : (dragHoverTarget === node.id ? -(circleHoverSize - circleBaseSize) / 2 : 0);

            return (
              <React.Fragment key={node.id}>
                {/* Ghost Circle */}
                {shouldShowGhost && (
                  <div style={{
                    position: 'absolute',
                    left: activeGhostPosition.x - 25,
                    top: activeGhostPosition.y - 25,
                    width: '50px', height: '50px',
                    zIndex: 50, pointerEvents: 'none'
                  }}>
                    <div style={{
                      width: '100%', height: '100%', borderRadius: '50%',
                      border: '2px dashed #9ca3af', backgroundColor: 'rgba(156, 163, 175, 0.1)', opacity: 0.6
                    }} />
                  </div>
                )}

                {/* Modify Line */}
                {showModifyLine && (
                  <svg style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 55 }}>
                    <line
                      x1={dragVisualState.ghostPosition.x} y1={dragVisualState.ghostPosition.y}
                      x2={dragVisualState.newPosition.x} y2={dragVisualState.newPosition.y}
                      stroke="#fbbf24" strokeWidth={2} strokeDasharray="6,6"
                    />
                  </svg>
                )}
                {/* Merge Line */}
                {showMergeLine && targetPosition && (
                  <svg style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 55 }}>
                    <line
                      x1={activeGhostPosition.x} y1={activeGhostPosition.y}
                      x2={targetPosition.pixelX} y2={targetPosition.pixelY}
                      stroke="#9ca3af" strokeWidth={2} strokeDasharray="6,6"
                    />
                  </svg>
                )}

                {/* Actual Node */}
                <div
                  data-node-id={node.id}
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                  onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setDragHoverTarget(node.id); }}
                  onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragHoverTarget(null); }}
                  onDrop={(e) => { e.preventDefault(); e.stopPropagation(); setDragHoverTarget(null); handleNodeDropOnEvaluation(e, node); }}
                  style={{
                    position: 'absolute',
                    left: overridePosition ? overridePosition.left : pixelX - 25,
                    top: overridePosition ? overridePosition.top : pixelY - 25,
                    width: '50px', height: '50px',
                    zIndex: draggingNodeId === node.id ? 9999 : 100,
                    cursor: 'grab',
                    opacity: isMergeSource ? 0 : 1,
                    transform: billboardTransform,
                    transformStyle: 'preserve-3d',
                    backfaceVisibility: 'hidden'
                  }}
                  onMouseDown={(e) => {
                    e.preventDefault(); e.stopPropagation();
                    setDraggingNodeId(node.id);

                    const nodeElement = e.currentTarget;
                    // We use fixed positioning for dragging to escape the container's transform context if needed,
                    // but here we are inside a 3D transformed container.
                    // Fixed positioning might break the 3D illusion or coordinate system.
                    // However, the original code used fixed positioning.
                    // Let's stick to absolute within the face for now to keep it simple and 3D-compatible.
                    // If we use fixed, it will pop out of the 3D face.

                    // Actually, if we want to drag ACROSS faces, we might need fixed.
                    // But for now, let's assume dragging is within the face.

                    const startX = e.clientX;
                    const startY = e.clientY;
                    const startLeft = parseFloat(nodeElement.style.left);
                    const startTop = parseFloat(nodeElement.style.top);

                    let isDragging = true;
                    let hasMoved = false;

                    const handleMouseMove = (moveEvent) => {
                      if (!isDragging) return;
                      hasMoved = true;
                      const deltaX = moveEvent.clientX - startX;
                      const deltaY = moveEvent.clientY - startY;
                      nodeElement.style.left = `${startLeft + deltaX}px`;
                      nodeElement.style.top = `${startTop + deltaY}px`;
                      nodeElement.style.cursor = 'grabbing';
                    };

                    const handleMouseUp = (upEvent) => {
                      isDragging = false;
                      document.removeEventListener('mousemove', handleMouseMove);
                      document.removeEventListener('mouseup', handleMouseUp);
                      nodeElement.style.cursor = 'grab';

                      if (hasMoved) {
                        // Logic for drop (Merge or Modify)
                        // We need to calculate collision based on screen coordinates or relative coordinates
                        // The original code used screen coordinates for collision detection
                        const screenX = upEvent.clientX;
                        const screenY = upEvent.clientY;
                        const targetNode = detectNodeCollision(screenX, screenY, node.id, nodes);

                        if (targetNode) {
                          // Merge
                          setDragVisualState({ type: 'merge', sourceNodeId: node.id, targetNodeId: targetNode.id, ghostPosition: { x: startLeft + 25, y: startTop + 25 } });
                          setPendingMerge({ action: 'merge', sourceNode: node, targetNode: targetNode, screenX: screenX, screenY: screenY });
                          // Reset position
                          nodeElement.style.left = `${startLeft}px`;
                          nodeElement.style.top = `${startTop}px`;
                        } else {
                          // Modify
                          const currentLeft = parseFloat(nodeElement.style.left) + 25;
                          const currentTop = parseFloat(nodeElement.style.top) + 25;

                          // Convert back to score
                          const newScores = pixelToScore(currentLeft, currentTop, containerWidth, containerHeight, -50, 50, -50, 50);

                          // ... (Modify logic similar to original) ...
                          // Simplified for brevity, assuming pixelToScore works
                          const oldBaseXScore = getNodeDimensionScore(node, xDim);
                          const oldBaseYScore = getNodeDimensionScore(node, yDim);

                          const actualNewX = newScores.xScore;
                          const actualNewY = newScores.yScore;

                          setDragVisualState({ type: 'modify', sourceNodeId: node.id, ghostPosition: { x: startLeft + 25, y: startTop + 25 }, newPosition: { x: currentLeft, y: currentTop } });

                          const modifications = [
                            { metric: `${xDim.dimensionA}-${xDim.dimensionB}`, previousScore: oldBaseXScore, newScore: actualNewX, change: actualNewX - (oldBaseXScore ?? 0) },
                            { metric: `${yDim.dimensionA}-${yDim.dimensionB}`, previousScore: oldBaseYScore, newScore: actualNewY, change: actualNewY - (oldBaseYScore ?? 0) }
                          ];

                          const ghostId = `${node.id}-Xghost-${Date.now()}`;
                          const ghostNode = { ...node, id: ghostId, scores: { ...node.scores, [`${xDim.dimensionA}-${xDim.dimensionB}`]: actualNewX, [`${yDim.dimensionA}-${yDim.dimensionB}`]: actualNewY }, isGhost: true, isModified: true };

                          setPendingChange({ originalNode: node, ghostNode: ghostNode, modifications: modifications, behindNode: null, screenX: screenX, screenY: screenY, draggedElement: nodeElement, finalPosition: { x: currentLeft, y: currentTop } });
                        }
                      } else {
                        setSelectedNode(node);
                      }
                      setDraggingNodeId(null);
                    };
                    document.addEventListener('mousemove', handleMouseMove);
                    document.addEventListener('mouseup', handleMouseUp);
                  }}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  {/* Title */}
                  <div style={{
                    position: 'absolute', left: '50%', top: '-36px', transform: 'translateX(-50%)',
                    fontSize: '12px', fontWeight: '600', color: '#1f2937', backgroundColor: '#f3f4f6',
                    border: '1.5px solid #d1d5db', padding: '4px 8px', borderRadius: '4px', whiteSpace: 'nowrap',
                    boxShadow: isHovered || isSelected ? '0 2px 8px rgba(0,0,0,0.15)' : 'none',
                    pointerEvents: 'none', zIndex: 102, display: 'block'
                  }}>
                    {node.title.substring(0, 20)}
                  </div>
                  {/* Circle */}
                  <div style={{
                    width: `${circleSize}px`,
                    height: `${circleSize}px`,
                    borderRadius: '50%',
                    backgroundColor: node.isMergedResult ? '#B22222' : (node.isNewlyGenerated ? '#FFD700' : (colorMap[node.type] || '#FF6B6B')),
                    border: (dragHoverTarget === node.id ? '3px solid #fbbf24' : (isSelected ? '4px solid #000' : '2px solid #fff')),
                    boxShadow: (dragHoverTarget === node.id ? '0 0 20px rgba(251, 191, 36, 0.8)' : '0 2px 4px rgba(0,0,0,0.1)'),
                    transition: 'all 0.2s ease',
                    opacity: node.isGhost ? 0.5 : 1,
                    transform: circleOffset ? `translate(${circleOffset}px, ${circleOffset}px)` : 'none'
                  }} />
                </div>
              </React.Fragment>
            );
          })}

          {/* Merge connectors after merge is done (selection based) */}
          {(() => {
            if (!selectedNode) return null;
            const selectedId = selectedNode.id;

            const mergedInfos = nodes
              .filter(n => n.isMergedResult)
              .map(m => {
                const parents = new Set();
                if (Array.isArray(m.parentIds)) {
                  m.parentIds.forEach(id => parents.add(id));
                }
                links.forEach(lk => {
                  if (lk.target === m.id) parents.add(lk.source);
                  if (lk.source === m.id) parents.add(lk.target);
                });
                return { mergedId: m.id, parentIds: Array.from(parents) };
              });

            let relation = mergedInfos.find(info => info.mergedId === selectedId && info.parentIds.length > 0);
            if (!relation) {
              relation = mergedInfos.find(info => info.parentIds.includes(selectedId));
            }
            if (!relation) return null;

            const mergedPos = positionedNodes.find(p => p.node.id === relation.mergedId);
            if (!mergedPos) return null;
            const parentPositions = relation.parentIds
              .map(pid => positionedNodes.find(p => p.node.id === pid))
              .filter(Boolean);
            if (parentPositions.length === 0) return null;

            return (
              <svg style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 60 }}>
                {parentPositions.map((p, idx) => (
                  <g key={`${p.node.id}-${idx}`}>
                    <line
                      x1={p.pixelX} y1={p.pixelY}
                      x2={mergedPos.pixelX} y2={mergedPos.pixelY}
                      stroke="#9ca3af" strokeWidth={2}
                    />
                    <polygon points={getArrowPoints(p, mergedPos)} fill="#9ca3af" opacity={0.9} />
                  </g>
                ))}
              </svg>
            );
          })()}
        </div>
      );
    };

    // Determine mode
    const numActive = activeDimensionIndices.length;
    const is3D = numActive >= 3;

    // For 2D/1D, we only show the first active dimension pair
    const targetFaceIndex = is3D ? currentFaceIndex : activeDimensionIndices[0];

    const containerStyle = {
      width: containerWidth,
      height: containerHeight,
      position: 'relative',
      perspective: '1200px', // Perspective on the container
      margin: '0 auto', // Center it
      touchAction: is3D ? 'none' : undefined
    };

    const cubeWrapperStyle = {
      width: '100%',
      height: '100%',
      position: 'absolute',
      transformStyle: 'preserve-3d',
      transition: isSnapping ? 'transform 0.35s ease-out' : 'transform 0s linear',
      transform: (() => {
        const baseTransform = `translateZ(-${containerWidth / 2}px)`;
        return `${baseTransform} rotateY(${cubeRotation.yaw}deg) rotateX(${cubeRotation.pitch}deg)`;
      })()
    };

    const translateZ_X = containerWidth / 2;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', gap: '50px' }}>
        <div style={{ position: 'relative', width: '100%', height: containerHeight, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'visible' }}>
          {/* Cube Container */}
          <div style={containerStyle}
            data-uatrack-suppress-drag={is3D ? 'true' : undefined}
            onPointerDown={is3D ? handleFacePointerDown : undefined}
            onPointerUp={is3D ? handleFacePointerUp : undefined}
            onPointerMove={is3D ? handleFacePointerMove : undefined}
            onPointerCancel={is3D ? handleFacePointerCancel : undefined}
          >
            {/* Navigation Buttons - Only show in 3D mode */}
            {is3D && renderNavigationButtons()}

            <div style={is3D ? cubeWrapperStyle : { width: '100%', height: '100%', position: 'relative' }}>
              {/* Current Face (or the only face in 2D/1D) */}
              <div style={is3D ? {
                position: 'absolute', width: '100%', height: '100%',
                transform: `translateZ(${translateZ_X}px)`,
                backfaceVisibility: 'hidden'
              } : { width: '100%', height: '100%', position: 'absolute' }}>
                {renderFaceContent(targetFaceIndex, isSnapping)}
              </div>
            </div>
          </div>
        </div>

        {/* Fragment Bar */}
        <div style={{
          width: containerWidth,
          minHeight: '120px',
          maxHeight: '300px',
          marginTop: '0', // Controlled by parent flex gap/margin
          backgroundColor: '#fffbeb',
          border: '2px dashed #f59e0b',
          borderRadius: '8px',
          padding: '12px',
          display: 'flex',
          gap: '12px',
          overflowX: 'auto',
          overflowY: 'auto',
          alignItems: 'flex-start',
          boxSizing: 'border-box'
        }}>
          {fragmentNodes.length === 0 ? (
            <div style={{ fontSize: '0.875rem', color: '#92400e', fontStyle: 'italic', margin: 'auto', textAlign: 'center' }}>
              Select text in any idea card and click "Fragment" to create quick notes here
            </div>
          ) : (
            fragmentNodes.map(fragment => {
              const isExpanded = expandedFragmentId === fragment.id;
              const parentNode = nodes.find(n => n.id === fragment.parentId);
              const parentTitle = parentNode ? parentNode.title : fragment.parentId;
              const isMergeSource = dragVisualState && dragVisualState.type === 'merge' && dragVisualState.sourceNodeId === fragment.id;
              const showPendingOutline = isMergeSource || fragment.isPendingMerge;

              return (
                <div key={fragment.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '60px', maxWidth: isExpanded ? '220px' : '60px', flexShrink: 0, transition: 'max-width 0.3s ease' }}>
                  <div
                    draggable={true}
                    onDragStart={(e) => {
                      e.dataTransfer.setData('nodeId', fragment.id);
                      e.dataTransfer.setData('isFragment', 'true');
                      const container = evaluationContainerRef.current;
                      if (container) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const containerRect = container.getBoundingClientRect();
                        fragmentDragOriginRef.current = {
                          x: rect.left - containerRect.left + rect.width / 2,
                          y: rect.top - containerRect.top + rect.height / 2
                        };
                      } else {
                        fragmentDragOriginRef.current = null;
                      }
                      const dragImage = document.createElement('div');
                      dragImage.style.width = '50px'; dragImage.style.height = '50px'; dragImage.style.backgroundColor = '#fef08a';
                      dragImage.style.border = '2px solid #f59e0b'; dragImage.style.borderRadius = '50%';
                      dragImage.style.display = 'flex'; dragImage.style.alignItems = 'center'; dragImage.style.justifyContent = 'center';
                      dragImage.style.fontSize = '0.875rem'; dragImage.style.fontWeight = '700'; dragImage.style.color = '#92400e';
                      dragImage.style.position = 'fixed'; dragImage.style.top = '-1000px'; dragImage.style.left = '-1000px';
                      document.body.appendChild(dragImage);
                      e.dataTransfer.setDragImage(dragImage, 25, 25);
                      setTimeout(() => { document.body.removeChild(dragImage); }, 0);
                    }}
                    onDragEnd={() => {
                      fragmentDragOriginRef.current = null;
                    }}
                    onClick={() => {
                      setExpandedFragmentId(isExpanded ? null : fragment.id);
                      if (!isExpanded && parentNode) setSelectedNode(parentNode);
                    }}
                    style={{
                      minWidth: '50px', width: '50px', height: '50px', backgroundColor: '#fef08a',
                      border: showPendingOutline ? '3px dashed #fbbf24' : '2px solid #f59e0b',
                      borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                      cursor: 'pointer', fontSize: '0.875rem', fontWeight: '700', color: '#92400e',
                      position: 'relative', flexShrink: 0, marginBottom: isExpanded ? '8px' : '0',
                      opacity: showPendingOutline ? 0.3 : (fragment.evaluationOpacity !== undefined ? fragment.evaluationOpacity : 1),
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteFragmentNode(fragment.id); if (expandedFragmentId === fragment.id) setExpandedFragmentId(null); }}
                      style={{ position: 'absolute', top: '-6px', right: '-6px', width: '18px', height: '18px', borderRadius: '50%', backgroundColor: '#ef4444', color: 'white', border: 'none', cursor: 'pointer', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0, fontWeight: 'bold' }}
                    >
                      ×
                    </button>
                  </div>
                  {isExpanded && (
                    <div style={{ width: '100%', padding: '10px', backgroundColor: 'white', border: '1px solid #fbbf24', borderRadius: '6px', fontSize: '0.75rem', lineHeight: 1.4, color: '#374151', maxHeight: '150px', overflowY: 'auto', wordBreak: 'break-word', cursor: 'default', animation: 'fadeIn 0.2s ease-in' }}>
                      <div style={{ fontSize: '0.7rem', color: '#78716c', marginBottom: '6px', fontWeight: '600' }}>Fragment of {parentTitle}</div>
                      <div style={{ fontSize: '0.7rem', color: '#6b7280', lineHeight: 1.5 }}>{fragment.content}</div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  };

  void renderQuadrantLayout;

  // === 3D Drag Handler ===
  const handle3DNodeDragEnd = (nodeId, payload, event) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const { scoresMap = {}, mergeTargetId = null, clientX = event?.clientX, clientY = event?.clientY, cancelHold = false } = payload || {};

    // Determine active dimensions
    let dims = [];
    if (selectedDimensionPairs && selectedDimensionPairs.length >= 1) {
      dims = selectedDimensionPairs.slice(0, 3);
    }

    const modifications = [];

    // Helper to get old score
    const getScore = (dimObj) => {
      if (!dimObj) return 0;
      const v = getNodeDimensionScore(node, dimObj);
      if (v === undefined || v === null) return 0;
      return v;
    };

    // Check modifications for each dimension
    dims.forEach((dim, index) => {
      if (!dim) return;
      const dimKey = `${dim.dimensionA}-${dim.dimensionB}`;
      const newValue = scoresMap[dimKey];

      if (newValue !== undefined) {
        const oldVal = getScore(dim);
        if (Math.abs(newValue - oldVal) > 2) {
          modifications.push({
            metric: dimKey,
            previousScore: Math.round(oldVal),
            newScore: Math.round(newValue),
            change: Math.round(newValue - oldVal)
          });
        }
      }
    });

    if (!cancelHold) {
    }

    if (cancelHold) {
      setPendingChange(null);
      setPendingMerge(null);
      setDragVisualState(null); // Clear drag visual on cancel
    } else if (mergeTargetId) {
      const targetNode = nodes.find(n => n.id === mergeTargetId);
      if (targetNode) {
        setPendingMerge({
          action: 'merge',
          sourceNode: node,
          targetNode,
          isFragmentMerge: false,
          screenX: clientX ?? window.innerWidth / 2,
          screenY: clientY ?? window.innerHeight / 2
        });
      }
    } else if (modifications.length > 0) {
      // Create ghost node
      const ghostId = `${node.id}-Xghost-${Date.now()}`;
      const ghostNode = {
        ...node,
        id: ghostId,
        scores: { ...(node.scores || {}) },
        isGhost: true,
        isModified: true
      };

      // Update ghost scores
      modifications.forEach(mod => {
        ghostNode.scores[mod.metric] = mod.newScore;
      });

      // Use event coordinates for popup
      const screenX = event ? event.clientX : window.innerWidth / 2;
      const screenY = event ? event.clientY : window.innerHeight / 2;

      // Add ghost node to state so it renders
      setNodes(prev => [...prev, ghostNode]);

      setPendingChange({
        originalNode: node,
        ghostNode: ghostNode,
        modifications: modifications,
        behindNode: null,
        screenX: screenX,
        screenY: screenY
      });
    } else {
      // No modifications - clear drag visual state
      setDragVisualState(null);
    }
  };

  const handle3DNodeDragEndRef = useRef(null);
  handle3DNodeDragEndRef.current = handle3DNodeDragEnd;

  const cancelPendingChange = useCallback((change, { reason = 'manual' } = {}) => {
    if (!change) return;


    const nodeElement = change.draggedElement || document.querySelector(`[data-node-id="${change.originalNode.id}"]`);
    if (nodeElement) {
      nodeElement.style.position = 'absolute';
      if (dragVisualState?.ghostPosition) {
        nodeElement.style.left = `${dragVisualState.ghostPosition.x - 25}px`;
        nodeElement.style.top = `${dragVisualState.ghostPosition.y - 25}px`;
      }
      nodeElement.style.zIndex = '100';
    }

    setDragVisualState(null);

    if (change.ghostNode?.id) {
      const updatedNodesAfterCancel = nodes.filter((nd) => nd.id !== change.ghostNode.id);
      setNodes(updatedNodesAfterCancel);
      setLinks((prev) => prev.filter((lk) => lk.target !== change.ghostNode.id));
    }

    setPendingChange(null);

    if (currentView === 'exploration') {
      handle3DNodeDragEndRef.current(change.originalNode.id, { cancelHold: true });
    }
  }, [currentView, dragVisualState, nodes]);

  const cancelPendingMerge = useCallback((merge) => {
    if (!merge) return;

    const sourceNode = merge.sourceNode || merge.nodeA;
    const targetNode = merge.targetNode || merge.nodeB;

    if (merge.action === 'evaluation_drag') {
      if (sourceNode && dragVisualState?.ghostPosition) {
        const nodeElement = document.querySelector(`[data-node-id="${sourceNode.id}"]`);
        if (nodeElement) {
          nodeElement.style.left = `${dragVisualState.ghostPosition.x - 25}px`;
          nodeElement.style.top = `${dragVisualState.ghostPosition.y - 25}px`;
        }
      }

      if (sourceNode || targetNode) {
        setNodes((prev) =>
          prev.map((n) => {
            if (n.id === sourceNode?.id && n.evaluationOpacity === 0) {
              return { ...n, evaluationOpacity: 1 };
            }
            if (n.id === targetNode?.id && n.isBeingMerged) {
              return { ...n, isBeingMerged: false };
            }
            return n;
          })
        );
      }

      setDragVisualState(null);
      setMergeAnimationState(null);
      setPendingMerge(null);
      return;
    }

    setDragVisualState(null);

    if (sourceNode && targetNode) {
      const isSourceFragment = fragmentNodes.some(f => f.id === sourceNode.id);

      if (isSourceFragment) {
        setFragmentNodes(prev => prev.map(f => {
          if (f.id === sourceNode.id) {
            const updated = { ...f, evaluationOpacity: 1 };
            delete updated.isPendingMerge;
            return updated;
          }
          return f;
        }));
        setNodes(prev => prev.map(n => {
          if (n.id === targetNode.id) {
            return { ...n, isBeingMerged: false };
          }
          return n;
        }));
      } else {
        const updatedNodesAfterCancelMerge = nodes.map(n => {
          if (n.id === sourceNode.id) {
            return { ...n, evaluationOpacity: 1 };
          }
          if (n.id === targetNode.id) {
            return { ...n, isBeingMerged: false };
          }
          return n;
        });
        setNodes(updatedNodesAfterCancelMerge);
      }
    }

    setMergeTargetId(null);
    setPendingMerge(null);
    setMergeAnimationState(null);
  }, [dragVisualState, fragmentNodes, nodes]);

  const last3DHoverIdRef = useRef(null);
  const handle3DNodeHover = (node) => {
    const nextId = node?.id || null;
    const prevId = last3DHoverIdRef.current;
    if (prevId && prevId !== nextId) {
      const prevNode = nodes.find(n => n.id === prevId);
      if (prevNode) {
      }
    }
    if (nextId && prevId !== nextId) {
    }
    last3DHoverIdRef.current = nextId;
    setHoveredNode(node);
  };

  const handle3DNodeClick = (node) => {
    if (!node) return;
    setSelectedNode(node);
  };

  const activeDimensions = getCurrentDimensions();

  // Toggle dimension index active state
  const toggleDimensionIndex = (index) => {
    setActiveDimensionIndices(prev => {
      if (prev.includes(index)) {
        // Don't allow unchecking the last one (at least 1 dimension needed)
        if (prev.length <= 1) return prev;
        return prev.filter(i => i !== index);
      } else {
        return [...prev, index].sort();
      }
    });
  };

  const extractScoreValue = (scoreEntry) => {
    if (scoreEntry === null || scoreEntry === undefined) return undefined;
    if (typeof scoreEntry === 'number') return scoreEntry;
    if (typeof scoreEntry === 'string' && scoreEntry.trim() !== '' && !Number.isNaN(Number(scoreEntry))) {
      return Number(scoreEntry);
    }
    if (typeof scoreEntry === 'object' && scoreEntry.value !== undefined) {
      if (typeof scoreEntry.value === 'number') return scoreEntry.value;
      if (typeof scoreEntry.value === 'string' && scoreEntry.value.trim() !== '' && !Number.isNaN(Number(scoreEntry.value))) {
        return Number(scoreEntry.value);
      }
    }
    return undefined;
  };

  const orientScoreValue = (value, flipped) => {
    if (!flipped || typeof value !== 'number') return value;
    if (value >= -50 && value <= 50) return -value;
    if (value >= 0 && value <= 100) return 100 - value;
    return -value;
  };

  const cloneWithSwappedScore = (node, fromKey, toKey) => {
    if (!fromKey || !toKey || fromKey === toKey) return node;
    let updatedNode = node;

    const maybeUpdateScores = (scoresObj) => {
      if (!scoresObj || scoresObj[toKey] !== undefined || scoresObj[fromKey] === undefined) return scoresObj;
      const baseEntry = scoresObj[fromKey];
      const baseValue = extractScoreValue(baseEntry);
      if (baseValue === undefined) return scoresObj;

      const flippedValue = orientScoreValue(baseValue, true);
      const newEntry = (typeof baseEntry === 'object' && baseEntry !== null && baseEntry.value !== undefined)
        ? { ...baseEntry, value: flippedValue }
        : flippedValue;
      return { ...scoresObj, [toKey]: newEntry };
    };

    const newScores = maybeUpdateScores(node.scores);
    const newOriginalScores = maybeUpdateScores(node.originalData?.scores);

    if (newScores !== node.scores || newOriginalScores !== node.originalData?.scores) {
      updatedNode = { ...node };
      if (newScores !== node.scores) {
        updatedNode.scores = newScores;
      }
      if (newOriginalScores !== node.originalData?.scores) {
        updatedNode.originalData = { ...(node.originalData || {}), scores: newOriginalScores };
      }
    }

    return updatedNode;
  };

  // Swap the direction of a dimension pair (A ↔ B)
  const swapDimensionDirection = (pairIndex) => {
    setSelectedDimensionPairs(prev => {
      const updated = [...prev];
      const pair = updated[pairIndex];
      if (!pair) return prev;

      const swapped = {
        ...pair,
        dimensionA: pair.dimensionB,
        dimensionB: pair.dimensionA
      };
      updated[pairIndex] = swapped;

      const fromKey = `${pair.dimensionA}-${pair.dimensionB}`;
      const toKey = `${swapped.dimensionA}-${swapped.dimensionB}`;

      // Ensure all nodes carry the swapped key so downstream code that expects the "primary" key still works
      setNodes(prevNodes => prevNodes.map(n => cloneWithSwappedScore(n, fromKey, toKey)));
      setFragmentNodes(prevFragments => prevFragments.map(f => cloneWithSwappedScore(f, fromKey, toKey)));

      return updated;
    });
  };

  // *** 维度编辑相关函数 ***
  const handleEditDimension = (pairIndex, anchorRect) => {
    setEditingDimensionIndex(pairIndex);
    setDimensionDropdownAnchor(anchorRect);
  };

  const handleCloseDimensionEdit = useCallback(() => {
    setEditingDimensionIndex(null);
    setDimensionDropdownAnchor(null);
  }, []);

  const cancelActivePanels = useCallback((reason = 'auto') => {
    if (pendingChange) cancelPendingChange(pendingChange, { reason });
    if (pendingMerge) cancelPendingMerge(pendingMerge);
    if (mergeMode.showDialog) handleMergeCancel();
    if (fragmentMenuState) hideFragmentMenu();
    if (showDimensionPanel) setShowDimensionPanel(false);
    if (editingDimensionIndex !== null) handleCloseDimensionEdit();
    if (isEditingSystemPrompt) {
      setIsEditingSystemPrompt(false);
      setModalAnchorEl(null);
    }
  }, [
    cancelPendingChange,
    cancelPendingMerge,
    fragmentMenuState,
    handleCloseDimensionEdit,
    handleMergeCancel,
    hideFragmentMenu,
    isEditingSystemPrompt,
    editingDimensionIndex,
    mergeMode.showDialog,
    pendingChange,
    pendingMerge,
    showDimensionPanel
  ]);

  const hasActivePanel = Boolean(
    pendingChange ||
    pendingMerge ||
    mergeMode.showDialog ||
    fragmentMenuState ||
    showDimensionPanel ||
    editingDimensionIndex !== null ||
    isEditingSystemPrompt
  );

  useEffect(() => {
    if (!hasActivePanel) return;

    const handlePointerDown = (event) => {
      const rawTarget = event.target;
      const target = rawTarget && rawTarget.nodeType === 3 ? rawTarget.parentElement : rawTarget;
      if (target?.closest?.('[data-panel-root]')) return;
      cancelActivePanels('auto');
    };

    document.addEventListener('pointerdown', handlePointerDown, true);
    return () => document.removeEventListener('pointerdown', handlePointerDown, true);
  }, [cancelActivePanels, hasActivePanel]);

  const handleDimensionEditConfirm = async (newPair, pairIndex) => {
    const oldPair = selectedDimensionPairs[pairIndex];
    if (!oldPair) return;

    const hasChanges = oldPair.dimensionA !== newPair.dimensionA ||
      oldPair.dimensionB !== newPair.dimensionB ||
      oldPair.descriptionA !== newPair.descriptionA ||
      oldPair.descriptionB !== newPair.descriptionB;

    // 1. 更新 selectedDimensionPairs
    const updatedPairs = [...selectedDimensionPairs];
    updatedPairs[pairIndex] = newPair;
    setSelectedDimensionPairs(updatedPairs);

    // 2. 迁移 score key
    // Note: If description changed but dimension name didn't, key remains same, but we still re-evaluate
    if (hasChanges) {
      const oldKey = `${oldPair.dimensionA}-${oldPair.dimensionB}`;
      const newKey = `${newPair.dimensionA}-${newPair.dimensionB}`;

      if (oldKey !== newKey) {
        setNodes(prevNodes => prevNodes.map(node => {
          if (!node.scores || node.scores[oldKey] === undefined) return node;
          const newScores = { ...node.scores };
          newScores[newKey] = newScores[oldKey];
          delete newScores[oldKey];
          return {
            ...node,
            scores: newScores,
            originalData: node.originalData ? {
              ...node.originalData,
              scores: newScores,
            } : node.originalData,
          };
        }));
      }

      // 3. 调用单维度评分 API
      await evaluateSingleDimension(newPair, pairIndex);
    }

    // 4. 关闭弹窗
    handleCloseDimensionEdit();
  };

  const evaluateSingleDimension = async (dimensionPair, dimensionIndex) => {
    setIsSingleDimensionEvaluating(true);

    try {
      const requestIdeas = nodes
        .filter(n => n.level !== 0 && n.type !== 'fragment')
        .map(n => ({
          ...n.originalData,
          id: n.originalData?.id || n.id,
          Title: n.originalData?.Title || n.title,
        }));

      const response = await fetch('/api/evaluate-dimension', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          ideas: requestIdeas,
          intent: currentIntent || analysisIntent,
          dimension_pair: dimensionPair,
          dimension_index: dimensionIndex,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to evaluate dimension: ${response.status}`);
      }

      const result = await response.json();
      const dimKey = `${dimensionPair.dimensionA}-${dimensionPair.dimensionB}`;
      const dimScoreKey = `Dimension${dimensionIndex + 1}Score`;
      const dimReasonKey = `Dimension${dimensionIndex + 1}Reason`;

      // 更新 nodes 中的单维度分数
      setNodes(prevNodes => prevNodes.map(node => {
        const evalResult = result.scores?.find(s => s.id === (node.originalData?.id || node.id));
        if (!evalResult) return node;

        return {
          ...node,
          scores: {
            ...node.scores,
            [dimKey]: evalResult.score,
          },
          [dimScoreKey]: evalResult.score,
          [dimReasonKey]: evalResult.reason,
          originalData: node.originalData ? {
            ...node.originalData,
            scores: {
              ...(node.originalData.scores || {}),
              [dimKey]: evalResult.score,
            },
            [dimReasonKey]: evalResult.reason,
          } : node.originalData,
        };
      }));

    } catch (err) {
      console.error('Error evaluating single dimension:', err);
      setError(`Failed to evaluate dimension: ${err.message}`);
    } finally {
      setIsSingleDimensionEvaluating(false);
    }
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', position: 'relative' }}>

      {/* 把 showTree 与 setShowTree 传给 TopNav，解决 setShowTree is not a function */}
      <TopNav
        currentView={currentView}
        setCurrentView={setCurrentView}
        hideEvaluationView={hideEvaluationView}
        onReEvaluateAll={() => reEvaluateAll()}
        isEvaluating={isEvaluating}
        showCodeView={!!workflowStep}
      />

      {currentView === 'home_view' ? (
        <OverviewPage />
      ) : (
        <>
          {currentView === 'exploration' && !isAnalysisSubmitted && (
            <div
              style={{
                backgroundColor: '#1f2937',
                display: 'flex',
                flexDirection: 'column',
                padding: '10px 20px',
                position: 'relative', // 重要: 让下拉面板相对于此容器定位
              }}
            >
              {/* Intent 输入表单 */}
              <form
                ref={analysisFormRef}
                onSubmit={handleAnalysisIntentSubmit}
                style={{ display: 'flex', alignItems: 'center' }}
              >
                <input
                  type="text"
                  value={analysisIntent}
                  onChange={(e) => {
                    setAnalysisIntent(e.target.value);
                  }}
                  placeholder="Enter research intent"
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
                      color: '#ef4444',
                      fontSize: '0.875rem',
                    }}
                  >
                    {error}
                  </div>
                )}
              </form>

              {/* 下拉展开的维度选择面板 */}
              <DimensionSelectorPanel
                isOpen={showDimensionPanel}
                onClose={() => setShowDimensionPanel(false)}
                onConfirm={handleDimensionConfirm}
                intent={currentIntent}
              />
            </div>
          )}

          <div
            style={{
              display: (currentView === 'code_view' || currentView === 'paper_view') ? 'none' : 'flex',
              alignItems: 'flex-start', // Prevent stretching
              padding: '40px 20px 20px 20px', // Push down slightly for top button
              maxWidth: '1600px',
              margin: '0 auto',
              boxSizing: 'border-box',
            }}
          >
            {/* 左侧图 - Plot View 使用 CSS Grid 四象限，Tree View 使用 SVG */}

            {/* === 3D Plot View === */}
            <div
              ref={evaluationContainerRef}
              style={{
                width: '60%',
                marginRight: '20px',
                position: 'relative',
                display: currentView === 'evaluation' ? 'flex' : 'none',
                flexDirection: 'column',
                gap: '20px'
              }}
            >
              <div style={{ height: '800px', width: '100%' }}>
                <Evaluation3D
                  nodes={nodes}
                  selectedDimensionPairs={selectedDimensionPairs}
                  activeDimensionIndices={activeDimensionIndices}
                  onNodeDragEnd={handle3DNodeDragEnd}
                  selectedNode={selectedNode}
                  onNodeClick={handle3DNodeClick}
                  hoveredNode={hoveredNode}
                  onNodeHover={handle3DNodeHover}
                  dragHoverTarget={dragHoverTarget}
                  onDragHover={setDragHoverTarget}
                  pendingChange={pendingChange}
                  pendingMerge={pendingMerge}
                  dragVisualState={dragVisualState}
                  setDragVisualState={setDragVisualState}
                  mergeAnimationState={mergeAnimationState}
                  operationStatus={operationStatus}
                  isGenerating={isGenerating}
                  onDropExternal={handleNodeDropOnEvaluation}
                />
              </div>

              {/* Fragment Bar */}
              <div style={{
                width: '100%',
                minHeight: '120px',
                maxHeight: '300px',
                backgroundColor: '#fffbeb',
                border: '2px dashed #f59e0b',
                borderRadius: '8px',
                padding: '12px',
                display: 'flex',
                gap: '12px',
                overflowX: 'auto',
                overflowY: 'auto',
                alignItems: 'flex-start',
                boxSizing: 'border-box'
              }}>
                {fragmentNodes.length === 0 ? (
                  <div style={{ fontSize: '0.875rem', color: '#92400e', fontStyle: 'italic', margin: 'auto', textAlign: 'center' }}>
                    Select text in any idea card and click "Fragment" to create quick notes here
                  </div>
                ) : (
                  fragmentNodes.map(fragment => {
                    const isExpanded = expandedFragmentId === fragment.id;
                    const parentNode = nodes.find(n => n.id === fragment.parentId);
                    const parentTitle = parentNode ? parentNode.title : fragment.parentId;
                    const isMergeSource = dragVisualState && dragVisualState.type === 'merge' && dragVisualState.sourceNodeId === fragment.id;
                    const showPendingOutline = isMergeSource || fragment.isPendingMerge;

                    return (
                      <div key={fragment.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '60px', maxWidth: isExpanded ? '220px' : '60px', flexShrink: 0, transition: 'max-width 0.3s ease' }}>
                        <div
                          draggable={true}
                          onDragStart={(e) => {
                            e.dataTransfer.setData('nodeId', fragment.id);
                            e.dataTransfer.setData('isFragment', 'true');
                            const container = evaluationContainerRef.current;
                            if (container) {
                              const rect = e.currentTarget.getBoundingClientRect();
                              const containerRect = container.getBoundingClientRect();
                              fragmentDragOriginRef.current = {
                                x: rect.left - containerRect.left + rect.width / 2,
                                y: rect.top - containerRect.top + rect.height / 2
                              };
                            } else {
                              fragmentDragOriginRef.current = null;
                            }
                            setDraggingNodeId(fragment.id);
                            // Use a default style or transparent image if needed, but browser default is usually okay if opacity is handled
                            const dragImage = document.createElement('div');
                            dragImage.style.width = '50px'; dragImage.style.height = '50px'; dragImage.style.backgroundColor = '#fef08a';
                            dragImage.style.border = '2px solid #f59e0b'; dragImage.style.borderRadius = '50%';
                            dragImage.style.display = 'flex'; dragImage.style.alignItems = 'center'; dragImage.style.justifyContent = 'center';
                            dragImage.style.fontSize = '0.875rem'; dragImage.style.fontWeight = '700'; dragImage.style.color = '#92400e';
                            dragImage.style.position = 'fixed'; dragImage.style.top = '-1000px'; dragImage.style.left = '-1000px';
                            document.body.appendChild(dragImage);
                            e.dataTransfer.setDragImage(dragImage, 25, 25);
                            setTimeout(() => { document.body.removeChild(dragImage); }, 0);
                          }}
                          onDragEnd={() => {
                            fragmentDragOriginRef.current = null;
                            setDraggingNodeId(null);
                          }}
                          onClick={() => {
                            setExpandedFragmentId(isExpanded ? null : fragment.id);
                            if (!isExpanded && parentNode) setSelectedNode(parentNode);
                          }}
                          style={{
                            minWidth: '50px', width: '50px', height: '50px', backgroundColor: '#fef08a',
                            border: showPendingOutline ? '3px dashed #fbbf24' : '2px solid #f59e0b',
                            borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                            cursor: 'pointer', fontSize: '0.875rem', fontWeight: '700', color: '#92400e',
                            position: 'relative', flexShrink: 0, marginBottom: isExpanded ? '8px' : '0',
                            opacity: draggingNodeId === fragment.id ? 0.4 : (showPendingOutline ? 0.3 : (fragment.evaluationOpacity !== undefined ? fragment.evaluationOpacity : 1)),
                            transition: 'all 0.2s ease'
                          }}
                        >                          <button
                          onClick={(e) => { e.stopPropagation(); deleteFragmentNode(fragment.id); if (expandedFragmentId === fragment.id) setExpandedFragmentId(null); }}
                          style={{ position: 'absolute', top: '-6px', right: '-6px', width: '18px', height: '18px', borderRadius: '50%', backgroundColor: '#ef4444', color: 'white', border: 'none', cursor: 'pointer', fontSize: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0, fontWeight: 'bold' }}
                        >
                            ×
                          </button>
                        </div>
                        {isExpanded && (
                          <div style={{ width: '100%', padding: '10px', backgroundColor: 'white', border: '1px solid #fbbf24', borderRadius: '6px', fontSize: '0.75rem', lineHeight: 1.4, color: '#374151', maxHeight: '150px', overflowY: 'auto', wordBreak: 'break-word', cursor: 'default', animation: 'fadeIn 0.2s ease-in' }}>
                            <div style={{ fontSize: '0.7rem', color: '#78716c', marginBottom: '6px', fontWeight: '600' }}>Fragment of {parentTitle}</div>
                            <div style={{ fontSize: '0.7rem', color: '#6b7280', lineHeight: 1.5 }}>{fragment.content}</div>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* === D3.js SVG (Tree View) === */}
            <div
              ref={explorationContainerRef}
              style={{
                flexBasis: '60%',
                marginRight: '20px',
                display: currentView === 'exploration' ? 'block' : 'none'
              }}
            >
              <svg ref={svgRef} />
            </div>

            {/* 右侧 Dashboard */}
            <div style={{ flexBasis: '40%' }} ref={dashboardContainerRef}>
              <Dashboard
                nodeId={hoveredNode?.id || selectedNode?.id}
                nodes={nodes}
                fragmentNodes={fragmentNodes}
                isEvaluating={isEvaluating}
                showTree={currentView === 'exploration'}
                setModalAnchorEl={setModalAnchorEl}
                // Pass props down to ContextAndGenerateCard
                isAddingCustom={isAddingCustom}
                userInput={userInput}
                setUserInput={setUserInput}
                generateChildNodes={generateChildNodes}
                selectedNode={selectedNode}
                isGenerating={isGenerating}
                isAnalysisSubmitted={isAnalysisSubmitted}
                setIsAddingCustom={setIsAddingCustom}
                handleAddCustomIdea={handleAddCustomIdea}
                customIdea={customIdea}
                setCustomIdea={setCustomIdea}
                setIsEditingSystemPrompt={setIsEditingSystemPrompt}
                editIcon={editIcon}
                handleProceedWithSelectedIdea={handleProceedWithSelectedIdea}
                getNodeTrackingInfo={getNodeTrackingInfo}
                // Props for score modification
                onModifyScore={handleScoreModification}
                showModifyButton={currentView === 'exploration'}
                pendingChanges={pendingChange}
                currentView={currentView}
                selectedDimensionPairs={selectedDimensionPairs}
                activeDimensions={activeDimensions}
                onShowFragmentMenu={showFragmentMenu}
                activeDimensionIndices={activeDimensionIndices}
                onToggleDimensionIndex={toggleDimensionIndex}
                onCreateFragmentFromHighlight={(text, parentId) => createFragmentNode(text, parentId || selectedNode?.id)}
                onSwapDimension={swapDimensionDirection}
                onEditDimension={handleEditDimension}
              />

              {/* Re-eval All Button */}
              {currentView === 'evaluation' && (
                <div style={{
                  padding: '16px',
                  borderTop: '1px solid #e5e7eb',
                  display: 'flex',
                  justifyContent: 'flex-start'
                }}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      reEvaluateAll && reEvaluateAll();
                    }}
                    disabled={isEvaluating}
                    style={{
                      padding: '8px 12px',
                      backgroundColor: isEvaluating ? '#f3f4f6' : '#2563EB',
                      color: isEvaluating ? '#9ca3af' : '#ffffff',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: isEvaluating ? 'not-allowed' : 'pointer',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      transition: 'all 0.2s ease',
                      boxShadow: isEvaluating ? 'none' : '0 1px 2px rgba(0, 0, 0, 0.05)',
                    }}
                    title="Re-evaluate all ideas with current criteria"
                  >
                    {isEvaluating ? 'Re-evaluating...' : 'Re-evaluate All'}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* ============ Code View ============ */}
          {currentView === 'code_view' && (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '20px', gap: '16px', boxSizing: 'border-box' }}>

              {/* Header row */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexShrink: 0, gap: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: '#0F172A' }}>
                    Code Generation
                    {workflowIdea && (
                      <span style={{ fontSize: '0.8rem', fontWeight: 400, color: '#6B7280', marginLeft: '10px' }}>
                        {workflowIdea.title}
                      </span>
                    )}
                  </h2>
                  {codeResult?.experiment_dir && (
                    <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'monospace', marginTop: '2px' }}>
                      {codeResult.experiment_dir}
                    </div>
                  )}
                </div>

                {/* Right-side actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
                  {/* Spinning indicator while coding */}
                  {workflowStep === 'coding' && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#6B7280', fontSize: '0.82rem' }}>
                      <div style={{ width: '14px', height: '14px', border: '2px solid #3B82F6', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                      Generating…
                    </div>
                  )}

                  {/* Status badge */}
                  {codeResult && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '4px 10px', borderRadius: '999px', fontSize: '0.8rem', fontWeight: 600, backgroundColor: codeResult.success ? '#DCFCE7' : '#FEE2E2', color: codeResult.success ? '#15803D' : '#DC2626' }}>
                      {codeResult.success ? '✅ Success' : '❌ Failed'}
                    </div>
                  )}

                  {/* Rerun Coder button — available once we have a result or error */}
                  {(workflowStep === 'code_done' || workflowStep === 'code_error') && (
                    <button
                      onClick={handleRerunCoder}
                      style={{ padding: '6px 14px', backgroundColor: '#fff', color: '#374151', border: '1px solid #D1D5DB', borderRadius: '6px', fontSize: '0.82rem', fontWeight: 600, cursor: 'pointer' }}
                    >
                      Rerun Coder
                    </button>
                  )}

                  {/* Proceed to Writer button */}
                  {(workflowStep === 'code_done' || workflowStep === 'writing' || workflowStep === 'paper_error') && (
                    workflowStep === 'writing' ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#6B7280', fontSize: '0.82rem' }}>
                        <div style={{ width: '14px', height: '14px', border: '2px solid #3B82F6', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                        Writing…
                      </div>
                    ) : (
                      <button
                        onClick={() => setShowWriterPrompt(p => !p)}
                        style={{ padding: '6px 14px', backgroundColor: '#0F172A', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '0.82rem', fontWeight: 600, cursor: 'pointer' }}
                      >
                        {showWriterPrompt ? 'Cancel' : 'Proceed to Writer'}
                      </button>
                    )
                  )}
                </div>
              </div>

              {/* Expandable S2 key panel — shown below header when writer prompt is open */}
              {showWriterPrompt && workflowStep !== 'writing' && (
                <div style={{ padding: '14px 16px', backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', flexShrink: 0 }}>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: '#6B7280', marginBottom: '6px' }}>
                    Semantic Scholar API Key <span style={{ color: '#9CA3AF' }}>(optional — improves citation quality)</span>
                  </label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="password"
                      value={s2ApiKey}
                      onChange={(e) => setS2ApiKey(e.target.value)}
                      placeholder="sk-… or leave blank"
                      style={{ flex: 1, padding: '7px 12px', border: '1px solid #D1D5DB', borderRadius: '6px', fontSize: '0.85rem', fontFamily: 'inherit' }}
                      onKeyDown={(e) => { if (e.key === 'Enter') handleWritePaper(); }}
                    />
                    <button
                      onClick={handleWritePaper}
                      style={{ padding: '7px 16px', backgroundColor: '#0F172A', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap' }}
                    >
                      Write Paper
                    </button>
                  </div>
                </div>
              )}

              {/* Loading banner */}
              {workflowStep === 'coding' && (
                <div style={{ padding: '14px 16px', backgroundColor: '#EFF6FF', border: '1px solid #BFDBFE', borderRadius: '8px', color: '#1D4ED8', fontSize: '0.875rem', flexShrink: 0 }}>
                  Generating experiment code — this may take several minutes…
                </div>
              )}

              {/* Error banner */}
              {(workflowStep === 'code_error') && workflowError && (
                <div style={{ padding: '12px 16px', backgroundColor: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '8px', color: '#DC2626', fontSize: '0.875rem', flexShrink: 0 }}>
                  <strong>Error:</strong> {workflowError}
                </div>
              )}

              {/* Error details (coder failure but returned 200) */}
              {codeResult && !codeResult.success && codeResult.error_details && (
                <details style={{ flexShrink: 0, backgroundColor: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '8px', padding: '10px 14px' }}>
                  <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: '#DC2626', fontWeight: 600 }}>Error details</summary>
                  <pre style={{ marginTop: '8px', fontSize: '0.75rem', overflow: 'auto', backgroundColor: '#fff', padding: '8px', borderRadius: '4px', maxHeight: '160px', color: '#374151' }}>
                    {codeResult.error_details}
                  </pre>
                </details>
              )}

              {/* ── Code browser ── */}
              {codeFiles.length > 0 && (
                <div style={{ display: 'flex', flex: 1, minHeight: 0, border: '1px solid #E2E8F0', borderRadius: '8px', overflow: 'hidden', fontFamily: 'monospace' }}>

                  {/* File sidebar */}
                  <div style={{ width: '190px', flexShrink: 0, borderRight: '1px solid #E2E8F0', backgroundColor: '#F8FAFC', overflowY: 'auto' }}>
                    <div style={{ padding: '8px 12px', fontSize: '0.7rem', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.05em', borderBottom: '1px solid #E2E8F0' }}>
                      Files
                    </div>
                    {codeFiles.map((file) => {
                      const isSelected = selectedCodeFile?.path === file.path;
                      const ext = file.name.split('.').pop();
                      const icon = ext === 'py' ? '🐍' : ext === 'json' ? '{}' : ext === 'md' ? '📝' : '📄';
                      return (
                        <div
                          key={file.path}
                          onClick={() => loadCodeFile(file)}
                          style={{
                            padding: '7px 12px',
                            fontSize: '0.78rem',
                            cursor: 'pointer',
                            backgroundColor: isSelected ? '#E0E7FF' : 'transparent',
                            color: isSelected ? '#3730A3' : '#374151',
                            fontWeight: isSelected ? 600 : 400,
                            borderLeft: isSelected ? '3px solid #6366F1' : '3px solid transparent',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            wordBreak: 'break-all',
                          }}
                        >
                          <span style={{ flexShrink: 0 }}>{icon}</span>
                          {file.name}
                        </div>
                      );
                    })}
                  </div>

                  {/* Code pane */}
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                    {/* Tab bar */}
                    {selectedCodeFile && (
                      <div style={{ display: 'flex', alignItems: 'center', padding: '6px 14px', borderBottom: '1px solid #E2E8F0', backgroundColor: '#FFFFFF', fontSize: '0.8rem', color: '#374151', gap: '8px', flexShrink: 0 }}>
                        <span>{selectedCodeFile.name.endsWith('.py') ? '🐍' : selectedCodeFile.name.endsWith('.json') ? '{}' : '📄'}</span>
                        <span style={{ fontWeight: 600 }}>{selectedCodeFile.name}</span>
                        <span style={{ color: '#9CA3AF', fontSize: '0.72rem', marginLeft: 'auto' }}>
                          {selectedCodeFile.content.split('\n').length} lines
                        </span>
                      </div>
                    )}

                    {/* Code content */}
                    <div style={{ flex: 1, overflow: 'auto', backgroundColor: '#1E1E2E' }}>
                      {selectedCodeFile ? (
                        <pre style={{
                          margin: 0,
                          padding: '16px',
                          fontSize: '0.8rem',
                          lineHeight: 1.6,
                          color: '#CDD6F4',
                          whiteSpace: 'pre',
                          minHeight: '100%',
                          boxSizing: 'border-box',
                          counterReset: 'line',
                        }}>
                          {selectedCodeFile.content.split('\n').map((line, i) => (
                            <div key={i} style={{ display: 'flex', gap: '16px' }}>
                              <span style={{ color: '#585B70', userSelect: 'none', minWidth: '2.5em', textAlign: 'right', flexShrink: 0 }}>{i + 1}</span>
                              <span style={{ flex: 1 }}>{line || ' '}</span>
                            </div>
                          ))}
                        </pre>
                      ) : (
                        <div style={{ padding: '24px', color: '#585B70', fontSize: '0.85rem' }}>Select a file to view</div>
                      )}
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}

          {/* ============ Paper View ============ */}
          {currentView === 'paper_view' && (
            <div style={{ padding: '24px', maxWidth: '900px', margin: '0 auto' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px', color: '#0F172A' }}>Paper</h2>

              {workflowStep === 'writing' ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 40px', gap: '20px' }}>
                  <div style={{ width: '48px', height: '48px', border: '4px solid #E2E8F0', borderTopColor: '#0F172A', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1rem', fontWeight: 600, color: '#0F172A', marginBottom: '6px' }}>Writing paper…</div>
                    <div style={{ fontSize: '0.85rem', color: '#9CA3AF' }}>This may take a few minutes</div>
                  </div>
                </div>
              ) : !paperResult ? (
                <div style={{ padding: '40px', textAlign: 'center', color: '#9CA3AF' }}>
                  No paper generated yet. Use "Proceed with this Idea" from the idea card to start the workflow.
                </div>
              ) : (
                <>
                  <div style={{ marginBottom: '20px', padding: '16px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <span>✅</span>
                      <strong style={{ color: '#15803D' }}>Paper generated: {paperResult.paper_name}</strong>
                    </div>
                    <a
                      href={paperResult.pdf_path}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ fontSize: '0.85rem', color: '#2563EB', textDecoration: 'underline' }}
                    >
                      View PDF
                    </a>
                  </div>

                  {/* Review section */}
                  <div style={{ padding: '16px', backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: '8px', marginBottom: '16px' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '12px', color: '#374151' }}>Review Paper</h3>
                    {workflowStep === 'reviewing' ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#6B7280', fontSize: '0.875rem' }}>
                        <div style={{ width: '16px', height: '16px', border: '2px solid #3B82F6', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                        Reviewing paper...
                      </div>
                    ) : workflowStep === 'review_done' && reviewResult ? null : (
                      <button
                        onClick={handleReviewPaper}
                        style={{ padding: '8px 16px', backgroundColor: '#0F172A', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '0.875rem', fontWeight: 600, cursor: 'pointer' }}
                      >
                        Review Paper
                      </button>
                    )}
                  </div>

                  {/* Review results */}
                  {workflowStep === 'review_done' && reviewResult && (
                    <div style={{ padding: '16px', backgroundColor: '#fff', border: '1px solid #E2E8F0', borderRadius: '8px' }}>
                      <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '12px', color: '#374151' }}>Review Results</h3>
                      {typeof reviewResult === 'object' ? (
                        <div>
                          {reviewResult.Summary && <div style={{ marginBottom: '12px' }}><strong>Summary:</strong><p style={{ margin: '4px 0 0', color: '#4B5563', lineHeight: 1.6 }}>{reviewResult.Summary}</p></div>}
                          {reviewResult.Strengths && Array.isArray(reviewResult.Strengths) && (
                            <div style={{ marginBottom: '12px' }}>
                              <strong>Strengths:</strong>
                              <ul style={{ margin: '4px 0 0', paddingLeft: '20px', color: '#4B5563' }}>
                                {reviewResult.Strengths.map((s, i) => <li key={i}>{s}</li>)}
                              </ul>
                            </div>
                          )}
                          {reviewResult.Weaknesses && Array.isArray(reviewResult.Weaknesses) && (
                            <div style={{ marginBottom: '12px' }}>
                              <strong>Weaknesses:</strong>
                              <ul style={{ margin: '4px 0 0', paddingLeft: '20px', color: '#4B5563' }}>
                                {reviewResult.Weaknesses.map((w, i) => <li key={i}>{w}</li>)}
                              </ul>
                            </div>
                          )}
                          {reviewResult.Decision && (
                            <div style={{ marginTop: '12px', padding: '10px 14px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '6px' }}>
                              <strong>Decision:</strong> {reviewResult.Decision}
                              {reviewResult.Rating && <span style={{ marginLeft: '12px', fontWeight: 700 }}>Rating: {reviewResult.Rating}/10</span>}
                            </div>
                          )}
                          {(!reviewResult.Summary && !reviewResult.Decision) && (
                            <pre style={{ fontSize: '0.8rem', overflow: 'auto', whiteSpace: 'pre-wrap' }}>{JSON.stringify(reviewResult, null, 2)}</pre>
                          )}
                        </div>
                      ) : (
                        <pre style={{ fontSize: '0.8rem', overflow: 'auto', whiteSpace: 'pre-wrap' }}>{String(reviewResult)}</pre>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>)}

      {/* ========== 段落12：悬浮确认修改的按钮 (pendingChange) ========== */}
      {pendingChange && (
        currentView === 'exploration' ? (
          // Tree View Modal - Smaller, simpler, positioned lower
          <div
            className="modify-popup"
            data-panel-root="pending-change"
            style={{
              position: 'absolute',
              left: pendingChange.screenX - 60, // Center the smaller modal
              top: pendingChange.screenY + 20, // Position lower to avoid covering score
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              padding: '8px',
              zIndex: 1000,
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
              minWidth: '120px'
            }}
          >
            <div style={{ display: 'flex', gap: '4px' }}>
              {/* New Idea */}
              <button
                style={{
                  padding: '4px 8px',
                  backgroundColor: '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  fontWeight: '500'
                }}
                onClick={() => {
                  modifyIdeaBasedOnModifications(
                    pendingChange.originalNode,
                    pendingChange.ghostNode,
                    pendingChange.modifications,
                    pendingChange.behindNode
                  );
                  setPendingChange(null);
                }}
              >
                New Idea
              </button>
              {/* Modify Eval */}
              <button
                style={{
                  padding: '4px 8px',
                  backgroundColor: '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  fontWeight: '500'
                }}
                onClick={() => {
                  if (!pendingChange) return;


                  // 在应用修改前，先恢复节点到absolute定位的正确位置
                  const nodeElement = pendingChange.draggedElement || document.querySelector(`[data-node-id="${pendingChange.originalNode.id}"]`);
                  if (nodeElement && pendingChange.finalPosition) {
                    nodeElement.style.position = 'absolute';
                    nodeElement.style.left = `${pendingChange.finalPosition.x - 25}px`;
                    nodeElement.style.top = `${pendingChange.finalPosition.y - 25}px`;
                    nodeElement.style.zIndex = '100';
                  }

                  applyModifyEvaluation(pendingChange);
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
                  fontSize: '0.75rem',
                  fontWeight: '500'
                }}
                onClick={() => {
                  cancelPendingChange(pendingChange, { reason: 'manual' });
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          // Plot View Modal - Original design
          <div
            className="modify-popup"
            data-panel-root="pending-change"
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
              {/* Generate New Idea */}
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
                  modifyIdeaBasedOnModifications(
                    pendingChange.originalNode,
                    pendingChange.ghostNode,
                    pendingChange.modifications,
                    pendingChange.behindNode
                  );
                  setPendingChange(null);
                }}
              >
                New Idea
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
                  if (!pendingChange) return;


                  // 在应用修改前，先恢复节点到absolute定位的正确位置
                  const nodeElement = pendingChange.draggedElement || document.querySelector(`[data-node-id="${pendingChange.originalNode.id}"]`);
                  if (nodeElement && pendingChange.finalPosition) {
                    nodeElement.style.position = 'absolute';
                    nodeElement.style.left = `${pendingChange.finalPosition.x - 25}px`;
                    nodeElement.style.top = `${pendingChange.finalPosition.y - 25}px`;
                    nodeElement.style.zIndex = '100';
                  }

                  applyModifyEvaluation(pendingChange);
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
                  cancelPendingChange(pendingChange, { reason: 'manual' });
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )
      )}

      {pendingMerge && (
        <div
          className="merge-popup"
          data-panel-root="pending-merge"
          style={{
            position: 'absolute',
            left: pendingMerge.screenX || window.innerWidth / 2 - 150,
            top: pendingMerge.screenY || window.innerHeight / 2 - 80,
            backgroundColor: '#f3f4f6',
            border: '1px solid #d1d5db',
            borderRadius: '4px',
            padding: '6px 8px',
            zIndex: 1000,
          }}
        >
          {pendingMerge.action === 'evaluation_drag' ? (
            // Plot View 拖放选项
            <>
              <div style={{ marginBottom: '6px', fontSize: '0.8rem', color: '#374151' }}>
                What would you like to do with "{pendingMerge.sourceNode.title}"?
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
                    // 生成子节点
                    generateChildNodes(pendingMerge.sourceNode);
                    setPendingMerge(null);
                  }}
                >
                  New Idea
                </button>
                <button
                  style={{
                    padding: '4px 12px',
                    backgroundColor: '#f59e0b',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    // 选中节点供用户查看详情
                    setSelectedNode(pendingMerge.sourceNode);
                    setPendingMerge(null);
                    // TODO: 实现修改评分的UI界面
                  }}
                >
                  Modify Eval
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
                    cancelPendingMerge(pendingMerge);
                  }}
                >
                  Cancel
                </button>
              </div>
            </>
          ) : (
            // 原有的 Merge 对话框
            <>
              <div style={{ marginBottom: '6px', fontSize: '0.8rem', color: '#374151' }}>
                {pendingMerge.isFragmentMerge
                  ? `Merge fragment with "${pendingMerge.targetNode?.title}"?`
                  : 'Merge these two ideas?'}
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
                  onClick={async () => {
                    const nodeA = pendingMerge.sourceNode || pendingMerge.nodeA;
                    const nodeB = pendingMerge.targetNode || pendingMerge.nodeB;

                    // Apply merge animation before starting merge
                    // Check if source is a fragment
                    const isSourceFragment = fragmentNodes.some(f => f.id === nodeA.id);
                    const ghostPos =
                      dragVisualState?.ghostPosition ||
                      getNodeCenterRelativeToContainer(nodeA.id) ||
                      getNodeCenterRelativeToContainer(nodeB.id);
                    setMergeAnimationState({
                      sourceId: nodeA.id,
                      targetId: nodeB.id,
                      ghostPosition: ghostPos
                    });

                    if (isSourceFragment) {
                      // If source is fragment, update fragmentNodes for nodeA and nodes for nodeB
                      setFragmentNodes((prev) =>
                        prev.map((f) => {
                          if (f.id === nodeA.id) {
                            return { ...f, evaluationOpacity: 0, isPendingMerge: true };
                          }
                          return f;
                        })
                      );

                      setNodes((prev) =>
                        prev.map((n) => {
                          if (n.id === nodeB.id) {
                            return { ...n, isBeingMerged: true };
                          }
                          return n;
                        })
                      );
                    } else {
                      // Normal node merge
                      setNodes((prev) =>
                        prev.map((n) => {
                          if (n.id === nodeA.id) {
                            return { ...n, evaluationOpacity: 0 };
                          } else if (n.id === nodeB.id) {
                            return { ...n, isBeingMerged: true };
                          }
                          return n;
                        })
                      );
                    }

                    // 清理 pending 状态
                    setPendingMerge(null);
                    setDragVisualState(null);

                    // 直接触发合并 - 通过设置 mergeMode 的方式
                    setMergeMode({
                      active: false,
                      firstNode: nodeA,
                      secondNode: nodeB,
                      cursorPosition: { x: 0, y: 0 },
                      showDialog: false
                    });

                    // 使用 setTimeout 确保状态已更新
                    setTimeout(() => {
                      handleMergeConfirm(nodeA, nodeB);
                    }, 0);
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
                    cancelPendingMerge(pendingMerge);
                  }}
                >
                  Cancel
                </button>
              </div>
            </>
          )}
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

      {/* Fragment Menu */}
      {fragmentMenuState && (
        <div
          data-fragment-menu
          data-panel-root="fragment-menu"
          style={{
            position: 'fixed',
            left: fragmentMenuState.x,
            top: fragmentMenuState.y,
            backgroundColor: 'white',
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
            borderRadius: '8px',
            padding: '8px',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            minWidth: '150px'
          }}
        >
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px', padding: '0 4px' }}>
            Create Fragment Node
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleFragmentConfirm}
              style={{
                padding: '6px 12px',
                backgroundColor: '#4C84FF',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}
            >
              Fragment
            </button>
            <button
              onClick={hideFragmentMenu}
              style={{
                padding: '6px 12px',
                backgroundColor: '#fff',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Dimension Edit Dropdown */}
      <DimensionEditDropdown
        isOpen={editingDimensionIndex !== null}
        anchorPosition={dimensionDropdownAnchor}
        currentPair={editingDimensionIndex !== null ? selectedDimensionPairs[editingDimensionIndex] : null}
        pairIndex={editingDimensionIndex}
        onClose={handleCloseDimensionEdit}
        onConfirm={handleDimensionEditConfirm}
        isLoading={isSingleDimensionEvaluating}
      />

    </div>

  );
};

export default TreePlotVisualization;
