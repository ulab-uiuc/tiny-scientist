// ============== 段落1：导入依赖 ==============
import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

import TopNav from './TopNav';
import HypothesisCard from './HypothesisCard';
import HypothesisFactorsAndScoresCard from './HypothesisFactorsAndScoresCard';

// ============== 段落2：定义 TreePlotVisualization 组件 ==============
const TreePlotVisualization = () => {
  const [currentView, setCurrentView] = useState('overview'); // Start with overview
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
              onChange={(e) => setSelectedModel(e.target.value)}
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




  // ============== 段落3：评估假设（evaluateHypotheses） ==============
  const evaluateHypotheses = async (hypotheses) => {

    setIsEvaluating(true);
    setOperationStatus('Evaluating hypotheses...');
    setError(null);

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          ideas: hypotheses.map(h => ({
            id: h.id,
            title: h.title,
            content: h.content
          })),
          intent: analysisIntent
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to evaluate ideas');
      }

      const evaluatedHypotheses = await response.json();

      // rank -> score
      const rankToScore = (rank, total) => {
        const maxScore = 90;
        const minScore = 10;
        const scoreRange = maxScore - minScore;
        // 分数从高到低
        const score = maxScore - ((rank - 1) * scoreRange) / (total - 1);
        return Math.round(score);
      };

      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          const evalHypo = evaluatedHypotheses.find((h) => h.id === node.id);
          if (evalHypo) {
            const totalHypotheses = hypotheses.length;
            const noveltyScore = rankToScore(evalHypo.novelty_rank, totalHypotheses);
            const feasibilityScore = rankToScore(evalHypo.feasibility_rank, totalHypotheses);
            const impactScore = rankToScore(evalHypo.impact_rank, totalHypotheses);

            return {
              ...node,
              noveltyScore,
              feasibilityScore,
              impactScore,
              noveltyReason: evalHypo.novelty_rank_reason || '(No reason provided)',
              feasibilityReason: evalHypo.feasibility_rank_reason || '(No reason provided)',
              impactReason: evalHypo.impact_rank_reason || '(No reason provided)',
            };
          }
          return node;
        })
      );
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
          original_idea: {
            id: originalNode.id,
            title: originalNode.title,
            content: originalNode.content
          },
          modifications: modifications,
          behind_idea: behindNode ? {
            id: behindNode.id,
            title: behindNode.title,
            content: behindNode.content
          } : null
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
          idea_a: {
            id: nodeA.id,
            title: nodeA.title,
            content: nodeA.content
          },
          idea_b: {
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

  // ============== 段落7：节点点击/悬浮/离开事件处理 ==============
  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };
  const handleNodeHover = (event, d) => {
    setHoveredNode(d);
    d3.select(event.currentTarget).raise();
  };
  const handleNodeLeave = () => {
    setHoveredNode(null);
  };

  const zoomTransformRef = useRef(null);

  // ============== 段落8：D3 渲染逻辑 useEffect ==============
  useEffect(() => {
    if (!svgRef.current || currentView === 'overview') return;

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
      const initialTransform = d3.zoomIdentity.translate(300, 130);

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
        const nodeSpacing = 150; // Consistent spacing between all nodes
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

      nodeG
        .append('circle')
        .attr('r', 25)
        .style('fill', (d) => {
          if (d.isMergedResult) return '#B22222';
          if (d.isNewlyGenerated) return '#FFD700';
          return colorMap[d.type] || '#FF6B6B';
        })
        .style('opacity', (d) => {
          // Check if custom opacity is set
          if (d.opacity !== undefined) return d.opacity;
          // Otherwise use the default logic
          return d.isGhost ? 0.5 : 1;
        })
        .style('stroke', (d) =>
          selectedNode?.id === d.id ? '#000' : hoveredNode?.id === d.id ? '#555' : '#fff'
        )
        .style('stroke-width', (d) =>
          selectedNode?.id === d.id ? 4 : hoveredNode?.id === d.id ? 3 : 2
        )
        .style('cursor', 'pointer')
        .on('mouseenter', handleNodeHover)
        .on('mouseleave', handleNodeLeave)
        .on('click', (_, d) => handleNodeClick(d));

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
        .on('click', () => svg.transition().call(zoom.scaleBy, 1.3));

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
        .on('click', () => svg.transition().call(zoom.scaleBy, 0.7));

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
      if (xSel) xSel.onchange = (e) => setXAxisMetric(e.target.value);
      if (ySel) ySel.onchange = (e) => setYAxisMetric(e.target.value);

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
          .style('cursor', 'pointer')
          .on('mouseenter', (e, d) => setHoveredNode(d))
          .on('mouseleave', () => setHoveredNode(null))
          .on('click', (e, d) => handleNodeClick(d))
          .call(
            d3
              .drag()
              .on('start', function () {
                if (isGenerating || isEvaluating) return;
                d3.select(this).raise();
              })
              .on('drag', function (event, d) {
                if (isGenerating || isEvaluating) return;
                const [cx, cy] = d3.pointer(event, g.node());
                d3.select(this).attr('transform', `translate(${cx},${cy})`);
                d._tmpX = cx;
                d._tmpY = cy;
              })
              .on('end', function (event, d) {
                if (isGenerating || isEvaluating) return;
                const endX = d._tmpX ?? d3.pointer(event, g.node())[0];
                const endY = d._tmpY ?? d3.pointer(event, g.node())[1];
                delete d._tmpX;
                delete d._tmpY;

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
                handleNodeClick(d);
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
    selectedNode,
    hoveredNode,
    xAxisMetric,
    yAxisMetric,
    isGenerating,
    isEvaluating,
    operationStatus,
  ]);

  const Dashboard = ({ node, isEvaluating, showTree }) => {
    // 新增 state 用于统一控制"After/Before"
    const [showAfter, setShowAfter] = useState(true);

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

    // 仅在 node 被修改过时，才显示 toggle
    // const hasPrevious = !!node.previousState;

    return (
      // 整个 dashboard
      <div
        style={{
          border: '1px solid #E5E7EB',
          borderRadius: '8px',
          backgroundColor: '#FFFFFF',
          padding: '16px',
        }}
      >

        {/* Hypothesis Card */}
        <HypothesisCard node={node} showAfter={showAfter} setShowAfter={setShowAfter} />

        {/* 合并的三要素 + 分数卡片 */}
        {/* <HypothesisFactorsAndScoresCard
          node={node}
          isEvaluating={isEvaluating}
          showAfter={showAfter}
        /> */}

        {showTree ? (
          // Exploration View 下方替换成：Add context + Generate
          <ContextAndGenerateCard />
        ) : null}
      </div>
    );
  };

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

  const ContextAndGenerateCard = () => {
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
            <div style={{ marginBottom: '8px', fontSize: '0.875rem', color: '#6b7280' }}>
              Add context for new hypotheses (optional)
            </div>
            <textarea
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

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  generateChildNodes();
                }}
                disabled={!selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#4C84FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  cursor: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 'not-allowed' : 'pointer',
                  opacity: !selectedNode || isGenerating || isEvaluating || !isAnalysisSubmitted ? 0.6 : 1,
                }}
              >
                Generate New Hypotheses
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsAddingCustom(true);
                }}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#fff',
                  color: '#4C84FF',
                  border: '1px solid #4C84FF',
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  cursor: 'pointer',
                }}
              >
                Add Custom Hypothesis
              </button>
            </div>
          </div>
        ) : (
          <form
            onSubmit={(e) => {
              e.stopPropagation();
              handleAddCustomHypothesis(e);
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
                Title (2-3 words)
              </label>
              <input
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
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.875rem', color: '#374151' }}>
                Hypothesis Content
              </label>
              <textarea
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

  // ============== 段落11：整体布局，渲染主界面 JSX ==============
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', position: 'relative' }}>
      {/* 把 showTree 与 setShowTree 传给 TopNav，解决 setShowTree is not a function */}
      <TopNav currentView={currentView} setCurrentView={setCurrentView} />
      {currentView === 'overview' ? (
        <OverviewPage />
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
                    borderRadius: '4px',
                    cursor: isGenerating || isEvaluating ? 'not-allowed' : 'pointer',
                    opacity: isGenerating || isEvaluating ? 0.7 : 1,
                  }}
                >
                  {isEvaluating ? 'Evaluating...' : isGenerating ? 'Generating...' : 'Submit'}
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
            <div style={{ flexBasis: '60%', marginRight: '20px' }}>
              <svg ref={svgRef} />
            </div>

            {/* 右侧 Dashboard */}
            <div style={{ flexBasis: '40%' }}>
              <Dashboard
                node={nodes.find((n) => n.id === (hoveredNode?.id || selectedNode?.id))}
                isEvaluating={isEvaluating}
                showTree={currentView === 'exploration'}
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
    </div>

  );
};

// ============== 段落13：默认导出 ==============
export default TreePlotVisualization;
