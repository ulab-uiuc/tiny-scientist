import { buildNodeContent } from './contentParser';

class PlotStatusTracker {
  constructor() {
    this.plotStates = [];
    this.trackedNodes = new Map(); // Keep track of already recorded nodes
    this.isTracking = true;
    this.sessionId = this.generateSessionId();

    // Bind methods to preserve 'this' context
    this.trackNodesUpdate = this.trackNodesUpdate.bind(this);
    this.startTracking = this.startTracking.bind(this);
    this.stopTracking = this.stopTracking.bind(this);
    this.exportData = this.exportData.bind(this);
    this.downloadJSON = this.downloadJSON.bind(this);
  }

  generateSessionId() {
    return `plot_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  trackNodesUpdate(nodes, trigger = 'unknown') {
    if (!this.isTracking) return;

    // Filter out ghost nodes and only include real nodes
    const realNodes = nodes.filter(node =>
      node.type !== 'ghost' && !node.isGhost
    );

    // For evaluation_complete, track nodes with scores that haven't been recorded yet
    // For other triggers, only track new nodes
    let nodesToTrack;
    if (trigger === 'evaluation_complete') {
      // Track nodes that have scores AND haven't been recorded yet
      // 优先检查新系统的 scores 对象,向后兼容旧系统
      nodesToTrack = realNodes.filter(node =>
        ((node.scores && Object.keys(node.scores).length > 0) ||
         (node.noveltyScore || node.feasibilityScore || node.impactScore)) &&
        !this.trackedNodes.has(node.id)
      );
    } else {
      // Only track new nodes that haven't been recorded yet
      nodesToTrack = realNodes.filter(node => !this.trackedNodes.has(node.id));
    }

    if (nodesToTrack.length === 0) return; // No nodes to track

    const newNodeStates = nodesToTrack.map(node => {
      const nodeState = {
        id: node.id,
        title: node.title,
        type: node.type,
        nodeContent: buildNodeContent(node)
      };

      // Mark this node as tracked
      this.trackedNodes.set(node.id, nodeState);

      return nodeState;
    });

    // Add only the new nodes to plotStates
    this.plotStates.push(...newNodeStates);
  }

  startTracking() {
    this.isTracking = true;
  }

  stopTracking() {
    this.isTracking = false;
  }

  getPlotStates() {
    return this.plotStates;
  }

  exportData() {
    return {
      sessionId: this.sessionId,
      totalUpdates: this.plotStates.length,
      plotStates: this.plotStates
    };
  }

  downloadJSON() {
    const data = this.exportData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('data-uatrack-suppress', 'true');
    a.setAttribute('data-uatrack-suppress-click', 'true');
    a.href = url;
    a.download = `plot_status_${this.sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  clearPlotStates() {
    this.plotStates = [];
    this.trackedNodes.clear(); // Clear tracked nodes when clearing plot states
    this.sessionId = this.generateSessionId();
  }
}

// Create a singleton instance
const plotStatusTracker = new PlotStatusTracker();

// Auto-start tracking when the module loads
plotStatusTracker.startTracking();

export default plotStatusTracker;
