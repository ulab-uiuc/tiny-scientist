class UserActionTracker {
  constructor() {
    this.actions = [];
    this.isTracking = true;
    this.sessionId = this.generateSessionId();
    this.schemaVersion = 2;
    this.sequence = 0;

    this._elementMetadata = new WeakMap();
    this._recentByKey = new Map();
    this._inputSessionByElement = new WeakMap();
    this._pointerSessionById = new Map();
    this._html5DragSessionByElement = new WeakMap();
    this._installedDomListeners = null;

    // Bind methods to preserve 'this' context
    this.trackAction = this.trackAction.bind(this);
    this.startTracking = this.startTracking.bind(this);
    this.stopTracking = this.stopTracking.bind(this);
    this.registerElement = this.registerElement.bind(this);
    this.unregisterElement = this.unregisterElement.bind(this);
    this.exportData = this.exportData.bind(this);
    this.downloadJSON = this.downloadJSON.bind(this);
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  _getNextId() {
    this.sequence += 1;
    return `${this.sessionId}:${this.sequence}`;
  }

  _getPageContext() {
    return {
      url: window.location.href,
      pathname: window.location.pathname,
      hash: window.location.hash
    };
  }

  _safeString(value, maxLen = 120) {
    if (value === null || value === undefined) return null;
    const str = String(value);
    if (str.length <= maxLen) return str;
    return `${str.slice(0, maxLen)}…`;
  }

  _hashStringFNV1a(input) {
    if (!input) return '0';
    let hash = 0x811c9dc5;
    for (let i = 0; i < input.length; i += 1) {
      hash ^= input.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16);
  }

  _buildElementLocator(element) {
    if (!element || element.nodeType !== 1) return null;

    const explicit =
      element.getAttribute?.('data-tracking-name') ||
      element.getAttribute?.('data-testid') ||
      element.getAttribute?.('aria-label') ||
      element.id ||
      null;
    if (explicit) return this._safeString(explicit, 160);

    const parts = [];
    let current = element;
    for (let depth = 0; current && depth < 4; depth += 1) {
      if (!current.tagName) break;
      const tag = current.tagName.toLowerCase();
      const parent = current.parentElement;
      if (!parent) {
        parts.push(tag);
        break;
      }
      const currentTagName = current.tagName;
      const siblings = Array.from(parent.children).filter((n) => n.tagName === currentTagName);
      const index = siblings.indexOf(current) + 1;
      parts.push(`${tag}:nth-of-type(${index})`);
      current = parent;
    }
    return parts.length > 0 ? parts.reverse().join('>') : null;
  }

  _extractElementInfo(element) {
    if (!element || element.nodeType !== 1) return null;
    const tagName = element.tagName?.toLowerCase() || null;
    const role = element.getAttribute?.('role') || null;
    const ariaLabel = element.getAttribute?.('aria-label') || null;
    const testId = element.getAttribute?.('data-testid') || null;
    const trackingName = element.getAttribute?.('data-tracking-name') || null;
    const nodeId = element.getAttribute?.('data-node-id') || null;

    let text = null;
    if (tagName === 'button' || role === 'button' || tagName === 'a') {
      text = this._safeString(element.textContent?.trim() || null, 80);
    }

    return {
      tagName,
      id: element.id || null,
      role,
      ariaLabel: this._safeString(ariaLabel, 120),
      testId: this._safeString(testId, 120),
      trackingName: this._safeString(trackingName, 120),
      nodeId: this._safeString(nodeId, 120),
      text,
      locator: this._buildElementLocator(element)
    };
  }

  _findElementWithMetadata(startElement) {
    let el = startElement;
    for (let i = 0; el && i < 6; i += 1) {
      if (this._elementMetadata.has(el)) return el;
      el = el.parentElement;
    }
    return null;
  }

  _pickTrackingElement(rawTarget) {
    const element = rawTarget?.nodeType === 3 ? rawTarget.parentElement : rawTarget;
    if (!element || element.nodeType !== 1) return null;

    const selector =
      '[data-tracking-name],[data-testid],[data-node-id],button,a,input,textarea,select,form,[role],[contenteditable="true"],svg,canvas';
    return element.closest?.(selector) || element;
  }

  _getElementMetadata(element) {
    if (!element) return null;
    return this._elementMetadata.get(element) || null;
  }

  _mergeObjects(base, incoming) {
    if (!incoming) return base;
    if (!base) return incoming;
    const out = { ...base };
    Object.entries(incoming).forEach(([key, value]) => {
      if (
        value &&
        typeof value === 'object' &&
        !Array.isArray(value) &&
        out[key] &&
        typeof out[key] === 'object' &&
        !Array.isArray(out[key])
      ) {
        out[key] = { ...out[key], ...value };
      } else {
        out[key] = value;
      }
    });
    return out;
  }

  _getPhysicalKind(actionType) {
    if (!actionType) return 'semantic';
    if (actionType === 'click' || actionType === 'button_click' || actionType === 'component_button_click') return 'click';
    if (actionType === 'double_click' || actionType === 'dblclick') return 'dblclick';
    if (actionType === 'hover_start' || actionType === 'hover_end' || actionType === 'button_hover_start' || actionType === 'button_hover_end') return 'hover';
    if (actionType === 'drag' || actionType === 'drag_start' || actionType === 'drag_end' || actionType === 'drop_complete') return 'drag';
    if (actionType === 'scroll') return 'scroll';
    if (actionType === 'focus' || actionType === 'blur') return 'focus';
    if (actionType === 'input_change' || actionType === 'form_input' || actionType === 'form_change' || actionType === 'input_start' || actionType === 'input_commit') return 'input';
    if (actionType === 'key_down' || actionType === 'key_up') return 'keyboard';
    return 'semantic';
  }

  _buildDedupKey(actionType, target, details, options) {
    if (options?.dedupKey) return options.dedupKey;

    const physicalKind = options?.physicalKind || this._getPhysicalKind(actionType);
    const locator = details?.element?.locator || null;
    const targetKey = locator || target || 'unknown';

    if (physicalKind === 'click' || physicalKind === 'dblclick') {
      const modifiers = details?.pointer
        ? `${details.pointer.button ?? 'na'}:${details.pointer.ctrlKey ? 1 : 0}${details.pointer.shiftKey ? 1 : 0}${details.pointer.altKey ? 1 : 0}${details.pointer.metaKey ? 1 : 0}`
        : 'na';
      return `${physicalKind}:${targetKey}:${modifiers}`;
    }

    if (physicalKind === 'hover') {
      return `hover:${actionType}:${targetKey}`;
    }

    if (physicalKind === 'input') {
      const value = details?.input?.value ?? null;
      const valueSig = value === null ? 'na' : `${details?.input?.length ?? 0}:${this._hashStringFNV1a(String(value))}`;
      return `input:${actionType}:${targetKey}:${valueSig}`;
    }

    if (physicalKind === 'drag') {
      return `drag:${actionType}:${targetKey}`;
    }

    if (physicalKind === 'scroll') {
      return `scroll:${targetKey}`;
    }

    if (physicalKind === 'keyboard') {
      return `key:${details?.keyboard?.key || 'unknown'}`;
    }

    return `${actionType}:${targetKey}`;
  }

  _shouldDedup(dedupKey, windowMs) {
    const now = Date.now();
    const record = this._recentByKey.get(dedupKey);
    if (!record) return false;
    return now - record.lastSeenMs <= windowMs;
  }

  _touchDedup(dedupKey, actionIndex) {
    this._recentByKey.set(dedupKey, { lastSeenMs: Date.now(), actionIndex });
    if (this._recentByKey.size <= 2000) return;

    const cutoff = Date.now() - 10_000;
    for (const [key, value] of this._recentByKey.entries()) {
      if (value.lastSeenMs < cutoff) this._recentByKey.delete(key);
    }
  }

  _recordAction(action, options) {
    const physicalKind = options?.physicalKind || this._getPhysicalKind(action.actionType);
    const dedupWindowMs =
      options?.dedupWindowMs ??
      (physicalKind === 'hover'
        ? 0
        : physicalKind === 'scroll'
          ? 200
          : physicalKind === 'keyboard'
            ? 0
            : physicalKind === 'semantic'
              ? 250
              : 25);
    const dedupKey = this._buildDedupKey(action.actionType, action.target, action.details, { ...options, physicalKind });

    if (this._shouldDedup(dedupKey, dedupWindowMs)) {
      const existingIndex = this._recentByKey.get(dedupKey)?.actionIndex;
      const existing = Number.isInteger(existingIndex) ? this.actions[existingIndex] : null;
      if (existing) {
        existing.details = this._mergeObjects(existing.details, action.details);
        this._touchDedup(dedupKey, existingIndex);
      }
      return;
    }

    const actionIndex = this.actions.push(action) - 1;
    this._touchDedup(dedupKey, actionIndex);
  }

  trackAction(actionType, target, details = {}, options = {}) {
    if (!this.isTracking) return;

    const physicalKind = options.physicalKind || this._getPhysicalKind(actionType);

    const action = {
      schemaVersion: this.schemaVersion,
      id: this._getNextId(),
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      actionType,
      target,
      details: {
        kind: physicalKind,
        page: this._getPageContext(),
        ...details
      }
    };

    this._recordAction(action, { ...options, physicalKind });
  }

  startTracking() {
    this.isTracking = true;
    this._ensureDomTrackingInstalled();
    // Don't track the start action - it's meaningless
  }

  stopTracking() {
    this.isTracking = false;
    this._uninstallDomTracking();
    // Don't track the stop action - it's meaningless for research
  }

  registerElement(element, targetName, additionalDetails = {}) {
    if (!element || element.nodeType !== 1) return;
    this._elementMetadata.set(element, { targetName, additionalDetails });
  }

  unregisterElement(element) {
    if (!element || element.nodeType !== 1) return;
    this._elementMetadata.delete(element);
  }

  _ensureDomTrackingInstalled() {
    if (this._installedDomListeners) return;
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    const onClick = (e) => this._handleClick(e);
    const onDblClick = (e) => this._handleDblClick(e);
    const onPointerOver = (e) => this._handlePointerOver(e);
    const onPointerOut = (e) => this._handlePointerOut(e);
    const onPointerDown = (e) => this._handlePointerDown(e);
    const onPointerMove = (e) => this._handlePointerMove(e);
    const onPointerUp = (e) => this._handlePointerUp(e);
    const onPointerCancel = (e) => this._handlePointerCancel(e);
    const onDragStart = (e) => this._handleHtml5DragStart(e);
    const onDragEnd = (e) => this._handleHtml5DragEnd(e);
    const onFocusOut = (e) => this._handleFocusOut(e);
    const onInput = (e) => this._handleInput(e);
    const onChange = (e) => this._handleChange(e);
    const onPopState = () => {
      this.trackAction('browser_navigation', 'popstate', {
        url: window.location.href,
        hash: window.location.hash
      });
    };

    document.addEventListener('click', onClick, true);
    document.addEventListener('dblclick', onDblClick, true);
    document.addEventListener('pointerover', onPointerOver, true);
    document.addEventListener('pointerout', onPointerOut, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('pointermove', onPointerMove, true);
    document.addEventListener('pointerup', onPointerUp, true);
    document.addEventListener('pointercancel', onPointerCancel, true);
    document.addEventListener('dragstart', onDragStart, true);
    document.addEventListener('dragend', onDragEnd, true);
    document.addEventListener('focusout', onFocusOut, true);
    document.addEventListener('input', onInput, true);
    document.addEventListener('change', onChange, true);
    window.addEventListener('popstate', onPopState);

    this._installedDomListeners = {
      onClick,
      onDblClick,
      onPointerOver,
      onPointerOut,
      onPointerDown,
      onPointerMove,
      onPointerUp,
      onPointerCancel,
      onDragStart,
      onDragEnd,
      onFocusOut,
      onInput,
      onChange,
      onPopState
    };
  }

  _uninstallDomTracking() {
    if (!this._installedDomListeners) return;
    const l = this._installedDomListeners;
    document.removeEventListener('click', l.onClick, true);
    document.removeEventListener('dblclick', l.onDblClick, true);
    document.removeEventListener('pointerover', l.onPointerOver, true);
    document.removeEventListener('pointerout', l.onPointerOut, true);
    document.removeEventListener('pointerdown', l.onPointerDown, true);
    document.removeEventListener('pointermove', l.onPointerMove, true);
    document.removeEventListener('pointerup', l.onPointerUp, true);
    document.removeEventListener('pointercancel', l.onPointerCancel, true);
    document.removeEventListener('dragstart', l.onDragStart, true);
    document.removeEventListener('dragend', l.onDragEnd, true);
    document.removeEventListener('focusout', l.onFocusOut, true);
    document.removeEventListener('input', l.onInput, true);
    document.removeEventListener('change', l.onChange, true);
    window.removeEventListener('popstate', l.onPopState);
    this._installedDomListeners = null;
    this._pointerSessionById.clear();
    this._inputSessionByElement = new WeakMap();
    this._html5DragSessionByElement = new WeakMap();
  }

  _isSuppressed(element, kind = null) {
    if (!element) return false;
    if (element.closest?.('[data-uatrack-suppress="true"]')) return true;
    if (kind && element.closest?.(`[data-uatrack-suppress-${kind}="true"]`)) return true;
    return false;
  }

  _shouldTrackHover(element) {
    if (!element) return false;
    return Boolean(element.getAttribute?.('data-node-id'));
  }

  _shouldTrackClick(element) {
    if (!element) return false;
    const clickFlag = element.getAttribute?.('data-tracking-click');
    if (clickFlag === 'true') return true;
    if (element.getAttribute?.('data-node-id')) return true;
    const buttonLike = element.closest?.('button,[role="button"]');
    return Boolean(buttonLike);
  }

  _decorateWithMetadata(details, element) {
    const metadataElement = this._findElementWithMetadata(element);
    const metadata = metadataElement ? this._getElementMetadata(metadataElement) : null;
    if (!metadata) return details;

    return this._mergeObjects(details, {
      context: {
        trackingTarget: metadata.targetName || null,
        ...metadata.additionalDetails
      }
    });
  }

  _buildTargetName(element, fallback = 'unknown') {
    if (!element) return fallback;
    const metadata = this._getElementMetadata(element);
    if (metadata?.targetName) return metadata.targetName;

    const trackingName = element.getAttribute?.('data-tracking-name');
    if (trackingName) return trackingName;
    const testId = element.getAttribute?.('data-testid');
    if (testId) return testId;
    const ariaLabel = element.getAttribute?.('aria-label');
    if (ariaLabel) return ariaLabel;
    if (element.id) return element.id;

    const tag = element.tagName?.toLowerCase() || fallback;
    const text = element.textContent?.trim();
    if (text) return `${tag}:${this._safeString(text, 40)}`;
    return tag;
  }

  _handleClick(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'click')) return;
    if (!this._shouldTrackClick(element)) return;

    const targetName = this._buildTargetName(element, 'click_target');
    const elementInfo = this._extractElementInfo(element);
    const details = this._decorateWithMetadata(
      {
        element: elementInfo,
        pointer: {
          x: e.clientX,
          y: e.clientY,
          button: e.button,
          pointerType: e.pointerType || null,
          ctrlKey: e.ctrlKey,
          shiftKey: e.shiftKey,
          altKey: e.altKey,
          metaKey: e.metaKey
        }
      },
      element
    );

    this.trackAction('click', targetName, details, { physicalKind: 'click' });
  }

  _handleDblClick(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'click')) return;

    const targetName = this._buildTargetName(element, 'dblclick_target');
    const elementInfo = this._extractElementInfo(element);
    const details = this._decorateWithMetadata(
      {
        element: elementInfo,
        pointer: {
          x: e.clientX,
          y: e.clientY,
          button: e.button,
          pointerType: e.pointerType || null,
          ctrlKey: e.ctrlKey,
          shiftKey: e.shiftKey,
          altKey: e.altKey,
          metaKey: e.metaKey
        }
      },
      element
    );

    this.trackAction('dblclick', targetName, details, { physicalKind: 'dblclick' });
  }

  _handlePointerOver(e) {
    if (!this.isTracking) return;
    const toEl = this._pickTrackingElement(e.target);
    const fromEl = this._pickTrackingElement(e.relatedTarget);
    if (!toEl || this._isSuppressed(toEl, 'hover')) return;
    if (!this._shouldTrackHover(toEl)) return;
    if (toEl === fromEl) return;

    const targetName = this._buildTargetName(toEl, 'hover_target');
    const elementInfo = this._extractElementInfo(toEl);
    const details = this._decorateWithMetadata(
      {
        element: elementInfo,
        pointer: { x: e.clientX, y: e.clientY, pointerType: e.pointerType || null }
      },
      toEl
    );
    this.trackAction('hover_start', targetName, details, { physicalKind: 'hover', dedupWindowMs: 0 });
  }

  _handlePointerOut(e) {
    if (!this.isTracking) return;
    const fromEl = this._pickTrackingElement(e.target);
    const toEl = this._pickTrackingElement(e.relatedTarget);
    if (!fromEl || this._isSuppressed(fromEl, 'hover')) return;
    if (!this._shouldTrackHover(fromEl)) return;
    if (fromEl === toEl) return;

    const targetName = this._buildTargetName(fromEl, 'hover_target');
    const elementInfo = this._extractElementInfo(fromEl);
    const details = this._decorateWithMetadata(
      {
        element: elementInfo,
        pointer: { x: e.clientX, y: e.clientY, pointerType: e.pointerType || null }
      },
      fromEl
    );
    this.trackAction('hover_end', targetName, details, { physicalKind: 'hover', dedupWindowMs: 0 });
  }

  _handlePointerDown(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'drag')) return;
    if (!element.getAttribute?.('data-node-id')) return;

    this._pointerSessionById.set(e.pointerId, {
      pointerId: e.pointerId,
      pointerType: e.pointerType || null,
      startX: e.clientX,
      startY: e.clientY,
      startTimeMs: Date.now(),
      element,
      started: false
    });
  }

  _handlePointerMove(e) {
    if (!this.isTracking) return;
    const session = this._pointerSessionById.get(e.pointerId);
    if (!session) return;
    if (session.started) return;

    const dx = e.clientX - session.startX;
    const dy = e.clientY - session.startY;
    const distanceSq = dx * dx + dy * dy;
    if (distanceSq < 64) return; // 8px threshold

    session.started = true;
  }

  _handlePointerUp(e) {
    if (!this.isTracking) return;
    const session = this._pointerSessionById.get(e.pointerId);
    if (!session) return;
    this._pointerSessionById.delete(e.pointerId);
    if (!session.started) return;

    const elementInfo = this._extractElementInfo(session.element);
    const targetName = this._buildTargetName(session.element, 'drag_target');
    const dx = e.clientX - session.startX;
    const dy = e.clientY - session.startY;
    const details = this._decorateWithMetadata(
      {
        element: elementInfo,
        pointer: {
          startX: session.startX,
          startY: session.startY,
          endX: e.clientX,
          endY: e.clientY,
          pointerType: session.pointerType,
          durationMs: Date.now() - session.startTimeMs,
          deltaX: dx,
          deltaY: dy
        }
      },
      session.element
    );
    this.trackAction('drag', targetName, details, { physicalKind: 'drag' });
  }

  _handlePointerCancel(e) {
    this._pointerSessionById.delete(e.pointerId);
  }

  _handleHtml5DragStart(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'drag')) return;
    if (!element.getAttribute?.('data-node-id')) return;
    this._html5DragSessionByElement.set(element, {
      startX: e.clientX,
      startY: e.clientY,
      startTimeMs: Date.now()
    });
  }

  _handleHtml5DragEnd(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'drag')) return;

    const session = this._html5DragSessionByElement.get(element);
    this._html5DragSessionByElement.delete(element);
    if (!session) return;

    const dx = e.clientX - session.startX;
    const dy = e.clientY - session.startY;
    const targetName = this._buildTargetName(element, 'drag_target');
    const details = this._decorateWithMetadata(
      {
        element: this._extractElementInfo(element),
        pointer: {
          startX: session.startX,
          startY: session.startY,
          endX: e.clientX,
          endY: e.clientY,
          deltaX: dx,
          deltaY: dy,
          durationMs: Date.now() - session.startTimeMs,
          pointerType: 'html5_drag'
        }
      },
      element
    );

    this.trackAction('drag', targetName, details, { physicalKind: 'drag' });
  }

  _handleFocusOut(e) {
    if (!this.isTracking) return;
    const raw = e.target?.nodeType === 3 ? e.target.parentElement : e.target;
    const element = raw && raw.nodeType === 1 ? raw : null;
    if (!element || this._isSuppressed(element, 'input')) return;

    // If this is an input-like element and we saw input events, treat focusout as "commit"
    const inputSession = this._inputSessionByElement.get(element);
    if (inputSession?.started) {
      if (element.tagName === 'INPUT' && element.type === 'password') {
        this._inputSessionByElement.delete(element);
        return;
      }
      const value = this._readInputValue(element);
      const targetName = this._buildTargetName(element, 'input_target');
      const details = this._decorateWithMetadata(
        {
          element: this._extractElementInfo(element),
          input: {
            value,
            length: value ? value.length : 0,
            inputType: element.type || null,
            name: element.name || null
          }
        },
        element
      );
      this.trackAction('input_commit', targetName, details, { physicalKind: 'input' });
      this._inputSessionByElement.delete(element);
    }
  }

  _readInputValue(element) {
    if (!element) return null;
    if (element.tagName === 'SELECT') return element.value ?? '';
    if (element.tagName === 'INPUT') {
      if (element.type === 'checkbox' || element.type === 'radio') return Boolean(element.checked);
      return element.value ?? '';
    }
    if (element.tagName === 'TEXTAREA') return element.value ?? '';
    const contentEditable = element.getAttribute?.('contenteditable');
    if (contentEditable === 'true') return element.innerText ?? '';
    return null;
  }

  _handleInput(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'input')) return;

    const isInputLike =
      element.tagName === 'INPUT' ||
      element.tagName === 'TEXTAREA' ||
      element.getAttribute?.('contenteditable') === 'true';
    if (!isInputLike) return;
    if (element.tagName === 'INPUT' && element.type === 'password') return;

    const session = this._inputSessionByElement.get(element) || { started: false };
    if (!session.started) {
      session.started = true;
      session.startTimeMs = Date.now();
      this._inputSessionByElement.set(element, session);
    }
  }

  _handleChange(e) {
    if (!this.isTracking) return;
    const element = this._pickTrackingElement(e.target);
    if (!element || this._isSuppressed(element, 'input')) return;

    const isChangeLike =
      element.tagName === 'SELECT' ||
      (element.tagName === 'INPUT' && (element.type === 'checkbox' || element.type === 'radio'));
    if (!isChangeLike) return;

    const value = this._readInputValue(element);
    if (element.tagName === 'INPUT' && element.type === 'password') return;
    const targetName = this._buildTargetName(element, 'change_target');
    const details = this._decorateWithMetadata(
      {
        element: this._extractElementInfo(element),
        input: {
          value,
          length: typeof value === 'string' ? value.length : null,
          inputType: element.type || null,
          name: element.name || null
        }
      },
      element
    );
    this.trackAction('input_commit', targetName, details, { physicalKind: 'input' });
  }

  _handleSubmit(e) {
    // Form submit tracking disabled
    return;
  }

  _handleKeyDown(e) {
    // Keyboard tracking disabled per request
    return;
  }

  _handleKeyUp(e) {
    // Keyboard tracking disabled per request
    return;
  }


  getActions() {
    return this.actions;
  }

  exportData() {
    return {
      schemaVersion: this.schemaVersion,
      sessionId: this.sessionId,
      startTime: this.actions.length > 0 ? this.actions[0].timestamp : null,
      endTime: this.actions.length > 0 ? this.actions[this.actions.length - 1].timestamp : null,
      totalActions: this.actions.length,
      actions: this.actions
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
    a.download = `user_actions_${this.sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  clearActions() {
    this.actions = [];
    this.sessionId = this.generateSessionId();
    this.sequence = 0;
    this._recentByKey.clear();
    this._pointerSessionById.clear();
    this._inputSessionByElement = new WeakMap();
  }
}

// Create a singleton instance
const userActionTracker = new UserActionTracker();

// Auto-start tracking when the module loads
userActionTracker.startTracking();

export default userActionTracker;
