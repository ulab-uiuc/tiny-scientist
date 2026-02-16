import React, { useState, useEffect, useRef } from 'react';
import './DimensionEditDropdown.css';

const DimensionEditDropdown = ({
  isOpen,
  anchorPosition,     // { top, left } 定位
  currentPair,        // { dimensionA, dimensionB, descriptionA, descriptionB }
  pairIndex,          // 0/1/2 - 维度索引
  onClose,
  onConfirm,          // (newPair, pairIndex) => void
  isLoading = false,
}) => {
  const [editedA, setEditedA] = useState('');
  const [editedB, setEditedB] = useState('');
  const [editedDescA, setEditedDescA] = useState('');
  const [editedDescB, setEditedDescB] = useState('');
  const dropdownRef = useRef(null);

  // 当currentPair变化时，更新输入框
  useEffect(() => {
    if (currentPair) {
      setEditedA(currentPair.dimensionA || '');
      setEditedB(currentPair.dimensionB || '');
      setEditedDescA(currentPair.descriptionA || '');
      setEditedDescB(currentPair.descriptionB || '');
    }
  }, [currentPair]);

  // 点击外部关闭
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  // ESC键关闭
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const axisLabel = pairIndex === 0 ? 'X-axis' : pairIndex === 1 ? 'Y-axis' : 'Z-axis';
  const hasChanges = editedA !== (currentPair?.dimensionA || '') ||
                     editedB !== (currentPair?.dimensionB || '') ||
                     editedDescA !== (currentPair?.descriptionA || '') ||
                     editedDescB !== (currentPair?.descriptionB || '');

  const isValid = editedA.trim() && editedB.trim();

  const handleConfirm = () => {
    if (!isValid || isLoading) return;
    onConfirm({
      ...currentPair,
      dimensionA: editedA.trim(),
      dimensionB: editedB.trim(),
      descriptionA: editedDescA.trim(),
      descriptionB: editedDescB.trim(),
    }, pairIndex);
  };

  return (
    <div
      ref={dropdownRef}
      className="dimension-edit-dropdown"
      data-panel-root="dimension-edit"
      style={{
        top: anchorPosition?.top || 0,
        left: anchorPosition?.left || 0,
      }}
    >
      {/* Header */}
      <div className="edit-dropdown-header">
        <span className="edit-dropdown-title">
          Dimension {pairIndex + 1} ({axisLabel})
        </span>
        <button className="edit-close-btn" onClick={onClose}>×</button>
      </div>

      {/* Input fields */}
      <div className="edit-dropdown-body">
        <div className="edit-field-group">
            <label>Dimension A</label>
            <input
              type="text"
              className="edit-dimension-input dimension-a"
              value={editedA}
              onChange={(e) => setEditedA(e.target.value)}
              placeholder="Dimension A Name"
              autoFocus
            />
            <textarea
              className="edit-dimension-desc"
              value={editedDescA}
              onChange={(e) => setEditedDescA(e.target.value)}
              placeholder="Description for Dimension A"
              rows={2}
            />
        </div>

        <div className="edit-vs-separator-row">
            <span className="edit-vs-separator">vs</span>
        </div>

        <div className="edit-field-group">
            <label>Dimension B</label>
            <input
              type="text"
              className="edit-dimension-input dimension-b"
              value={editedB}
              onChange={(e) => setEditedB(e.target.value)}
              placeholder="Dimension B Name"
            />
            <textarea
              className="edit-dimension-desc"
              value={editedDescB}
              onChange={(e) => setEditedDescB(e.target.value)}
              placeholder="Description for Dimension B"
              rows={2}
            />
        </div>
      </div>

      {/* Footer buttons */}
      <div className="edit-dropdown-footer">
        <button className="edit-btn-cancel" onClick={onClose}>
          Cancel
        </button>
        <button
          className="edit-btn-confirm"
          onClick={handleConfirm}
          disabled={!isValid || isLoading}
        >
          {isLoading ? 'Scoring...' : hasChanges ? 'Confirm & Re-score' : 'Confirm'}
        </button>
      </div>
    </div>
  );
};

export default DimensionEditDropdown;
