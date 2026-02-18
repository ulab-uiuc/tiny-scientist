import React, { useState, useEffect } from 'react';
import './DimensionSelectorPanel.css';

/**
 * 下拉展开面板式维度选择器
 * 在 Intent 输入框下方展开,覆盖在界面上
 * 显示 3 个 AI 建议的维度对,用户可选择或自定义 3 对维度
 */
const DimensionSelectorPanel = ({ isOpen, onClose, onConfirm, intent }) => {
    const [suggestedPairs, setSuggestedPairs] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // 用户自定义的3对维度
    const [customDimensions, setCustomDimensions] = useState({
        pair1: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' },
        pair2: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' },
        pair3: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' }
    });

    // 获取 AI 建议的维度对
    useEffect(() => {
        if (isOpen && intent) {
            fetchSuggestedDimensions();
        }
    }, [isOpen, intent]);

    const fetchSuggestedDimensions = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/suggest-dimensions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include', // 重要: 包含 session cookie
                body: JSON.stringify({ intent })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to fetch dimension suggestions');
            }

            const data = await response.json();
            setSuggestedPairs(data.dimension_pairs || []);
        } catch (err) {
            console.error('Error fetching dimensions:', err);
            setError(err.message || 'Failed to load dimension suggestions. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const findPairSlotForSuggestion = (pair, dimensions) => {
        return Object.entries(dimensions).find(([, value]) =>
            value.dimensionA === pair.dimensionA && value.dimensionB === pair.dimensionB
        )?.[0] || null;
    };

    const findFirstEmptyPairSlot = (dimensions) => {
        return ['pair1', 'pair2', 'pair3'].find(key => {
            const pair = dimensions[key];
            return !pair.dimensionA && !pair.dimensionB;
        }) || null;
    };

    // 点击建议卡片，按顺序填入空的 pair 槽位；再次点击则清空该槽位
    const handleToggleSuggestion = (pair) => {
        setCustomDimensions(prev => {
            const existingSlot = findPairSlotForSuggestion(pair, prev);
            if (existingSlot) {
                return {
                    ...prev,
                    [existingSlot]: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' }
                };
            }

            const emptySlot = findFirstEmptyPairSlot(prev);
            if (!emptySlot) return prev;

            return {
                ...prev,
                [emptySlot]: {
                    dimensionA: pair.dimensionA,
                    dimensionB: pair.dimensionB,
                    descriptionA: pair.descriptionA || '',
                    descriptionB: pair.descriptionB || ''
                }
            };
        });
    };

    // 更新自定义输入框
    const handleInputChange = (pairIndex, dimension, value) => {
        setCustomDimensions(prev => ({
            ...prev,
            [`pair${pairIndex}`]: {
                ...prev[`pair${pairIndex}`],
                [dimension]: value
            }
        }));
    };

    // 确认选择
    const handleConfirm = () => {
        const { pair1, pair2, pair3 } = customDimensions;

        // 验证所有输入框都已填写
        if (
            !pair1.dimensionA || !pair1.dimensionB ||
            !pair2.dimensionA || !pair2.dimensionB ||
            !pair3.dimensionA || !pair3.dimensionB
        ) {
            alert('Please fill in all dimension fields or select from suggestions above.');
            return;
        }

        // 返回3对维度
        const selectedPairs = [
            {
                dimensionA: pair1.dimensionA,
                dimensionB: pair1.dimensionB,
                descriptionA: pair1.descriptionA,
                descriptionB: pair1.descriptionB
            },
            {
                dimensionA: pair2.dimensionA,
                dimensionB: pair2.dimensionB,
                descriptionA: pair2.descriptionA,
                descriptionB: pair2.descriptionB
            },
            {
                dimensionA: pair3.dimensionA,
                dimensionB: pair3.dimensionB,
                descriptionA: pair3.descriptionA,
                descriptionB: pair3.descriptionB
            }
        ];

        onConfirm(selectedPairs);

        // 重置状态
        setCustomDimensions({
            pair1: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' },
            pair2: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' },
            pair3: { dimensionA: '', dimensionB: '', descriptionA: '', descriptionB: '' }
        });
    };

    if (!isOpen) return null;

    return (
        <div className="dimension-selector-panel" data-panel-root="dimension-selector">
            <div className="panel-header">
                <h3>Select Evaluation Dimensions</h3>
                <button className="close-btn" onClick={onClose}>×</button>
            </div>

            {isLoading && (
                <div className="panel-loading">
                    <div className="spinner"></div>
                    <p>Generating dimension suggestions...</p>
                </div>
            )}

            {error && (
                <div className="panel-error">
                    <p>{error}</p>
                    <button onClick={fetchSuggestedDimensions}>Retry</button>
                </div>
            )}

            {!isLoading && !error && suggestedPairs.length > 0 && (
                <>
                    {/* 左右两列布局 */}
                    <div className="panel-two-columns">
                        {/* 左列：AI 建议的预设选项 */}
                        <div className="suggestions-section">
                            <h4>AI Suggested Dimensions (click to select)</h4>
                            <div className="suggestions-list">
                                {suggestedPairs.map((pair, index) => {
                                    const selectedSlot = findPairSlotForSuggestion(pair, customDimensions);
                                    const isSelected = !!selectedSlot;

                                    return (
                                        <div
                                            key={index}
                                            className={`suggestion-card ${isSelected ? 'selected' : ''}`}
                                            onClick={() => handleToggleSuggestion(pair)}
                                        >
                                            {isSelected && (
                                                <span className="selected-badge">Pair {selectedSlot.replace('pair', '')}</span>
                                            )}

                                            <div className="suggestion-content">
                                                <div className="dimension-side side-a">
                                                    <span className="dimension-label-a">{pair.dimensionA}</span>
                                                    {pair.descriptionA && (
                                                        <p className="description-label-a">{pair.descriptionA}</p>
                                                    )}
                                                </div>

                                                <div className="vs-separator-vertical">vs</div>

                                                <div className="dimension-side side-b">
                                                    <span className="dimension-label-b">{pair.dimensionB}</span>
                                                    {pair.descriptionB && (
                                                        <p className="description-label-b">{pair.descriptionB}</p>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* 右列：用户选择的3个槽位 */}
                        <div className="custom-section">
                            <h4>Your Selected Dimensions</h4>
                            <div className="custom-inputs-box">
                                {/* Pair 1 */}
                                <div className="dimension-pair-inputs">
                                    <div className="pair-label">Pair 1 (X-axis)</div>
                                    <div className="pair-row">
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension A Name"
                                                value={customDimensions.pair1.dimensionA}
                                                onChange={(e) => handleInputChange(1, 'dimensionA', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension A"
                                                value={customDimensions.pair1.descriptionA}
                                                onChange={(e) => handleInputChange(1, 'descriptionA', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                        <span className="input-separator">vs</span>
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension B Name"
                                                value={customDimensions.pair1.dimensionB}
                                                onChange={(e) => handleInputChange(1, 'dimensionB', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension B"
                                                value={customDimensions.pair1.descriptionB}
                                                onChange={(e) => handleInputChange(1, 'descriptionB', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Pair 2 */}
                                <div className="dimension-pair-inputs">
                                    <div className="pair-label">Pair 2 (Y-axis)</div>
                                    <div className="pair-row">
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension A Name"
                                                value={customDimensions.pair2.dimensionA}
                                                onChange={(e) => handleInputChange(2, 'dimensionA', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension A"
                                                value={customDimensions.pair2.descriptionA}
                                                onChange={(e) => handleInputChange(2, 'descriptionA', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                        <span className="input-separator">vs</span>
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension B Name"
                                                value={customDimensions.pair2.dimensionB}
                                                onChange={(e) => handleInputChange(2, 'dimensionB', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension B"
                                                value={customDimensions.pair2.descriptionB}
                                                onChange={(e) => handleInputChange(2, 'descriptionB', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Pair 3 */}
                                <div className="dimension-pair-inputs">
                                    <div className="pair-label">Pair 3 (Z-axis)</div>
                                    <div className="pair-row">
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension A Name"
                                                value={customDimensions.pair3.dimensionA}
                                                onChange={(e) => handleInputChange(3, 'dimensionA', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension A"
                                                value={customDimensions.pair3.descriptionA}
                                                onChange={(e) => handleInputChange(3, 'descriptionA', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                        <span className="input-separator">vs</span>
                                        <div className="dimension-field-group">
                                            <input
                                                type="text"
                                                className="dimension-input"
                                                placeholder="Dimension B Name"
                                                value={customDimensions.pair3.dimensionB}
                                                onChange={(e) => handleInputChange(3, 'dimensionB', e.target.value)}
                                            />
                                            <textarea
                                                className="description-input"
                                                placeholder="Description for Dimension B"
                                                value={customDimensions.pair3.descriptionB}
                                                onChange={(e) => handleInputChange(3, 'descriptionB', e.target.value)}
                                                rows={3}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 底部操作按钮 */}
                    <div className="panel-footer">
                        <button className="btn-cancel" onClick={onClose}>
                            Cancel
                        </button>
                        <button className="btn-confirm" onClick={handleConfirm}>
                            Confirm & Generate Ideas
                        </button>
                    </div>
                </>
            )}
        </div>
    );
};

export default DimensionSelectorPanel;
