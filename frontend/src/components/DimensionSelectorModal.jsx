import React, { useState, useEffect, useCallback } from 'react';
import './DimensionSelectorModal.css';

/**
 * 弹窗式维度选择器
 * 用户输入 Intent 后弹出,显示 3 个 AI 建议的维度对
 * 用户可以选择预设或自定义 2 对维度
 */
const DimensionSelectorModal = ({ isOpen, onClose, onConfirm, intent }) => {
    const [suggestedPairs, setSuggestedPairs] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // 用户自定义的2对维度 (4个输入框)
    const [customDimensions, setCustomDimensions] = useState({
        pair1: { dimensionA: '', dimensionB: '' },
        pair2: { dimensionA: '', dimensionB: '' }
    });

    const fetchSuggestedDimensions = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:5000/api/suggest-dimensions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ intent })
            });

            if (!response.ok) throw new Error('Failed to fetch dimension suggestions');

            const data = await response.json();
            setSuggestedPairs(data.dimension_pairs || []);
        } catch (err) {
            console.error('Error fetching dimensions:', err);
            setError('Failed to load dimension suggestions. Please try again.');
        } finally {
            setIsLoading(false);
        }
    }, [intent]);

    // 获取 AI 建议的维度对
    useEffect(() => {
        if (isOpen && intent) {
            fetchSuggestedDimensions();
        }
    }, [isOpen, intent, fetchSuggestedDimensions]);

    // 点击预设选项,填入对应的输入框
    const handleSelectPreset = (pair, targetPairIndex) => {
        setCustomDimensions(prev => ({
            ...prev,
            [`pair${targetPairIndex}`]: {
                dimensionA: pair.dimensionA,
                dimensionB: pair.dimensionB
            }
        }));
    };

    // 输入框变化
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
        const pair1 = customDimensions.pair1;
        const pair2 = customDimensions.pair2;

        // 验证必须填写2对维度
        if (!pair1.dimensionA || !pair1.dimensionB || !pair2.dimensionA || !pair2.dimensionB) {
            alert('Please fill in both dimension pairs (all 4 fields)');
            return;
        }

        // 返回选中的2对维度
        const selectedPairs = [
            { dimensionA: pair1.dimensionA, dimensionB: pair1.dimensionB },
            { dimensionA: pair2.dimensionA, dimensionB: pair2.dimensionB }
        ];

        onConfirm(selectedPairs);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="dimension-modal-overlay" onClick={onClose}>
            <div className="dimension-modal-content" data-panel-root="dimension-selector-modal" onClick={(e) => e.stopPropagation()}>
                <div className="dimension-modal-header">
                    <h2>Select Evaluation Dimensions</h2>
                    <button className="dimension-modal-close" onClick={onClose}>×</button>
                </div>

                {isLoading && (
                    <div className="dimension-modal-loading">
                        <div className="spinner"></div>
                        <p>Generating dimension suggestions...</p>
                    </div>
                )}

                {error && (
                    <div className="dimension-modal-error">
                        <p>{error}</p>
                        <button onClick={fetchSuggestedDimensions}>Retry</button>
                    </div>
                )}

                {!isLoading && !error && (
                    <>
                        {/* AI 建议的 3 对维度 */}
                        <div className="dimension-modal-suggestions">
                            <h3>AI Suggested Dimension Pairs (click to use)</h3>
                            <div className="dimension-suggestions-grid">
                                {suggestedPairs.map((pair, index) => (
                                    <div key={index} className="dimension-suggestion-card">
                                        <div className="dimension-pair-header">
                                            <span className="dimension-label">{pair.dimensionA}</span>
                                            <span className="dimension-arrow">←→</span>
                                            <span className="dimension-label">{pair.dimensionB}</span>
                                        </div>
                                        <p className="dimension-explanation">{pair.explanation}</p>
                                        <div className="dimension-actions">
                                            <button
                                                className="btn-use-as-pair1"
                                                onClick={() => handleSelectPreset(pair, 1)}
                                            >
                                                Use as Pair 1
                                            </button>
                                            <button
                                                className="btn-use-as-pair2"
                                                onClick={() => handleSelectPreset(pair, 2)}
                                            >
                                                Use as Pair 2
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* 用户自定义的 2 对维度 (4个输入框) */}
                        <div className="dimension-modal-custom">
                            <h3>Your Selected Dimensions</h3>

                            {/* 第一对维度 */}
                            <div className="dimension-custom-pair">
                                <div className="dimension-pair-label">Dimension Pair 1 (X-axis)</div>
                                <div className="dimension-input-row">
                                    <input
                                        type="text"
                                        placeholder="Dimension A (left/0)"
                                        value={customDimensions.pair1.dimensionA}
                                        onChange={(e) => handleInputChange(1, 'dimensionA', e.target.value)}
                                        className="dimension-input"
                                    />
                                    <span className="dimension-separator">←→</span>
                                    <input
                                        type="text"
                                        placeholder="Dimension B (right/100)"
                                        value={customDimensions.pair1.dimensionB}
                                        onChange={(e) => handleInputChange(1, 'dimensionB', e.target.value)}
                                        className="dimension-input"
                                    />
                                </div>
                            </div>

                            {/* 第二对维度 */}
                            <div className="dimension-custom-pair">
                                <div className="dimension-pair-label">Dimension Pair 2 (Y-axis)</div>
                                <div className="dimension-input-row">
                                    <input
                                        type="text"
                                        placeholder="Dimension A (bottom/0)"
                                        value={customDimensions.pair2.dimensionA}
                                        onChange={(e) => handleInputChange(2, 'dimensionA', e.target.value)}
                                        className="dimension-input"
                                    />
                                    <span className="dimension-separator">←→</span>
                                    <input
                                        type="text"
                                        placeholder="Dimension B (top/100)"
                                        value={customDimensions.pair2.dimensionB}
                                        onChange={(e) => handleInputChange(2, 'dimensionB', e.target.value)}
                                        className="dimension-input"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* 确认按钮 */}
                        <div className="dimension-modal-footer">
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
        </div>
    );
};

export default DimensionSelectorModal;
