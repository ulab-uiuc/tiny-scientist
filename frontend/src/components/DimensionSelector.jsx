import React, { useState, useEffect, useCallback } from 'react';
import './DimensionSelector.css';

const DimensionSelector = ({ intent, onDimensionsSelected }) => {
    const [suggestedPairs, setSuggestedPairs] = useState([]);
    const [selectedPairs, setSelectedPairs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showCustomInput, setShowCustomInput] = useState(false);
    const [customPair, setCustomPair] = useState({
        dimensionA: '',
        dimensionB: '',
        descriptionA: '',
        descriptionB: '',
    });

    const fetchDimensionSuggestions = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch('/api/suggest-dimensions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ intent }),
            });
            const data = await response.json();
            setSuggestedPairs(data.dimension_pairs || []);
        } catch (error) {
            console.error('Error fetching dimension suggestions:', error);
        } finally {
            setLoading(false);
        }
    }, [intent]);

    useEffect(() => {
        if (intent) {
            fetchDimensionSuggestions();
        }
    }, [intent, fetchDimensionSuggestions]);

    const togglePairSelection = (pair) => {
        if (selectedPairs.find(p => p.dimensionA === pair.dimensionA && p.dimensionB === pair.dimensionB)) {
            setSelectedPairs(selectedPairs.filter(p => !(p.dimensionA === pair.dimensionA && p.dimensionB === pair.dimensionB)));
        } else {
            if (selectedPairs.length < 3) {
                setSelectedPairs([...selectedPairs, pair]);
            } else {
                // Replace the oldest selected pair to keep recency
                setSelectedPairs([...selectedPairs.slice(1), pair]);
            }
        }
    };

    const handleCustomPairSubmit = () => {
        if (customPair.dimensionA && customPair.dimensionB) {
            if (selectedPairs.length < 3) {
                setSelectedPairs([...selectedPairs, customPair]);
            } else {
                setSelectedPairs([...selectedPairs.slice(1), customPair]);
            }
            setCustomPair({
                dimensionA: '',
                dimensionB: '',
                descriptionA: '',
                descriptionB: '',
            });
            setShowCustomInput(false);
        }
    };

    const handleConfirm = () => {
        if (selectedPairs.length === 3) {
            onDimensionsSelected(selectedPairs);
        }
    };

    const isPairSelected = (pair) => {
        return selectedPairs.some(p => p.dimensionA === pair.dimensionA && p.dimensionB === pair.dimensionB);
    };

    if (loading) {
        return (
            <div className="dimension-selector">
                <div className="dimension-selector-loading">
                    <div className="spinner"></div>
                    <p>Suggesting evaluation dimensions...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="dimension-selector">
            <div className="dimension-selector-header">
                <h3>Select Evaluation Dimensions</h3>
                <p className="dimension-selector-subtitle">
                    Choose 3 dimension pairs to evaluate your ideas ({selectedPairs.length}/3 selected)
                </p>
            </div>

            <div className="dimension-pairs-grid">
                {suggestedPairs.map((pair, index) => (
                    <div
                        key={index}
                        className={`dimension-pair-card ${isPairSelected(pair) ? 'selected' : ''}`}
                        data-tracking-click="true"
                        data-tracking-name={`dimension_pair_${pair.dimensionA}_${pair.dimensionB}`}
                        onClick={() => togglePairSelection(pair)}
                    >
                        <div className="dimension-pair-spectrum">
                            <div className="dimension-end dimension-a">
                                <span className="dimension-label">{pair.dimensionA}</span>
                                <span className="dimension-score">0</span>
                            </div>
                            <div className="spectrum-line">
                                <div className="spectrum-arrow">←</div>
                                <div className="spectrum-center">50</div>
                                <div className="spectrum-arrow">→</div>
                            </div>
                            <div className="dimension-end dimension-b">
                                <span className="dimension-label">{pair.dimensionB}</span>
                                <span className="dimension-score">100</span>
                            </div>
                        </div>
                        <p className="dimension-description">{pair.descriptionA}</p>
                        <p className="dimension-description">{pair.descriptionB}</p>
                        {pair.explanation && (
                            <p className="dimension-explanation">{pair.explanation}</p>
                        )}
                    </div>
                ))}
            </div>

            {!showCustomInput && (
                <button
                    className="custom-dimension-button"
                    onClick={() => setShowCustomInput(true)}
                >
                    + Add Custom Dimension Pair
                </button>
            )}

            {showCustomInput && (
                <div className="custom-dimension-input">
                    <h4>Custom Dimension Pair</h4>
                    <div className="custom-input-row">
                        <div className="custom-input-group">
                            <label>Dimension A (0)</label>
                            <input
                                type="text"
                                placeholder="e.g., HCI-oriented"
                                value={customPair.dimensionA}
                                onChange={(e) => setCustomPair({ ...customPair, dimensionA: e.target.value })}
                            />
                            <textarea
                                placeholder="Description for Dimension A"
                                value={customPair.descriptionA}
                                onChange={(e) => setCustomPair({ ...customPair, descriptionA: e.target.value })}
                            />
                        </div>
                        <div className="custom-input-divider">↔</div>
                        <div className="custom-input-group">
                            <label>Dimension B (100)</label>
                            <input
                                type="text"
                                placeholder="e.g., AI-oriented"
                                value={customPair.dimensionB}
                                onChange={(e) => setCustomPair({ ...customPair, dimensionB: e.target.value })}
                            />
                            <textarea
                                placeholder="Description for Dimension B"
                                value={customPair.descriptionB}
                                onChange={(e) => setCustomPair({ ...customPair, descriptionB: e.target.value })}
                            />
                        </div>
                    </div>
                    <div className="custom-input-actions">
                        <button onClick={handleCustomPairSubmit}>Add Custom Pair</button>
                        <button onClick={() => setShowCustomInput(false)}>Cancel</button>
                    </div>
                </div>
            )}

            {selectedPairs.length > 0 && (
                <div className="selected-pairs-summary">
                    <h4>Selected Dimensions:</h4>
                    <div className="selected-pairs-list">
                        {selectedPairs.map((pair, index) => (
                            <div key={index} className="selected-pair-item">
                                <span className="pair-number">Dimension {index + 1}:</span>
                                <span className="pair-text">
                                    {pair.dimensionA} ←→ {pair.dimensionB}
                                </span>
                                <button
                                    className="remove-pair-button"
                                    onClick={() => setSelectedPairs(selectedPairs.filter((_, i) => i !== index))}
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <button
                className="confirm-dimensions-button"
                onClick={handleConfirm}
                disabled={selectedPairs.length !== 3}
            >
                Confirm Dimensions ({selectedPairs.length}/3)
            </button>
        </div>
    );
};

export default DimensionSelector;
