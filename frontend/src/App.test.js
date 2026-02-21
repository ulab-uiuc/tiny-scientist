import { render, screen } from '@testing-library/react';

jest.mock('./components/TreePlotVisualization', () => {
  const React = require('react');
  return function MockedTreePlotVisualization() {
    return React.createElement('div', { 'data-testid': 'tree-plot-visualization' });
  };
});

import App from './App';

test('renders the main visualization container', () => {
  render(<App />);
  expect(screen.getByTestId('tree-plot-visualization')).toBeInTheDocument();
});
