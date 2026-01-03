import React from 'react';

const ALGORITHMS = [
  { id: 'fcfs', label: 'FCFS' },
  { id: 'sjf', label: 'SJF' },
  { id: 'rr', label: 'Round Robin' },
  { id: 'priority', label: 'Priority' }
];

const AlgorithmSelector = ({ selectedAlgorithm, onAlgorithmChange, disabled }) => {
  return (
    <div className="algorithm-selector" style={{ marginTop: '1rem' }}>
      <h4 style={{ marginBottom: '0.5rem', color: '#888' }}>Select Your Algorithm:</h4>
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
        {ALGORITHMS.map((algo) => (
          <button
            key={algo.id}
            disabled={disabled}
            className={selectedAlgorithm === algo.id ? 'active' : ''}
            onClick={() => onAlgorithmChange(algo.id)}
            style={{
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid ' + (selectedAlgorithm === algo.id ? 'var(--primary-color)' : 'var(--border-color)'),
              backgroundColor: selectedAlgorithm === algo.id ? 'var(--primary-color)' : 'var(--card-background-color)',
              color: 'white',
              cursor: disabled ? 'not-allowed' : 'pointer',
              fontSize: '0.85em',
              transition: 'all 0.2s'
            }}
          >
            {algo.label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default AlgorithmSelector;
