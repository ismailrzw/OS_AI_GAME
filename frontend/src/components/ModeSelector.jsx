import React from 'react';

const MODES = ['Real-Time'];

const ModeSelector = ({ selectedMode, onModeChange }) => {
  return (
    <div className="mode-selector">
      {MODES.map((mode) => (
        <button
          key={mode}
          className={selectedMode === mode ? 'active' : ''}
          onClick={() => onModeChange(mode)}
        >
          {mode}
        </button>
      ))}
    </div>
  );
};

export default ModeSelector;
