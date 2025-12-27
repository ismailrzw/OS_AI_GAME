import React, { useState } from 'react';

const NamePopup = ({ onSubmit }) => {
  const [name, setName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) {
      onSubmit(name.trim());
    }
  };

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Welcome to the Scheduler Simulation</h2>
        <p>Please enter your name to play.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your Name"
            className="popup-input"
            autoFocus
          />
          <button type="submit" className="popup-button">Start Playing</button>
        </form>
      </div>
    </div>
  );
};

export default NamePopup;
