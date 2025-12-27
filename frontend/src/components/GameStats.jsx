import React from 'react';

const GameStats = ({ playerName, stats }) => {
  const { wins, losses, ties, total } = stats;
  return (
    <div className="game-stats-container">
      <h3>Player Stats: {playerName}</h3>
      <div className="stats-grid">
        <span>Wins:</span><span>{wins}</span>
        <span>Losses:</span><span>{losses}</span>
        <span>Ties:</span><span>{ties}</span>
        <span>Total:</span><span>{total}</span>
      </div>
    </div>
  );
};

export default GameStats;
