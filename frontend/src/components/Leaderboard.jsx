import React from 'react';

const Leaderboard = ({ leaderboard }) => {
    
  const getDifficulty = (mode) => {
    switch(mode) {
      case 'Efficiency': return 'Easy (Efficiency)';
      case 'Fairness': return 'Medium (Fairness)';
      case 'Real-Time': return 'Hard (Real-Time)';
      default: return 'N/A';
    }
  };

  return (
    <div className="leaderboard-container">
      <h3>Top Players</h3>
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Score</th>
            <th>Difficulty</th>
          </tr>
        </thead>
        <tbody>
          {leaderboard.map((player, index) => (
            <tr key={index}>
              <td>{index + 1}</td>
              <td>{player.name}</td>
              <td>{player.score}</td>
              <td>{getDifficulty(player.mode)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Leaderboard;
