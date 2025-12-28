import React, { useState, useEffect, useCallback } from 'react';

// Import API service and components
import * as api from './services/api';
import ModeSelector from './components/ModeSelector';
import ProcessQueue from './components/ProcessQueue';
import GanttChart from './components/GanttChart';
import Timer from './components/Timer';
import ResultsDashboard from './components/ResultsDashboard';
import NamePopup from './components/NamePopup';
import GameStats from './components/GameStats';
import Leaderboard from './components/Leaderboard';

const initialStats = { wins: 0, losses: 0, ties: 0, total: 0 };

function App() {
  // Core simulation state
  const [mode, setMode] = useState('Efficiency');
  const [processes, setProcesses] = useState([]);
  const [simulationResults, setSimulationResults] = useState(null);
  const [gameState, setGameState] = useState('idle'); // idle, running, finished
  const [timerResetKey, setTimerResetKey] = useState(0);

  // New features state
  const [playerName, setPlayerName] = useState('');
  const [showNamePopup, setShowNamePopup] = useState(false);
  const [playerStats, setPlayerStats] = useState(initialStats);
  const [leaderboard, setLeaderboard] = useState([]);
  const [explanation, setExplanation] = useState(''); // New state for explanation
  const [difficulty, setDifficulty] = useState('Medium'); // New state for difficulty

  // Load player data and leaderboard on initial mount
  useEffect(() => {
    const storedName = localStorage.getItem('playerName');
    if (storedName) {
      setPlayerName(storedName);
      const storedStats = localStorage.getItem('playerStats');
      setPlayerStats(storedStats ? JSON.parse(storedStats) : initialStats);
    } else {
      setShowNamePopup(true);
    }

    const fetchLeaderboard = async () => {
      const data = await api.getLeaderboard();
      setLeaderboard(data);
    };
    fetchLeaderboard();
  }, []);
  
  // Function to start a new game
  const handleStartGame = useCallback(async (selectedMode) => {
    setGameState('running');
    setSimulationResults(null);
    setExplanation(''); // Clear previous explanation
    
    // Determine difficulty based on mode
    let currentDifficulty;
    switch(selectedMode) {
        case 'Efficiency': currentDifficulty = 'Hard'; break;
        case 'Fairness': currentDifficulty = 'Medium'; break;
        case 'Real-Time': currentDifficulty = 'Hard'; break;
        default: currentDifficulty = 'Medium';
    }
    setDifficulty(currentDifficulty);

    const { processes: newProcesses } = await api.startNewGame(selectedMode);
    setProcesses(newProcesses);
    setGameState('idle');
    setTimerResetKey(prev => prev + 1); // Reset timer
  }, []);
  
  // Effect to start game when mode changes, but only if a player name is set
  useEffect(() => {
    if (playerName) {
      handleStartGame(mode);
    }
  }, [mode, playerName, handleStartGame]);

  const handleModeChange = (newMode) => {
    setMode(newMode);
  };

  const handleSubmitName = (name) => {
    setPlayerName(name);
    localStorage.setItem('playerName', name);
    setShowNamePopup(false);
    handleStartGame(mode); // Start the first game after name is submitted
  };

  const handleSubmitSchedule = useCallback(async () => {
    if (gameState === 'running' || processes.length === 0) return;
    
    setGameState('running');
    const results = await api.submitSchedule(mode, processes);
    setSimulationResults(results);
    setExplanation(results.explanation); // Set explanation from API response
    setGameState('finished');

    // Update player stats
    const humanScore = results.human.metrics.final_score;
    const aiScore = results.ai.metrics.final_score;
    
    let newStats = { ...playerStats };
    newStats.total += 1;
    if (humanScore < aiScore) {
      newStats.wins += 1;
    } else if (humanScore > aiScore) {
      newStats.losses += 1;
    } else {
      newStats.ties += 1;
    }
    setPlayerStats(newStats);
    localStorage.setItem('playerStats', JSON.stringify(newStats));

  }, [mode, processes, gameState, playerStats]);

  const maxTime = simulationResults 
    ? Math.max(
        simulationResults.human.gantt_chart_log.at(-1)?.end || 0,
        simulationResults.ai.gantt_chart_log.at(-1)?.end || 0
      )
    : 0;
  
  const controlsDisabled = gameState === 'running' || showNamePopup || !playerName;

  return (
    <div className="app-container">
      {showNamePopup && <NamePopup onSubmit={handleSubmitName} />}

      <header>
        <h1>Adaptive OS Scheduler</h1>
        <ModeSelector selectedMode={mode} onModeChange={handleModeChange} />
      </header>

      <main className="main-content">
        <div className="controls-column">
          {playerName && <GameStats playerName={playerName} stats={playerStats} />}
          {mode === 'Real-Time' && gameState === 'idle' && (
            <Timer 
                duration={10} 
                onTimeout={handleSubmitSchedule}
                onReset={timerResetKey}
            />
          )}
          <ProcessQueue
            mode={mode}
            processes={processes}
            setProcesses={setProcesses}
            onSubmit={handleSubmitSchedule}
            disabled={controlsDisabled}
          />
           <button onClick={() => handleStartGame(mode)} disabled={controlsDisabled} className="process-submit-button" style={{backgroundColor: '#3f51b5'}}>
             New Process Set
           </button>
        </div>
        <div className="results-column">
          <Leaderboard leaderboard={leaderboard} />
          <ResultsDashboard results={simulationResults} explanation={explanation} difficulty={difficulty} />
          <GanttChart title="Your Gantt Chart" log={simulationResults?.human.gantt_chart_log} maxTime={maxTime} />
          <GanttChart title="AI Gantt Chart" log={simulationResults?.ai.gantt_chart_log} maxTime={maxTime} />
        </div>
      </main>
    </div>
  );
}

export default App;