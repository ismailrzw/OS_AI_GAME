import React from 'react';

const MetricDisplay = ({ label, value, isOptimal, hidePenalties }) => {
    const className = `result-metric ${hidePenalties ? '' : (isOptimal === true ? 'good' : isOptimal === false ? 'bad' : '')}`;
    return (
        <div>
            {label}: <span className={className}>{value}</span>
        </div>
    );
}

const ResultsDashboard = ({ results, explanation, difficulty }) => {
  if (!results) {
    return (
        <div className="results-dashboard">
            <div className="result-card placeholder">Human Results</div>
            <div className="result-card placeholder">AI Results</div>
        </div>
    );
  }
  
  const { human, ai } = results;
  const hidePenalties = difficulty === 'Hard'; // Hide detailed optimal/suboptimal indicators

  return (
    <div className="results-dashboard">
      <div className="result-card">
        <h3>Your Results</h3>
        <MetricDisplay 
            label="Algorithm" 
            value={human.algorithm}
        />
        <MetricDisplay 
            label="Avg. Wait Time" 
            value={human.metrics.average_waiting_time.toFixed(2)}
            isOptimal={human.metrics.avg_wait_is_optimal}
            hidePenalties={hidePenalties}
        />
        <MetricDisplay 
            label="Context Switches" 
            value={human.metrics.context_switches}
            isOptimal={human.metrics.switches_are_optimal}
            hidePenalties={hidePenalties}
        />
      </div>
      <div className="result-card">
        <h3>AI Results</h3>
        <MetricDisplay 
            label="Algorithm" 
            value={ai.algorithm}
        />
        <MetricDisplay 
            label="Avg. Wait Time" 
            value={ai.metrics.average_waiting_time.toFixed(2)}
            isOptimal={ai.metrics.avg_wait_is_optimal}
            hidePenalties={hidePenalties}
        />
        <MetricDisplay 
            label="Context Switches" 
            value={ai.metrics.context_switches}
            isOptimal={ai.metrics.switches_are_optimal}
            hidePenalties={hidePenalties}
        />
      </div>
      {explanation && (
        <div className="explanation-card">
          <h3>Outcome Explanation</h3>
          <p>{explanation}</p>
        </div>
      )}
    </div>
  );
};

export default ResultsDashboard;
