import React from 'react';

const GanttChart = ({ title, log, maxTime }) => {
  if (!log || log.length === 0) {
    return (
        <div className="gantt-chart-container">
            <h3>{title}</h3>
            <div className="placeholder">Run simulation to see Gantt Chart</div>
        </div>
    );
  }

  return (
    <div className="gantt-chart-container">
      <h3>{title}</h3>
      <div className="gantt-chart">
        {log.map((entry, index) => {
          const left = (entry.start / maxTime) * 100;
          const width = ((entry.end - entry.start) / maxTime) * 100;
          return (
            <div
              key={index}
              className="gantt-bar"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                // Add some color variation for different processes
                backgroundColor: `hsl(${entry.pid * 60}, 70%, 50%)`
              }}
              title={`PID: ${entry.pid} (${entry.start}-${entry.end})`}
            >
              P{entry.pid}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default GanttChart;
