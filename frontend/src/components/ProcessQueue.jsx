import React, { useState, useRef } from 'react';

const ProcessQueue = ({ mode, processes, setProcesses, onSubmit, disabled }) => {
  const [draggingItem, setDraggingItem] = useState(null);
  const dragItemNode = useRef(null);

  const handleDragStart = (e, index) => {
    setDraggingItem(index);
    dragItemNode.current = e.target;
    dragItemNode.current.addEventListener('dragend', handleDragEnd);
    setTimeout(() => {
      // Use a timeout to allow DOM update before applying dragging class
      e.target.classList.add('dragging');
    }, 0);
  };

  const handleDragEnter = (e, targetIndex) => {
    if (dragItemNode.current !== e.target) {
      const newList = [...processes];
      const draggedItemContent = newList.splice(draggingItem, 1)[0];
      newList.splice(targetIndex, 0, draggedItemContent);
      setDraggingItem(targetIndex);
      setProcesses(newList);
    }
  };
  
  const handleDragEnd = (e) => {
      e.target.classList.remove('dragging');
      dragItemNode.current.removeEventListener('dragend', handleDragEnd);
      setDraggingItem(null);
      dragItemNode.current = null;
  }

  return (
    <div className="process-queue-container">
      <h3>Your Schedule (Drag to Reorder)</h3>
      <ul className="process-list">
        {processes.map((p, index) => (
          <li
            key={p.pid}
            draggable
            onDragStart={(e) => handleDragStart(e, index)}
            onDragEnter={(e) => handleDragEnter(e, index)}
            className="process-item"
          >
            {mode === 'Real-Time' ? (
              <>
                <span>Case: {p.case_type}</span>
                <span>Severity: {p.severity}</span>
                <span>Duration: {p.burst_time}</span>
                <span>Arrived: {p.arrival_time}</span>
              </>
            ) : (
              <>
                <span>PID: {p.pid}</span>
                <span>Arrival: {p.arrival_time}</span>
                <span>Burst: {p.burst_time}</span>
              </>
            )}
          </li>
        ))}
      </ul>
      <button onClick={onSubmit} disabled={disabled} className="process-submit-button">
        Run Simulation
      </button>
    </div>
  );
};

export default ProcessQueue;
