import React, { useState, useEffect } from 'react';

const Timer = ({ duration, onTimeout, onReset }) => {
  const [timeLeft, setTimeLeft] = useState(duration);

  useEffect(() => {
    setTimeLeft(duration);
  }, [duration, onReset]); // Reset when duration or onReset flag changes

  useEffect(() => {
    if (timeLeft <= 0) {
      onTimeout();
      return;
    }

    const intervalId = setInterval(() => {
      setTimeLeft(timeLeft - 1);
    }, 1000);

    return () => clearInterval(intervalId);
  }, [timeLeft, onTimeout]);

  return (
    <div className="timer-container">
      Time Left: {timeLeft}s
    </div>
  );
};

export default Timer;
