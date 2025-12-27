// src/services/api.js

/**
 * This file mocks the backend API.
 * In a real application, this would make HTTP requests to the server.
 * The data structures returned here are designed to match the expected
 * backend responses.
 */

// A helper function to generate random process data for the mock API
const generateRandomProcesses = (num) => {
    const processes = [];
    for (let i = 0; i < num; i++) {
        processes.push({
            pid: i,
            arrival_time: Math.floor(Math.random() * 10),
            burst_time: Math.floor(Math.random() * 15) + 1,
            priority: Math.floor(Math.random() * 5) + 1,
        });
    }
    // Sort by arrival time, as the backend would
    return processes.sort((a, b) => a.arrival_time - b.arrival_time);
};

// A helper to simulate a scheduling algorithm and return metrics
const simulateScheduling = (processes, algorithm) => {
    // This is a simplified, fake simulation for mock purposes.
    let gantt = [];
    let time = 0;
    let waitingTime = 0;
    let contextSwitches = 0;
    
    // Naive simulation logic
    processes.forEach((p, index) => {
        if (time < p.arrival_time) {
            time = p.arrival_time;
        }
        const wait = time - p.arrival_time;
        waitingTime += wait > 0 ? wait : 0;
        
        gantt.push({ pid: p.pid, start: time, end: time + p.burst_time });
        time += p.burst_time;
        if (index < processes.length - 1) {
             contextSwitches += (Math.random() > 0.5 ? 1 : 0); // Randomly add switches
        }
    });

    return {
        metrics: {
            average_waiting_time: waitingTime / processes.length,
            average_turnaround_time: (time - processes[0].arrival_time) / processes.length,
            context_switches: contextSwitches
        },
        gantt_chart_log: gantt,
    };
};

export const startNewGame = async (mode) => {
    console.log(`[API MOCK] Starting new game in ${mode} mode.`);
    const processCount = Math.floor(Math.random() * 5) + 4; // 4-8 processes
    const processes = generateRandomProcesses(processCount);
    return { processes };
};

export const submitSchedule = async (mode, humanSchedule) => {
    console.log(`[API MOCK] Submitting schedule for ${mode} mode.`);
    console.log('Human schedule:', humanSchedule.map(p => p.pid));

    // Simulate human run
    const humanResult = simulateScheduling(humanSchedule, 'manual');
    humanResult.algorithm = 'Manual';

    // Simulate AI run (let's pretend AI chose SJF)
    const aiAlgo = 'SJF';
    const aiSchedule = [...humanSchedule].sort((a, b) => a.burst_time - b.burst_time);
    const aiResult = simulateScheduling(aiSchedule, aiAlgo);
    aiResult.algorithm = aiAlgo;
    
    const calculateFinalScore = (metrics) => {
        // A simple score for comparison. Lower is better.
        return metrics.average_waiting_time + (metrics.context_switches * 2);
    }
    
    humanResult.metrics.final_score = calculateFinalScore(humanResult.metrics);
    aiResult.metrics.final_score = calculateFinalScore(aiResult.metrics);

    // Add some random coloring logic for results
    if (humanResult.metrics.final_score < aiResult.metrics.final_score) {
        humanResult.metrics.avg_wait_is_optimal = true;
        aiResult.metrics.avg_wait_is_optimal = false;
    } else {
        humanResult.metrics.avg_wait_is_optimal = false;
        aiResult.metrics.avg_wait_is_optimal = true;
    }
     if (humanResult.metrics.context_switches < aiResult.metrics.context_switches) {
        humanResult.metrics.switches_are_optimal = true;
        aiResult.metrics.switches_are_optimal = false;
    } else {
        humanResult.metrics.switches_are_optimal = false;
        aiResult.metrics.switches_are_optimal = true;
    }


    return {
        human: humanResult,
        ai: aiResult,
    };
};

export const getLeaderboard = async () => {
    console.log('[API MOCK] Fetching leaderboard data.');
    // In a real app, this would be a database call.
    // Here, we return a static, sorted list.
    return [
        { name: 'HAL 9000', score: 18, mode: 'Efficiency' },
        { name: 'Skynet', score: 22, mode: 'Real-Time' },
        { name: 'Jane', score: 25, mode: 'Fairness' },
        { name: 'Bob', score: 31, mode: 'Efficiency' },
        { name: 'Alice', score: 45, mode: 'Fairness' },
    ];
};
