// src/services/api.js

/**
 * This file mocks the backend API.
 * In a real application, this would make HTTP requests to the server.
 * The data structures returned here are designed to match the expected
 * backend responses.
 */

// A helper function to generate random process data for the mock API
const generateRandomProcesses = (num, mode) => {
    const processes = [];
    const MEDICAL_CASES_DATA = {
        'Stroke': { severity: 'Critical', priority: 1, duration: [15, 20] },
        'Heart Attack': { severity: 'Critical', priority: 1, duration: [15, 20] },
        'Trauma': { severity: 'High', priority: 2, duration: [10, 18] },
        'Asthma Attack': { severity: 'High', priority: 2, duration: [8, 12] },
        'Infection': { severity: 'Medium', priority: 3, duration: [5, 10] },
        'Broken Bone': { severity: 'Medium', priority: 3, duration: [8, 15] },
        'Fever': { severity: 'Low', priority: 4, duration: [3, 6] },
        'Migraine': { severity: 'Low', priority: 4, duration: [2, 5] },
    };

    for (let i = 0; i < num; i++) {
        if (mode === 'Real-Time') {
            const caseType = Object.keys(MEDICAL_CASES_DATA)[Math.floor(Math.random() * Object.keys(MEDICAL_CASES_DATA).length)];
            const caseData = MEDICAL_CASES_DATA[caseType];
            const [minDuration, maxDuration] = caseData.duration;
            processes.push({
                pid: i,
                arrival_time: Math.floor(Math.random() * 10),
                burst_time: Math.floor(Math.random() * (maxDuration - minDuration + 1)) + minDuration,
                priority: caseData.priority,
                case_type: caseType,
                severity: caseData.severity,
            });
        } else {
            processes.push({
                pid: i,
                arrival_time: Math.floor(Math.random() * 10),
                burst_time: Math.floor(Math.random() * 15) + 1,
                priority: Math.floor(Math.random() * 5) + 1,
            });
        }
    }
    return processes.sort((a, b) => a.arrival_time - b.arrival_time);
};

// A helper to simulate a scheduling algorithm and return metrics more realistically
const simulateScheduling = (processes, algorithm, mode) => {
    let gantt = [];
    let time = 0;
    let waitingTimes = [];
    let contextSwitches = 0;
    let preemptions = 0;
    
    // Detailed mock simulation logic
    let currentProcesses = JSON.parse(JSON.stringify(processes)); // Deep copy
    currentProcesses.forEach(p => {
        p.remaining_time = p.burst_time;
        p.start_time = -1;
        p.finish_time = -1;
        p.waiting_time = 0;
    });

    let readyQueue = [];
    let completedProcesses = [];
    let cpuOccupied = false;
    let currentCpuProcess = null;
    let lastPid = -1;

    // To prevent infinite loops in mock if logic isn't perfect
    let safetyBreak = 0; 
    const MAX_SIM_TIME = 1000; 

    while (completedProcesses.length < processes.length && safetyBreak < MAX_SIM_TIME) {
        safetyBreak++;
        
        // Add arrived processes to ready queue
        currentProcesses.filter(p => p.arrival_time <= time && !readyQueue.includes(p) && !completedProcesses.includes(p) && p !== currentCpuProcess)
                         .forEach(p => readyQueue.push(p));
        
        // Handle preemption (for SRTF/Priority)
        if (currentCpuProcess && ['SRTF', 'Priority'].includes(algorithm)) {
            // Find highest priority/shortest remaining time in ready queue
            let highestPrioShortestTimeProcess = null;
            if (readyQueue.length > 0) {
                highestPrioShortestTimeProcess = readyQueue.reduce((prev, curr) => {
                    if (algorithm === 'Priority') return (prev.priority < curr.priority) ? prev : curr;
                    else return (prev.remaining_time < curr.remaining_time) ? prev : curr;
                });
            }

            if (highestPrioShortestTimeProcess) {
                const currentVal = algorithm === 'Priority' ? currentCpuProcess.priority : currentCpuProcess.remaining_time;
                const nextVal = algorithm === 'Priority' ? highestPrioShortestTimeProcess.priority : highestPrioShortestTimeProcess.remaining_time;

                if (nextVal < currentVal) { // Lower value is higher priority/shorter time
                    contextSwitches++;
                    preemptions++;
                    readyQueue.push(currentCpuProcess); // Put current back
                    currentCpuProcess = null; // Preempt
                    // Sort ready queue after preemption to ensure correct next pick
                    readyQueue.sort((a,b) => (algorithm === 'Priority' ? a.priority - b.priority : a.remaining_time - b.remaining_time));
                }
            }
        }
        
        // Select next process for CPU if current one is null
        if (!currentCpuProcess && readyQueue.length > 0) {
            // Sort ready queue before picking for SJF/SRTF/Priority
            if (['SJF', 'SRTF', 'Priority'].includes(algorithm)) {
                 readyQueue.sort((a,b) => {
                    if (algorithm === 'Priority') return a.priority - b.priority;
                    else return a.burst_time - b.burst_time; // For SJF/SRTF at pick time
                 });
            }
            
            const nextProcess = readyQueue.shift();
            if (lastPid !== -1 && lastPid !== nextProcess.pid) {
                contextSwitches++;
            }
            currentCpuProcess = nextProcess;
            if (currentCpuProcess.start_time === -1) {
                currentCpuProcess.start_time = time;
            }
        }

        // Execute process for one time unit (or quantum for RR, or to completion for non-preemptive)
        if (currentCpuProcess) {
            const execStart = time;
            let execDuration = 1;

            if (algorithm === 'RR') {
                execDuration = Math.min(currentCpuProcess.remaining_time, 4); // Mock time quantum
            } else if (['FCFS', 'SJF', 'Priority'].includes(algorithm) && currentCpuProcess.remaining_time === currentCpuProcess.burst_time) {
                // For non-preemptive, if just starting, run to completion
                // Simplified mock: just run for total burst to avoid complex idle/arrival logic
                execDuration = currentCpuProcess.remaining_time;
            }
            
            // For preemptive, run one tick at a time to allow new arrivals to trigger preemption
            if (['SRTF', 'Priority'].includes(algorithm) && currentCpuProcess.remaining_time > 1) {
                execDuration = 1;
            }


            time += execDuration;
            currentCpuProcess.remaining_time -= execDuration;
            gantt.push({ pid: currentCpuProcess.pid, start: execStart, end: time });

            if (currentCpuProcess.remaining_time <= 0) {
                currentCpuProcess.finish_time = time;
                currentCpuProcess.waiting_time = currentCpuProcess.finish_time - currentCpuProcess.arrival_time - currentCpuProcess.burst_time;
                waitingTimes.push(currentCpuProcess.waiting_time);
                completedProcesses.push(currentCpuProcess);
                lastPid = currentCpuProcess.pid;
                currentCpuProcess = null;
            } else if (algorithm === 'RR' && execDuration === 4) { // Finished a quantum
                contextSwitches++;
                preemptions++; // RR switches are preemptions
                readyQueue.push(currentCpuProcess); // Re-add to end of queue
                lastPid = currentCpuProcess.pid;
                currentCpuProcess = null; // CPU becomes available
            } else {
                lastPid = currentCpuProcess.pid;
            }
        } else {
            // CPU idle, advance time to next process arrival
            const nextArrivalTime = currentProcesses.filter(p => !completedProcesses.includes(p) && !readyQueue.includes(p)).map(p => p.arrival_time);
            if (nextArrivalTime.length > 0) {
                time = Math.max(time + 1, Math.min(...nextArrivalTime));
            } else {
                time++; // No more arrivals, just advance time until completed loop condition
            }
        }
    }

    const avgWaitingTime = waitingTimes.length > 0 ? waitingTimes.reduce((sum, w) => sum + w, 0) / waitingTimes.length : 0;
    const maxWaitingTime = waitingTimes.length > 0 ? Math.max(...waitingTimes) : 0;
    const waitingTimeVariance = waitingTimes.length > 0 ? waitingTimes.reduce((sum, w) => sum + Math.pow(w - avgWaitingTime, 2), 0) / waitingTimes.length : 0;
    
    return {
        metrics: {
            average_waiting_time: avgWaitingTime,
            max_waiting_time: maxWaitingTime,
            waiting_time_variance: waitingTimeVariance,
            context_switches: contextSwitches,
            preemption_count: preemptions,
        },
        gantt_chart_log: gantt,
        completed_processes: completedProcesses, // Return completed processes for explanation logic
    };
};

export const startNewGame = async (mode) => {
    console.log(`[API MOCK] Starting new game in ${mode} mode.`);
    const processCount = Math.floor(Math.random() * 5) + 4; // 4-8 processes
    const processes = generateRandomProcesses(processCount, mode); // Pass mode to generator
    return { processes };
};

export const submitSchedule = async (mode, humanSchedule) => {
    console.log(`[API MOCK] Submitting schedule for ${mode} mode.`);
    // Ensure humanSchedule processes have Real-Time mode specific fields if mode is Real-Time
    const finalHumanSchedule = humanSchedule.map(p => {
        if (mode === 'Real-Time' && !p.case_type) {
            // Re-generate a medical case for it to have the fields
            const MEDICAL_CASES_DATA = { /* ... copied from generateRandomProcesses ... */ }; // Need to copy or refactor
            const caseTypes = Object.keys(MEDICAL_CASES_DATA);
            const randomCaseType = caseTypes[Math.floor(Math.random() * caseTypes.length)];
            const caseData = MEDICAL_CASES_DATA[randomCaseType];
            return {
                ...p,
                case_type: randomCaseType,
                severity: caseData.severity,
            };
        }
        return p;
    });


    // Define mock cost weights for explanation generation (simplified for frontend mock)
    const MOCK_COST_WEIGHTS = {
        'Efficiency': { avg_wait_weight: 1.5, fairness_weight: 0.05, starvation_weight: 0.2, cs_penalty: 0.5, pre_cost: 0.5 },
        'Fairness': { avg_wait_weight: 0.1, fairness_weight: 1.5, starvation_weight: 3.0, cs_penalty: 2.0, pre_cost: 3.0 },
        'Real-Time': { avg_wait_weight: 0.5, fairness_weight: 0.2, starvation_weight: 5.0, cs_penalty: 1.0, pre_cost: 0.5 },
    };
    const weights = MOCK_COST_WEIGHTS[mode];

    // --- Simulate Human Result ---
    const humanSimResult = simulateScheduling(finalHumanSchedule, 'FCFS', mode); // Treat manual as FCFS on given order
    const humanResult = {
        algorithm: 'Manual',
        metrics: {
            average_waiting_time: humanSimResult.metrics.average_waiting_time,
            max_waiting_time: humanSimResult.metrics.max_waiting_time,
            waiting_time_variance: humanSimResult.metrics.waiting_time_variance,
            context_switches: humanSimResult.metrics.context_switches,
            preemption_count: humanSimResult.metrics.preemption_count,
        },
        gantt_chart_log: humanSimResult.gantt_chart_log,
        completed_processes: humanSimResult.completed_processes, // Keep for explanation logic
    };
    humanResult.metrics.final_score = (
        weights.avg_wait_weight * humanResult.metrics.average_waiting_time +
        weights.fairness_weight * humanResult.metrics.waiting_time_variance +
        weights.starvation_weight * humanResult.metrics.max_waiting_time +
        weights.cs_penalty * humanResult.metrics.context_switches +
        weights.pre_cost * humanResult.metrics.preemption_count +
        (mode === 'Real-Time' ? humanResult.completed_processes.filter(p => p.priority === 1 && p.waiting_time > 10).length * 1000 : 0) // Mock critical case penalty
    );

    // --- Simulate AI Result ---
    // For mock, AI picks a "good" algorithm. In Real-Time, it would use the DT regressor.
    let aiAlgo = 'FCFS'; // Default
    if (mode === 'Efficiency') aiAlgo = 'SJF';
    else if (mode === 'Fairness') aiAlgo = (Math.random() > 0.5 ? 'RR' : 'FCFS');
    else if (mode === 'Real-Time') aiAlgo = 'Priority'; // AI prioritizes critical cases

    let aiScheduleOptimized = JSON.parse(JSON.stringify(humanSchedule)); // Deep copy to prevent original modification
    if (aiAlgo === 'SJF' || aiAlgo === 'SRTF') aiScheduleOptimized.sort((a,b) => a.burst_time - b.burst_time);
    else if (aiAlgo === 'Priority') aiScheduleOptimized.sort((a,b) => a.priority - b.priority);

    const aiSimResult = simulateScheduling(aiScheduleOptimized, aiAlgo, mode);
    const aiResult = {
        algorithm: aiAlgo,
        metrics: {
            average_waiting_time: aiSimResult.metrics.average_waiting_time,
            max_waiting_time: aiSimResult.metrics.max_waiting_time,
            waiting_time_variance: aiSimResult.metrics.waiting_time_variance,
            context_switches: aiSimResult.metrics.context_switches,
            preemption_count: aiSimResult.metrics.preemption_count,
        },
        gantt_chart_log: aiSimResult.gantt_chart_log,
        completed_processes: aiSimResult.completed_processes,
    };
    aiResult.metrics.final_score = (
        weights.avg_wait_weight * aiResult.metrics.average_waiting_time +
        weights.fairness_weight * aiResult.metrics.waiting_time_variance +
        weights.starvation_weight * aiResult.metrics.max_waiting_time +
        weights.cs_penalty * aiResult.metrics.context_switches +
        weights.pre_cost * aiResult.metrics.preemption_count +
        (mode === 'Real-Time' ? aiResult.completed_processes.filter(p => p.priority === 1 && p.waiting_time > 10).length * 1000 : 0) // Mock critical case penalty
    );

    // Determine difficulty
    let difficulty;
    switch(mode) {
        case 'Efficiency': difficulty = 'Hard'; break; // As per updated rule
        case 'Fairness': difficulty = 'Medium'; break;
        case 'Real-Time': difficulty = 'Hard'; break;
        default: difficulty = 'Medium';
    }

    // Generate intuitive explanations
    let explanation = "";
    if (humanResult.metrics.final_score < aiResult.metrics.final_score) {
        explanation = "You beat the AI! Excellent work managing the workload efficiently.";
    } else if (humanResult.metrics.final_score > aiResult.metrics.final_score) {
        let humanProblems = [];
        let aiStrengths = [];

        // Check for specific penalties based on mode
        if (mode === 'Efficiency') {
            if (humanResult.metrics.average_waiting_time > aiResult.metrics.average_waiting_time * 1.5) {
                humanProblems.push("your total waiting time was too high");
            }
            if (humanResult.metrics.context_switches > aiResult.metrics.context_switches * 2) {
                humanProblems.push("you had too many context switches");
            }
        } else if (mode === 'Fairness') {
            if (humanResult.metrics.max_waiting_time > aiResult.metrics.max_waiting_time * 1.5) {
                humanProblems.push("you made some processes wait too long (starvation)");
            }
            if (humanResult.metrics.waiting_time_variance > aiResult.metrics.waiting_time_variance * 1.5) {
                 humanProblems.push("your scheduling was not fair enough");
            }
        } else if (mode === 'Real-Time') {
            const humanCriticalDelayed = humanResult.completed_processes.filter(p => p.priority === 1 && p.waiting_time > 10).length;
            if (humanCriticalDelayed > 0) {
                 humanProblems.push(`you delayed ${humanCriticalDelayed} critical case(s)`);
            }
            if (humanResult.metrics.average_waiting_time > aiResult.metrics.average_waiting_time * 1.5) {
                humanProblems.push("your overall treatment time was inefficient for critical cases");
            }
        }

        if (humanProblems.length > 0) {
            explanation += "The AI won because " + humanProblems.join(" and ") + ". ";
        } else {
            explanation += "The AI won this round. ";
        }
        
        // AI Strengths (general)
        if (aiResult.metrics.average_waiting_time < humanResult.metrics.average_waiting_time * 0.8) {
            aiStrengths.push("the AI handled the overall waiting time better");
        }
        if (aiResult.metrics.context_switches < humanResult.metrics.context_switches * 0.5) {
            aiStrengths.push("the AI optimized for fewer context switches");
        }
        if (aiResult.metrics.max_waiting_time < humanResult.metrics.max_waiting_time * 0.8) {
            aiStrengths.push("the AI prevented starvation more effectively");
        }
        
        if (aiStrengths.length > 0) {
            explanation += "Specifically, " + aiStrengths.join(" and ") + ".";
        } else if (explanation === "The AI won this round. ") {
            explanation += "It seems the AI found a slightly more optimal path.";
        }

    } else {
        explanation = "It's a tie! You and the AI performed equally well.";
    }

    // Add some random coloring logic for results (existing logic)
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
        aiResult.metrics.switches_are_optimal = true; // The AI is good at this
        humanResult.metrics.switches_are_optimal = false;
    }

    return {
        human: humanResult,
        ai: aiResult,
        explanation: explanation,
        difficulty: difficulty,
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