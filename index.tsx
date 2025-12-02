import React, { useState, useEffect, useRef, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, Type } from "@google/genai";

// --- Constants & Types ---

const GRID_W = 25;
const GRID_H = 15;
const CELL_SIZE = 34;
const TICK_RATE_MS = 100;
const MOVEMENT_SPEED = 0.15;
const DEADLOCK_THRESHOLD_TICKS = 300; // 30 seconds

type CellType = 'empty' | 'track' | 'station';

interface Cell {
  x: number;
  y: number;
  type: CellType;
  stationName?: string;
  id: string; // "x-y"
}

interface Train {
  id: string;
  name: string;
  fromId: string;
  toId: string;
  path: string[];
  currentPathIndex: number;
  progress: number;
  state: 'moving' | 'waiting' | 'arrived';
  priority: number;
  type: 'manual' | 'schedule';
  color: string;
  waitingTime: number; // Ticks spent waiting
}

interface ScheduleItem {
  id: string;
  time: number;
  from: string;
  to: string;
  priority: number;
  dispatched: boolean;
}

interface HistoryItem {
  id: string;
  time: number; // Dispatch time
  from: string;
  to: string;
  priority: number;
  type: 'manual' | 'schedule';
}

// --- Helper Functions ---

const getCellId = (x: number, y: number) => `${x}-${y}`;
const parseCellId = (id: string) => {
  const [x, y] = id.split('-').map(Number);
  return { x, y };
};

// Expanded color palette for uniqueness
const TRAIN_COLORS = [
  '#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981',
  '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#f43f5e',
  '#ec4899', '#14b8a6', '#6366f1', '#a855f7', '#fbbf24',
  '#a3e635', '#22d3ee', '#818cf8', '#c084fc', '#fb7185'
];

// --- A* Pathfinding Logic (Weighted) ---

interface Node {
  id: string;
  x: number;
  y: number;
  g: number; // Cost from start
  h: number; // Heuristic to end
  f: number; // Total cost (g + h)
  parent?: Node;
}

const getHeuristic = (x1: number, y1: number, x2: number, y2: number) => {
  return Math.abs(x1 - x2) + Math.abs(y1 - y2);
};

class PriorityQueue<T extends { f: number }> {
  private items: T[] = [];
  push(item: T) {
    this.items.push(item);
    this.items.sort((a, b) => a.f - b.f);
  }
  pop(): T | undefined {
    return this.items.shift();
  }
  isEmpty() {
    return this.items.length === 0;
  }
}

// --- Component ---

function App() {
  // --- State ---
  const [grid, setGrid] = useState<Record<string, Cell>>({});
  const [editMode, setEditMode] = useState<'track' | 'station'>('track');
  
  const [trains, setTrains] = useState<Train[]>([]);
  const [time, setTime] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [schedule, setSchedule] = useState<ScheduleItem[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  
  const [aiPrompt, setAiPrompt] = useState("");
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [apiKeyMissing, setApiKeyMissing] = useState(false);

  const [dispatchFrom, setDispatchFrom] = useState("");
  const [dispatchTo, setDispatchTo] = useState("");
  const [dispatchPriority, setDispatchPriority] = useState(5);

  const [historyFilterFrom, setHistoryFilterFrom] = useState("");
  const [historyFilterTo, setHistoryFilterTo] = useState("");

  const [hoveredTrainId, setHoveredTrainId] = useState<string | null>(null);
  const [previewPath, setPreviewPath] = useState<string[]>([]);
  const [globalDeadlockWarning, setGlobalDeadlockWarning] = useState(false);

  // Refs
  const gridRef = useRef(grid);
  const trainsRef = useRef(trains);
  const scheduleRef = useRef(schedule);
  const timeRef = useRef(time); // Use ref for stable loop access
  
  gridRef.current = grid;
  trainsRef.current = trains;
  scheduleRef.current = schedule;
  timeRef.current = time;

  const isMouseDown = useRef(false);
  const dragAction = useRef<'track' | 'erase' | null>(null);

  // Initialize Grid
  useEffect(() => {
    const initialGrid: Record<string, Cell> = {};
    for (let y = 0; y < GRID_H; y++) {
      for (let x = 0; x < GRID_W; x++) {
        const id = getCellId(x, y);
        initialGrid[id] = { x, y, type: 'empty', id };
      }
    }
    const addStation = (x: number, y: number, name: string) => {
      const id = getCellId(x, y);
      initialGrid[id] = { ...initialGrid[id], type: 'station', stationName: name };
    };
    const addTrack = (x: number, y: number) => {
      const id = getCellId(x, y);
      initialGrid[id] = { ...initialGrid[id], type: 'track' };
    };

    // Default map
    addStation(2, 7, "西站");
    addStation(22, 7, "东站");
    addStation(12, 2, "北站");
    addStation(12, 12, "南站");

    for(let x=3; x<22; x++) addTrack(x, 7);
    for(let y=3; y<12; y++) if(y !== 7) addTrack(12, y);
    addTrack(12, 7); 
    
    // Loops
    addTrack(11, 6); addTrack(11, 5); addTrack(12, 5); addTrack(13, 5); addTrack(13, 6);
    addTrack(11, 8); addTrack(11, 9); addTrack(12, 9); addTrack(13, 9); addTrack(13, 8);

    setGrid(initialGrid);
  }, []);

  useEffect(() => {
    if (!process.env.API_KEY) {
      setApiKeyMissing(true);
    }
  }, []);

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      isMouseDown.current = false;
      dragAction.current = null;
    };
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  // --- Helpers ---

  const getUniqueColor = (currentTrains: Train[]) => {
    const usedColors = new Set(currentTrains.map(t => t.color));
    const available = TRAIN_COLORS.filter(c => !usedColors.has(c));
    if (available.length > 0) {
      return available[Math.floor(Math.random() * available.length)];
    }
    return TRAIN_COLORS[Math.floor(Math.random() * TRAIN_COLORS.length)];
  };

  const resetSystem = () => {
    setIsRunning(false);
    setTime(0);
    setTrains([]);
    setSchedule([]);
    setHistory([]);
    setGlobalDeadlockWarning(false);
    setDispatchFrom("");
    setDispatchTo("");
  };

  // --- Pathfinding ---

  const findPathAStar = (
    startId: string, 
    endId: string, 
    currentGrid: Record<string, Cell>,
    currentTrains: Train[] = [], 
    ignoreTrainId?: string,
    reservedCells?: Set<string>
  ): string[] | null => {
    
    if (!currentGrid[startId] || !currentGrid[endId]) return null;

    const occupiedCells = new Set<string>();
    const nearCongestionCells = new Set<string>();

    currentTrains.forEach(t => {
      if (t.id === ignoreTrainId) return;
      const tPos = t.path[t.currentPathIndex];
      occupiedCells.add(tPos);
      
      // Add adjacent for "near congestion" penalty
      const pos = parseCellId(tPos);
      const neighbors = [[0,1], [0,-1], [1,0], [-1,0]];
      neighbors.forEach(([dx, dy]) => {
         nearCongestionCells.add(getCellId(pos.x + dx, pos.y + dy));
      });
    });

    const startNode = parseCellId(startId);
    const endNode = parseCellId(endId);

    const openSet = new PriorityQueue<Node>();
    const startH = getHeuristic(startNode.x, startNode.y, endNode.x, endNode.y);
    openSet.push({ 
      id: startId, 
      x: startNode.x, 
      y: startNode.y, 
      g: 0, 
      h: startH, 
      f: startH 
    });

    const visited = new Map<string, number>();
    visited.set(startId, 0);

    const cameFrom = new Map<string, Node>();

    while (!openSet.isEmpty()) {
      const current = openSet.pop()!;

      if (current.id === endId) {
        const path: string[] = [];
        let curr: Node | undefined = current;
        while (curr) {
          path.unshift(curr.id);
          curr = curr.parent;
        }
        return path;
      }

      const neighbors = [
        { nx: current.x, ny: current.y - 1 },
        { nx: current.x, ny: current.y + 1 },
        { nx: current.x - 1, ny: current.y },
        { nx: current.x + 1, ny: current.y }
      ];

      for (const { nx, ny } of neighbors) {
        const nid = getCellId(nx, ny);
        const neighborCell = currentGrid[nid];

        if (neighborCell && (neighborCell.type === 'track' || neighborCell.type === 'station')) {
          let moveCost = 1;
          
          // --- Dynamic Weighting Strategy ---
          // 1. Occupied by a train physically: Very high cost (avoid collision)
          if (occupiedCells.has(nid) && nid !== endId) {
             moveCost += 500;
          }

          // 2. Reserved by another train's critical path: Extremely high cost (avoid deadlock)
          if (reservedCells && reservedCells.has(nid) && nid !== endId) {
            moveCost += 1000;
          }

          // 3. Near other trains: Small penalty to prefer clearance
          if (nearCongestionCells.has(nid)) {
            moveCost += 10;
          }

          const newG = current.g + moveCost;
          
          if (!visited.has(nid) || newG < visited.get(nid)!) {
            visited.set(nid, newG);
            const h = getHeuristic(nx, ny, endNode.x, endNode.y);
            
            const neighborNode: Node = {
              id: nid,
              x: nx,
              y: ny,
              g: newG,
              h: h,
              f: newG + h,
              parent: current
            };
            openSet.push(neighborNode);
          }
        }
      }
    }
    // No path found
    return null;
  };

  const getNextStationIndex = (path: string[], currentIndex: number, currentGrid: Record<string, Cell>): number => {
    for (let i = currentIndex + 1; i < path.length; i++) {
      if (currentGrid[path[i]].type === 'station') {
        return i;
      }
    }
    return path.length - 1; 
  };

  const isPathClear = (
    pathSegment: string[], 
    myTrainId: string, 
    allTrains: Train[],
    reservedCells: Set<string>
  ): boolean => {
    for (const cellId of pathSegment) {
      // Check Reservations
      if (reservedCells.has(cellId)) return false;

      // Check Physical Presence
      const occupier = allTrains.find(t => {
        if (t.id === myTrainId) return false;
        const tPos = t.path[t.currentPathIndex];
        // Occupied if train is at cell OR moving into cell
        const isAtCell = tPos === cellId;
        const isMovingToCell = (t.state === 'moving') && 
                               (t.currentPathIndex + 1 < t.path.length) &&
                               (t.path[t.currentPathIndex + 1] === cellId);
        return isAtCell || isMovingToCell;
      });

      if (occupier) return false;
    }
    return true;
  };

  const dispatchTrain = (fromName: string, toName: string, priority: number, isManual: boolean) => {
    if (fromName === toName) return; // Prevent same station dispatch

    const stationsList = (Object.values(gridRef.current) as Cell[]).filter(c => c.type === 'station');
    const from = stationsList.find(s => s.stationName === fromName);
    const to = stationsList.find(s => s.stationName === toName);
    
    if (from && to) {
      // Initial pathfinding (ignoring strict reservations to just find A path)
      const path = findPathAStar(from.id, to.id, gridRef.current, trainsRef.current);
      if (path) {
        const trainId = `${isManual ? 'man' : 'sch'}-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`;
        const newTrain: Train = {
          id: trainId,
          name: `${isManual ? '临' : 'T'}-${trainId.slice(-4)}`,
          fromId: from.id,
          toId: to.id,
          path,
          currentPathIndex: 0,
          progress: 0,
          state: 'waiting',
          priority: priority,
          type: isManual ? 'manual' : 'schedule',
          color: getUniqueColor(trainsRef.current),
          waitingTime: 0
        };
        setTrains(prev => [...prev, newTrain]);
        
        // Add to history
        setHistory(prev => [{
          id: `hist-${Date.now()}-${Math.random()}`,
          time: timeRef.current,
          from: fromName,
          to: toName,
          priority: priority,
          type: isManual ? 'manual' : 'schedule'
        }, ...prev]);

      } else {
        if (isManual) alert(`无法规划路径：可能所有轨道已断开。`);
      }
    }
  };

  // --- AI ---
  const generateScheduleWithAI = async () => {
    if (!process.env.API_KEY || !aiPrompt) return;
    setIsAiLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const stationNames = (Object.values(grid) as Cell[])
        .filter(c => c.type === 'station')
        .map(c => c.stationName)
        .join(", ");

      const systemInstruction = `
        You are a train scheduler assistant.
        The current map has these stations: ${stationNames}.
        The current simulation time is ${time}.
        
        The user will describe a traffic scenario in Chinese or English.
        Generate a JSON schedule.
        Start scheduling trains from time = ${time + 5} or later.
        Priority 1 (Low) to 10 (High).
        IMPORTANT: Use EXACT station names from the provided list.
      `;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: aiPrompt,
        config: {
          systemInstruction: systemInstruction,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                timeOffset: { type: Type.INTEGER, description: "How many seconds from NOW to start." },
                from: { type: Type.STRING, description: "Exact station name from the list." },
                to: { type: Type.STRING, description: "Exact station name from the list." },
                priority: { type: Type.INTEGER, description: "Priority 1-10" }
              },
              required: ["timeOffset", "from", "to", "priority"]
            }
          }
        }
      });

      const generatedData = JSON.parse(response.text);
      
      const newScheduleItems: ScheduleItem[] = generatedData.map((item: any, idx: number) => ({
        id: `sched-ai-${Date.now()}-${idx}`,
        time: time + item.timeOffset,
        from: item.from,
        to: item.to,
        priority: item.priority,
        dispatched: false
      }));

      setSchedule(prev => [...prev, ...newScheduleItems].sort((a, b) => a.time - b.time));
      setAiPrompt("");
    } catch (e) {
      console.error("AI Error:", e);
      alert("生成时刻表失败，请重试。");
    } finally {
      setIsAiLoading(false);
    }
  };

  // --- Simulation Loop ---

  useEffect(() => {
    if (!isRunning) return;

    // Use a stable loop that references current state via refs, instead of recreating interval on every time change
    const tick = setInterval(() => {
      // 1. Advance Time
      const currentTime = timeRef.current + 1;
      setTime(currentTime);
      
      // 2. Dispatch New Trains (from Schedule)
      const currentSched = scheduleRef.current;
      const updatedSched = currentSched.map(item => {
        if (!item.dispatched && item.time <= currentTime) {
          dispatchTrain(item.from, item.to, item.priority, false);
          return { ...item, dispatched: true };
        }
        return item;
      });
      
      if (updatedSched.some((s, i) => s.dispatched !== currentSched[i].dispatched)) {
        setSchedule(updatedSched);
      }

      // 3. Process Trains
      let activeTrains = [...trainsRef.current];
      const globalReservations = new Set<string>();
      
      activeTrains.sort((a, b) => b.priority - a.priority);

      const nextTrainsState = activeTrains.map(train => {
        let { currentPathIndex, progress, state, path, waitingTime } = train;
        if (state === 'arrived') return train;
        const currentCellId = path[currentPathIndex];
        
        // MOVING -> WAITING
        if (state === 'moving') {
          waitingTime = 0; // Reset waiting time if moving
          progress += MOVEMENT_SPEED;
          if (progress >= 1.0) {
            progress = 0;
            currentPathIndex++;
            state = 'waiting';
          }
        }

        // WAITING Logic
        if (state === 'waiting') {
           if (currentPathIndex >= path.length - 1) {
             return { ...train, state: 'arrived', progress: 0, currentPathIndex, waitingTime: 0 };
           }

           const isAtStation = gridRef.current[path[currentPathIndex]]?.type === 'station';

           // Determine the segment we MUST reserve to move safely
           let nextStationIdx = getNextStationIndex(path, currentPathIndex, gridRef.current);
           let segmentToReserve = path.slice(currentPathIndex + 1, nextStationIdx + 1);
           
           if (isAtStation) {
              const blocked = !isPathClear(segmentToReserve, train.id, trainsRef.current, globalReservations);
              if (blocked) {
                // Reroute Attempt: Try to find ANY other valid path accounting for current reservations
                const reroutePath = findPathAStar(
                  path[currentPathIndex], 
                  train.toId, 
                  gridRef.current, 
                  trainsRef.current,
                  train.id,
                  globalReservations
                );

                // If a path is found, adopt it immediately
                // We check if it exists and has length > 0.
                // We do NOT check "detourPath[1] !== current" because even if the first step is the same,
                // the subsequent steps might differ, or the reservation logic needs a refreshed path.
                if (reroutePath && reroutePath.length > 0) {
                   path = [...path.slice(0, currentPathIndex), ...reroutePath.slice(1)];
                   
                   // Recalculate reservation needs based on NEW path
                   nextStationIdx = getNextStationIndex(path, currentPathIndex, gridRef.current);
                   segmentToReserve = path.slice(currentPathIndex + 1, nextStationIdx + 1);
                }
              }
           }

           // Try to move (on old path or new rerouted path)
           if (isPathClear(segmentToReserve, train.id, trainsRef.current, globalReservations)) {
             segmentToReserve.forEach(cid => globalReservations.add(cid));
             state = 'moving';
             waitingTime = 0;
           } else {
             // Still blocked
             globalReservations.add(currentCellId); // Keep reserving current spot
             waitingTime++;
           }
        } 
        else if (state === 'moving') {
           // Maintain reservation for moving trains
           const nextStationIdx = getNextStationIndex(path, currentPathIndex, gridRef.current);
           const segmentToKeep = path.slice(currentPathIndex + 1, nextStationIdx + 1);
           segmentToKeep.forEach(cid => globalReservations.add(cid));
        }

        return { ...train, currentPathIndex, progress, state, path, waitingTime };
      });

      const runningTrains = nextTrainsState.filter(t => t.state !== 'arrived');
      
      if (runningTrains.length > 1 && runningTrains.every(t => t.state === 'waiting' && t.waitingTime > 20)) {
         setGlobalDeadlockWarning(true);
      } else {
         setGlobalDeadlockWarning(false);
      }

      setTrains(runningTrains);

    }, TICK_RATE_MS);

    return () => clearInterval(tick);
  }, [isRunning]); // Removed 'time' dependency to stabilize the loop


  // --- Event Handlers (Editor) ---

  const modifyCell = (x: number, y: number, action: 'track' | 'station' | 'erase') => {
    setGrid(prev => {
      const id = getCellId(x, y);
      const cell = prev[id];
      if (!cell) return prev;

      if (action === 'erase') {
        if (cell.type === 'empty') return prev;
        return { ...prev, [id]: { ...cell, type: 'empty', stationName: undefined } };
      }

      if (action === 'track') {
        if (cell.type === 'track') return prev;
        return { ...prev, [id]: { ...cell, type: 'track', stationName: undefined } };
      }

      if (action === 'station') {
        if (cell.type === 'station') return prev;
        return { ...prev, [id]: { ...cell, type: 'station', stationName: `站${x}-${y}` } };
      }
      return prev;
    });
  };

  const handleCellMouseDown = (x: number, y: number, e: React.MouseEvent) => {
    if (isRunning) return;
    if (e.button === 2) e.preventDefault();
    isMouseDown.current = true;
    const isRight = e.button === 2;
    if (isRight) {
      dragAction.current = 'erase';
      modifyCell(x, y, 'erase');
    } else {
      if (editMode === 'track') {
        dragAction.current = 'track';
        modifyCell(x, y, 'track');
      } else {
        dragAction.current = null;
        modifyCell(x, y, 'station');
      }
    }
  };

  const handleCellMouseEnter = (x: number, y: number) => {
    if (isRunning || !isMouseDown.current || !dragAction.current) return;
    modifyCell(x, y, dragAction.current);
  };

  const handleCellDoubleClick = (x: number, y: number) => {
    if (isRunning) return;
    const id = getCellId(x, y);
    const cell = grid[id];
    if (cell.type === 'station') {
      const newName = prompt("重命名车站:", cell.stationName);
      if (newName && newName.trim() !== "") {
        setGrid(prev => ({
           ...prev,
           [id]: { ...cell, stationName: newName.trim() }
        }));
      }
    }
  };

  // Preview logic
  const stations = useMemo(() => (Object.values(grid) as Cell[]).filter(c => c.type === 'station'), [grid]);
  useEffect(() => {
    if (!dispatchFrom || !dispatchTo || dispatchFrom === dispatchTo) {
      setPreviewPath([]);
      return;
    }
    const s1 = stations.find(s => s.stationName === dispatchFrom);
    const s2 = stations.find(s => s.stationName === dispatchTo);
    if (s1 && s2) {
      const path = findPathAStar(s1.id, s2.id, grid, trains);
      setPreviewPath(path || []);
    } else {
      setPreviewPath([]);
    }
  }, [dispatchFrom, dispatchTo, grid, stations, trains]);

  const handleManualDispatch = () => {
    dispatchTrain(dispatchFrom, dispatchTo, dispatchPriority, true);
  };

  // --- Render Helpers ---

  const trackLines = useMemo(() => {
    const lines: React.ReactElement[] = [];
    (Object.values(grid) as Cell[]).forEach(cell => {
      if (cell.type === 'empty') return;
      const neighbors = [{ dx: 1, dy: 0 }, { dx: 0, dy: 1 }];
      neighbors.forEach(({ dx, dy }) => {
        const nx = cell.x + dx;
        const ny = cell.y + dy;
        const nid = getCellId(nx, ny);
        const neighbor = grid[nid];
        if (neighbor && neighbor.type !== 'empty') {
          lines.push(
            <line
              key={`${cell.id}-${nid}`}
              x1={cell.x * CELL_SIZE + CELL_SIZE/2}
              y1={cell.y * CELL_SIZE + CELL_SIZE/2}
              x2={nx * CELL_SIZE + CELL_SIZE/2}
              y2={ny * CELL_SIZE + CELL_SIZE/2}
              stroke="#475569" 
              strokeWidth="4"
              strokeLinecap="round"
            />
          );
        }
      });
    });
    return lines;
  }, [grid]);

  // Visualize path for ALL active trains
  const activeTrainPaths = useMemo(() => {
    return trains.map(t => {
      // Draw from current position to end
      const remainingPath = t.path.slice(Math.max(0, t.currentPathIndex));
      if (remainingPath.length < 2) return null;
      
      const points = remainingPath.map(id => {
        const { x, y } = parseCellId(id);
        return `${x * CELL_SIZE + CELL_SIZE/2},${y * CELL_SIZE + CELL_SIZE/2}`;
      }).join(" ");

      return (
        <polyline
          key={`path-${t.id}`}
          points={points}
          fill="none"
          stroke={t.color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeOpacity="0.15" // Subtle path indication
        />
      );
    });
  }, [trains]);

  const previewPolyline = useMemo(() => {
    if (previewPath.length < 2) return null;
    const points = previewPath.map(id => {
      const { x, y } = parseCellId(id);
      return `${x * CELL_SIZE + CELL_SIZE/2},${y * CELL_SIZE + CELL_SIZE/2}`;
    }).join(" ");
    return (
      <polyline
        points={points}
        fill="none"
        stroke="#eab308"
        strokeWidth="3"
        strokeDasharray="6 4"
        strokeLinecap="round"
        strokeOpacity="0.8"
        className="animate-pulse"
      />
    );
  }, [previewPath]);

  const hoveredTrainInfo = useMemo(() => {
    if (!hoveredTrainId) return null;
    const train = trains.find(t => t.id === hoveredTrainId);
    if (!train) return null;
    const currentPos = parseCellId(train.path[train.currentPathIndex]);
    const nextPos = train.currentPathIndex < train.path.length - 1 
      ? parseCellId(train.path[train.currentPathIndex + 1]) 
      : currentPos;
    const screenX = (currentPos.x + (nextPos.x - currentPos.x) * train.progress) * CELL_SIZE + CELL_SIZE/2;
    const screenY = (currentPos.y + (nextPos.y - currentPos.y) * train.progress) * CELL_SIZE + CELL_SIZE/2;
    const fromStation = stations.find(s => s.id === train.fromId)?.stationName || "Unknown";
    const toStation = stations.find(s => s.id === train.toId)?.stationName || "Unknown";
    return { ...train, screenX, screenY, fromStation, toStation };
  }, [hoveredTrainId, trains, stations]);

  const pendingTrains = schedule.filter(s => !s.dispatched);
  
  // Deduplicate and Filter history
  const uniqueHistory = useMemo(() => {
    const seen = new Set<string>();
    // History is newest first. We want to show unique routes, keeping the parameters of the most recent dispatch.
    return history.filter(item => {
      // 1. Apply Text Filters
      if (historyFilterFrom && item.from !== historyFilterFrom) return false;
      if (historyFilterTo && item.to !== historyFilterTo) return false;

      // 2. Apply Deduplication (Route based)
      const key = `${item.from}->${item.to}`;
      if (seen.has(key)) return false;
      
      seen.add(key);
      return true;
    });
  }, [history, historyFilterFrom, historyFilterTo]);

  return (
    <div className="flex flex-row h-full bg-slate-900 text-slate-100 font-sans select-none">
      {/* LEFT PANEL: MAP */}
      <div className="flex-1 relative overflow-hidden flex flex-col items-center justify-center p-6 bg-slate-950">
        
        {/* Header Indicators */}
        <div className="absolute top-4 left-6 flex items-center space-x-4 bg-slate-900/80 backdrop-blur-md p-3 rounded-xl border border-slate-800 shadow-xl z-10">
           <div className="flex flex-col">
             <span className="text-xs text-slate-400 font-medium">仿真时间</span>
             <span className="text-xl font-mono text-cyan-400">{time}s</span>
           </div>
           <div className="h-8 w-px bg-slate-700"></div>
           <div className="flex flex-col">
             <span className="text-xs text-slate-400 font-medium">在途列车</span>
             <span className="text-xl font-mono text-emerald-400">{trains.length}</span>
           </div>
           {globalDeadlockWarning && (
             <>
               <div className="h-8 w-px bg-slate-700"></div>
               <div className="flex items-center space-x-2 text-rose-500 animate-pulse font-bold">
                 <span className="text-2xl">⚠</span>
                 <span>系统拥堵警告</span>
               </div>
             </>
           )}
        </div>

        <div 
          className="relative bg-slate-900/50 shadow-2xl border border-slate-800 rounded-lg backdrop-blur-sm transition-all duration-300"
          style={{ width: GRID_W * CELL_SIZE, height: GRID_H * CELL_SIZE }}
          onContextMenu={(e) => e.preventDefault()}
        >
          <div className="absolute inset-0 opacity-10 pointer-events-none" 
            style={{ 
               backgroundImage: `linear-gradient(#475569 1px, transparent 1px), linear-gradient(90deg, #475569 1px, transparent 1px)`,
               backgroundSize: `${CELL_SIZE}px ${CELL_SIZE}px`
            }}
          ></div>

          {/* Grid Cells - Set pointer-events-none when running to allow SVG interaction */}
          <div className={`absolute inset-0 grid z-10 ${isRunning ? 'pointer-events-none' : 'pointer-events-auto'}`} style={{ 
            gridTemplateColumns: `repeat(${GRID_W}, 1fr)`,
            gridTemplateRows: `repeat(${GRID_H}, 1fr)`
          }}>
            {Array.from({ length: GRID_H * GRID_W }).map((_, i) => {
              const x = i % GRID_W;
              const y = Math.floor(i / GRID_W);
              const id = getCellId(x, y);
              const cell = grid[id];
              return (
                <div 
                  key={id}
                  className={`border border-transparent hover:border-white/10`}
                  style={{ cursor: isRunning ? 'default' : 'crosshair' }}
                  onMouseDown={(e) => handleCellMouseDown(x, y, e)}
                  onMouseEnter={() => handleCellMouseEnter(x, y)}
                  onDoubleClick={() => handleCellDoubleClick(x, y)}
                />
              );
            })}
          </div>

          <svg className="absolute inset-0 pointer-events-none z-0" width="100%" height="100%">
            {trackLines}
            {activeTrainPaths}
            {previewPolyline}

            {stations.map(s => (
              <g key={s.id}>
                <rect x={s.x * CELL_SIZE + 2} y={s.y * CELL_SIZE + 4} width={CELL_SIZE - 4} height={CELL_SIZE - 8} fill="#000" opacity="0.3" rx="4" />
                <rect x={s.x * CELL_SIZE + 2} y={s.y * CELL_SIZE + 2} width={CELL_SIZE - 4} height={CELL_SIZE - 8} fill="#3b82f6" stroke="#60a5fa" strokeWidth="1" rx="4" />
                <g transform={`translate(${s.x * CELL_SIZE + CELL_SIZE/2}, ${s.y * CELL_SIZE - 8})`}>
                  <rect x="-20" y="-8" width="40" height="14" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
                  <text x="0" y="2" textAnchor="middle" fill="#e2e8f0" fontSize="9" fontWeight="bold" dominantBaseline="middle">{s.stationName}</text>
                </g>
              </g>
            ))}
            
            {trains.map(t => {
              const currentPos = parseCellId(t.path[t.currentPathIndex]);
              const nextPos = t.currentPathIndex < t.path.length - 1 
                ? parseCellId(t.path[t.currentPathIndex + 1]) 
                : currentPos;
              const screenX = (currentPos.x + (nextPos.x - currentPos.x) * t.progress) * CELL_SIZE + CELL_SIZE/2;
              const screenY = (currentPos.y + (nextPos.y - currentPos.y) * t.progress) * CELL_SIZE + CELL_SIZE/2;
              const isHovered = hoveredTrainId === t.id;
              const isStuck = t.waitingTime > DEADLOCK_THRESHOLD_TICKS;

              return (
                <g 
                  key={t.id} 
                  style={{ transition: `all ${TICK_RATE_MS}ms linear` }}
                  className="pointer-events-auto cursor-help"
                  onMouseEnter={() => setHoveredTrainId(t.id)}
                  onMouseLeave={() => setHoveredTrainId(null)}
                >
                  {/* Warning Icon if Stuck */}
                  {isStuck && (
                    <text x={screenX} y={screenY - 24} textAnchor="middle" fontSize="16" className="animate-bounce">⚠️</text>
                  )}

                  {isHovered && <circle cx={screenX} cy={screenY} r={CELL_SIZE * 0.8} fill={t.color} opacity="0.3" className="animate-pulse" />}
                  <circle cx={screenX} cy={screenY} r={CELL_SIZE * 0.5} fill={t.color} opacity="0.2" />
                  <circle 
                    cx={screenX} 
                    cy={screenY} 
                    r={CELL_SIZE * 0.35} 
                    fill={t.state === 'waiting' ? '#ef4444' : t.color}
                    stroke="white"
                    strokeWidth={isHovered ? "3" : "2"}
                    className="shadow-lg drop-shadow-md"
                  />
                  <text x={screenX} y={screenY - 14} textAnchor="middle" fill="white" fontSize="10" fontWeight="bold" filter="drop-shadow(0px 1px 1px rgba(0,0,0,0.8))">{t.name}</text>
                </g>
              );
            })}
          </svg>
          
          {hoveredTrainInfo && (
            <div 
              className="absolute z-50 pointer-events-none flex flex-col gap-1 p-2 bg-slate-800/90 backdrop-blur text-slate-100 rounded-lg shadow-2xl border border-slate-600 text-[10px]"
              style={{
                left: hoveredTrainInfo.screenX,
                top: hoveredTrainInfo.screenY + 20,
                transform: 'translate(-50%, 0)',
                minWidth: '120px'
              }}
            >
              <div className="font-bold text-xs text-white border-b border-slate-600 pb-1 mb-1 flex justify-between items-center">
                <span>{hoveredTrainInfo.name}</span>
                <span className="px-1.5 rounded-full" style={{backgroundColor: hoveredTrainInfo.color}}></span>
              </div>
              <div className="flex justify-between"><span>状态:</span> <span className={hoveredTrainInfo.state === 'waiting' ? 'text-red-400' : 'text-emerald-400'}>{hoveredTrainInfo.state === 'waiting' ? '等待中' : '运行中'}</span></div>
              <div className="flex justify-between"><span>等待:</span> <span className={`${hoveredTrainInfo.waitingTime > 20 ? 'text-rose-400 font-bold' : 'text-slate-400'}`}>{hoveredTrainInfo.waitingTime} Ticks</span></div>
              <div className="flex justify-between"><span>优先级:</span> <span className="text-orange-300 font-mono">P{hoveredTrainInfo.priority}</span></div>
              <div className="flex justify-between"><span>路线:</span> <span className="text-slate-400">{hoveredTrainInfo.fromStation} → {hoveredTrainInfo.toStation}</span></div>
            </div>
          )}
        </div>
        
        <div className="absolute bottom-6 left-6 flex space-x-4 text-xs text-slate-500 bg-slate-900/80 p-2 rounded-lg border border-slate-800">
          <div className="flex items-center space-x-1"><span className="w-3 h-3 bg-blue-500 rounded"></span><span>车站</span></div>
          <div className="flex items-center space-x-1"><span className="w-3 h-3 bg-slate-600 rounded"></span><span>轨道</span></div>
          <div className="border-l border-slate-700 pl-2">
            操作: 左键拖拽铺路/点击建站 | 右键拖拽擦除 | 悬浮查看车辆详情 | 双击重命名
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col shadow-2xl z-20">
        <div className="p-6 border-b border-slate-800 bg-slate-900">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
            列车调度中心
          </h1>
          <p className="text-xs text-slate-500 mt-1">Advanced Train Dispatch System</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">

          <div className="flex space-x-2">
            <button 
              onClick={() => setIsRunning(!isRunning)}
              className={`flex-1 py-2 px-4 rounded-lg font-bold text-white shadow-lg transition-all active:scale-95 text-sm
                ${isRunning ? 'bg-rose-600 hover:bg-rose-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}
            >
              {isRunning ? "⏹ 停止仿真" : "▶ 启动仿真"}
            </button>
            <button 
              onClick={resetSystem}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-bold shadow-lg text-sm active:scale-95"
            >
              ↺ 重置
            </button>
          </div>
          
          <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg text-xs text-slate-400 border border-slate-700">
            <button className={`flex-1 py-1 rounded ${editMode === 'track' ? 'bg-slate-600 text-white' : 'hover:bg-slate-700'}`} onClick={() => setEditMode('track')}>
              轨道模式
            </button>
            <button className={`flex-1 py-1 rounded ${editMode === 'station' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700'}`} onClick={() => setEditMode('station')}>
              车站模式
            </button>
          </div>

          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">临时发车指令</h2>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <select className="bg-slate-900 border border-slate-600 text-xs rounded-lg p-2 text-slate-200 outline-none" value={dispatchFrom} onChange={(e) => setDispatchFrom(e.target.value)}>
                  <option value="">始发站...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
                <select className="bg-slate-900 border border-slate-600 text-xs rounded-lg p-2 text-slate-200 outline-none" value={dispatchTo} onChange={(e) => setDispatchTo(e.target.value)}>
                  <option value="">终点站...</option>
                  {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                </select>
              </div>
              <input type="range" min="1" max="10" value={dispatchPriority} onChange={(e) => setDispatchPriority(Number(e.target.value))} className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"/>
              <button onClick={handleManualDispatch} disabled={previewPath.length === 0 || dispatchFrom === dispatchTo} className="w-full bg-orange-600 hover:bg-orange-500 disabled:bg-slate-700 text-white py-2 rounded-lg text-xs font-bold shadow-lg">立即发车</button>
            </div>
          </div>

          {/* Pending Trains Queue */}
          <div className="bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden min-h-[100px]">
             <div className="p-3 bg-slate-800/80 border-b border-slate-700 flex justify-between items-center">
               <label className="text-xs font-bold text-yellow-500 uppercase flex items-center gap-1">
                 <span>⏳</span> 候车队列 (未发车)
               </label>
               <span className="text-[10px] bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">{pendingTrains.length}</span>
             </div>
             <div className="max-h-32 overflow-y-auto custom-scrollbar p-2 space-y-1">
                {pendingTrains.length === 0 && <p className="text-[10px] text-slate-600 text-center py-2">队列空闲</p>}
                {pendingTrains.map(item => (
                   <div key={item.id} className="flex justify-between items-center text-[10px] p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                      <span>T+{item.time}s | {item.from} → {item.to}</span>
                      <span className="text-yellow-500">Waiting</span>
                   </div>
                ))}
             </div>
          </div>

          {/* Dispatched History & Filtering */}
          <div className="flex-1 flex flex-col min-h-[250px] bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
             <div className="p-3 bg-slate-800/80 border-b border-slate-700 space-y-2">
               <div className="flex justify-between items-center">
                 <label className="text-xs font-bold text-slate-400 uppercase">历史路线 (已去重)</label>
                 <span className="text-[10px] bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">{uniqueHistory.length}</span>
               </div>
               
               {/* Filters */}
               <div className="flex gap-2">
                  <select className="flex-1 bg-slate-900 border border-slate-600 text-[10px] rounded p-1 text-slate-200 outline-none" 
                    value={historyFilterFrom} onChange={(e) => setHistoryFilterFrom(e.target.value)}>
                    <option value="">全部始发...</option>
                    {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                  </select>
                  <select className="flex-1 bg-slate-900 border border-slate-600 text-[10px] rounded p-1 text-slate-200 outline-none" 
                    value={historyFilterTo} onChange={(e) => setHistoryFilterTo(e.target.value)}>
                    <option value="">全部终点...</option>
                    {stations.map(s => <option key={s.id} value={s.stationName}>{s.stationName}</option>)}
                  </select>
               </div>
             </div>

             <div className="overflow-y-auto flex-1 p-2 space-y-1 custom-scrollbar">
               {uniqueHistory.length === 0 && <p className="text-[10px] text-slate-600 text-center py-4">无历史记录</p>}
               {uniqueHistory.map(item => (
                 <div key={item.id} className="flex justify-between items-center text-xs p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-transparent hover:border-slate-600 group transition-all">
                   <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                         <span className="font-mono font-bold text-slate-400">P{item.priority}</span>
                         <span className={`text-[9px] px-1 rounded ${item.type === 'manual' ? 'bg-orange-900/50 text-orange-400' : 'bg-blue-900/50 text-blue-400'}`}>
                           {item.type === 'manual' ? '手动' : '计划'}
                         </span>
                      </div>
                      <span className="text-[10px] mt-0.5">{item.from} ➔ {item.to}</span>
                   </div>
                   <button 
                     onClick={() => dispatchTrain(item.from, item.to, item.priority, true)}
                     className="opacity-0 group-hover:opacity-100 bg-emerald-600 hover:bg-emerald-500 text-white px-2 py-1 rounded text-[10px] transition-opacity"
                     title="使用相同参数再次发车"
                   >
                     ↺ 再发
                   </button>
                 </div>
               ))}
             </div>
          </div>

          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 rounded-xl p-4 border border-indigo-500/30">
            <h2 className="text-xs font-bold text-indigo-300 uppercase tracking-wider mb-2">AI 智能调度</h2>
            <textarea className="w-full bg-slate-900/80 border border-indigo-500/30 rounded-lg p-2 text-xs text-indigo-100 h-16 outline-none resize-none" placeholder="描述场景..." value={aiPrompt} onChange={(e) => setAiPrompt(e.target.value)} />
            <button onClick={generateScheduleWithAI} disabled={isAiLoading || !aiPrompt} className="w-full mt-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white py-1.5 rounded-lg text-xs font-bold">
              {isAiLoading ? "生成中..." : "生成计划"}
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);